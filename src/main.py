"""
增强版FastAPI应用入口
基于server.py的成熟实现，整合完整的API端点和流式处理
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, List, Optional, Union, Literal

import litellm
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .config import settings
from .utils import (
    model_manager, error_handler, retry_manager, streaming_handler,
    HealthChecker, LoggingManager, create_request_id, parse_tool_result_content
)

# 加载环境变量
load_dotenv()

# 配置日志
LoggingManager.setup_logging()
logger = logging.getLogger(__name__)


# ===== Pydantic模型定义 =====
class ContentBlockText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"] = "image"
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any]]


class SystemContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="消息角色")
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]] = Field(..., description="消息内容")


class Tool(BaseModel):
    name: str = Field(..., description="工具名称")
    description: Optional[str] = Field(None, description="工具描述")
    input_schema: Dict[str, Any] = Field(..., description="输入参数架构")


class ThinkingConfig(BaseModel):
    enabled: bool = Field(True, description="是否启用思考模式")


class MessagesRequest(BaseModel):
    model: str = Field(..., description="请求的模型名称")
    max_tokens: int = Field(..., description="最大token数")
    messages: List[Message] = Field(..., description="消息列表")
    system: Optional[Union[str, List[SystemContent]]] = Field(None, description="系统提示")
    stop_sequences: Optional[List[str]] = Field(None, description="停止序列")
    stream: Optional[bool] = Field(False, description="是否启用流式响应")
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0, description="温度参数")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="top_p采样参数")
    top_k: Optional[int] = Field(None, ge=1, description="top_k采样参数")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    tools: Optional[List[Tool]] = Field(None, description="工具列表")
    tool_choice: Optional[Dict[str, Any]] = Field(None, description="工具选择配置")
    thinking: Optional[ThinkingConfig] = Field(None, description="思考配置")
    original_model: Optional[str] = Field(None, description="原始模型名称")

    @field_validator('model')
    @classmethod
    def validate_model_field(cls, v):
        """验证并映射模型名称"""
        original_model = v
        mapped_model, was_mapped = model_manager.validate_and_map_model(v)
        
        logger.debug(f"模型验证: 原始='{original_model}', 映射后='{mapped_model}', 是否映射={was_mapped}")
        
        return mapped_model


class TokenCountRequest(BaseModel):
    model: str = Field(..., description="模型名称")
    messages: List[Message] = Field(..., description="消息列表")
    system: Optional[Union[str, List[SystemContent]]] = Field(None, description="系统提示")
    tools: Optional[List[Tool]] = Field(None, description="工具列表")
    thinking: Optional[ThinkingConfig] = Field(None, description="思考配置")
    original_model: Optional[str] = Field(None, description="原始模型名称")

    @field_validator('model')
    @classmethod
    def validate_model_token_count(cls, v):
        """验证token计数的模型名称"""
        mapped_model, _ = model_manager.validate_and_map_model(v)
        return mapped_model


class Usage(BaseModel):
    input_tokens: int = Field(0, description="输入token数")
    output_tokens: int = Field(0, description="输出token数")
    cache_creation_input_tokens: int = Field(0, description="缓存创建输入token数")
    cache_read_input_tokens: int = Field(0, description="缓存读取输入token数")


class MessagesResponse(BaseModel):
    id: str = Field(..., description="响应ID")
    model: str = Field(..., description="使用的模型")
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]] = Field(..., description="响应内容")
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "error"]] = Field(None, description="停止原因")
    stop_sequence: Optional[str] = Field(None, description="停止序列")
    usage: Usage = Field(..., description="使用统计")


class TokenCountResponse(BaseModel):
    input_tokens: int = Field(..., description="输入token数")


# ===== 请求/响应转换 =====
class RequestConverter:
    """请求格式转换器，基于server.py实现"""
    
    @staticmethod
    def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
        """将Anthropic API请求转换为LiteLLM格式"""
        litellm_messages = []

        # 处理系统消息
        if anthropic_request.system:
            system_text = ""
            if isinstance(anthropic_request.system, str):
                system_text = anthropic_request.system
            elif isinstance(anthropic_request.system, list):
                text_parts = []
                for block in anthropic_request.system:
                    if hasattr(block, 'type') and block.type == "text":
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                system_text = "\n\n".join(text_parts)

            if system_text.strip():
                litellm_messages.append({"role": "system", "content": system_text.strip()})

        # 处理消息
        for msg in anthropic_request.messages:
            if isinstance(msg.content, str):
                litellm_messages.append({"role": msg.role, "content": msg.content})
                continue

            # 处理内容块
            text_parts = []
            image_parts = []
            tool_calls = []
            pending_tool_messages = []

            for block in msg.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "image":
                    if (isinstance(block.source, dict) and
                        block.source.get("type") == "base64" and
                        "media_type" in block.source and "data" in block.source):
                        image_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{block.source['media_type']};base64,{block.source['data']}"
                            }
                        })
                elif block.type == "tool_use" and msg.role == "assistant":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input)
                        }
                    })
                elif block.type == "tool_result" and msg.role == "user":
                    # 分割用户消息以处理工具结果
                    if text_parts or image_parts:
                        content_parts = []
                        text_content = "".join(text_parts).strip()
                        if text_content:
                            content_parts.append({"type": "text", "text": text_content})
                        content_parts.extend(image_parts)

                        litellm_messages.append({
                            "role": "user",
                            "content": content_parts[0]["text"] if len(content_parts) == 1 and content_parts[0]["type"] == "text" else content_parts
                        })
                        text_parts.clear()
                        image_parts.clear()

                    # 添加工具结果作为单独的"tool"角色消息
                    parsed_content = parse_tool_result_content(block.content)
                    pending_tool_messages.append({
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": parsed_content
                    })

            # 根据角色完成消息处理
            if msg.role == "user":
                if text_parts or image_parts:
                    content_parts = []
                    text_content = "".join(text_parts).strip()
                    if text_content:
                        content_parts.append({"type": "text", "text": text_content})
                    content_parts.extend(image_parts)

                    litellm_messages.append({
                        "role": "user",
                        "content": content_parts[0]["text"] if len(content_parts) == 1 and content_parts[0]["type"] == "text" else content_parts
                    })
                litellm_messages.extend(pending_tool_messages)

            elif msg.role == "assistant":
                assistant_msg = {"role": "assistant"}

                content_parts = []
                text_content = "".join(text_parts).strip()
                if text_content:
                    content_parts.append({"type": "text", "text": text_content})
                content_parts.extend(image_parts)

                if content_parts:
                    assistant_msg["content"] = content_parts[0]["text"] if len(content_parts) == 1 and content_parts[0]["type"] == "text" else content_parts
                else:
                    assistant_msg["content"] = None

                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls

                if assistant_msg.get("content") or assistant_msg.get("tool_calls"):
                    litellm_messages.append(assistant_msg)

        # 构建LiteLLM请求
        litellm_request = {
            "model": anthropic_request.model,
            "messages": litellm_messages,
            "max_tokens": min(anthropic_request.max_tokens, settings.MAX_TOKENS_LIMIT),
            "temperature": anthropic_request.temperature,
            "stream": anthropic_request.stream,
            "api_key": settings.GEMINI_API_KEY,
        }

        # 添加可选参数
        if anthropic_request.stop_sequences:
            litellm_request["stop"] = anthropic_request.stop_sequences
        if anthropic_request.top_p is not None:
            litellm_request["top_p"] = anthropic_request.top_p
        if anthropic_request.top_k is not None:
            litellm_request["topK"] = anthropic_request.top_k

        # 添加代理配置
        if settings.effective_proxy_url:
            litellm_request["proxies"] = {"all://": settings.effective_proxy_url}

        # 添加工具
        if anthropic_request.tools:
            valid_tools = []
            for tool in anthropic_request.tools:
                if tool.name and tool.name.strip():
                    from .utils import StreamingHandler
                    cleaned_schema = StreamingHandler.clean_gemini_schema(tool.input_schema)
                    valid_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": cleaned_schema
                        }
                    })
            if valid_tools:
                litellm_request["tools"] = valid_tools

        # 添加工具选择配置
        if anthropic_request.tool_choice:
            choice_type = anthropic_request.tool_choice.get("type")
            if choice_type == "auto":
                litellm_request["tool_choice"] = "auto"
            elif choice_type == "any":
                litellm_request["tool_choice"] = "auto"
            elif choice_type == "tool" and "name" in anthropic_request.tool_choice:
                litellm_request["tool_choice"] = {
                    "type": "function",
                    "function": {"name": anthropic_request.tool_choice["name"]}
                }
            else:
                litellm_request["tool_choice"] = "auto"

        # 添加思考配置
        if anthropic_request.thinking is not None:
            if anthropic_request.thinking.enabled:
                litellm_request["thinkingConfig"] = {"thinkingBudget": 24576}
            else:
                litellm_request["thinkingConfig"] = {"thinkingBudget": 0}

        # 添加用户元数据
        if (anthropic_request.metadata and
            "user_id" in anthropic_request.metadata and
            isinstance(anthropic_request.metadata["user_id"], str)):
            litellm_request["user"] = anthropic_request.metadata["user_id"]

        # 添加超时和重试配置
        litellm_request["timeout"] = settings.REQUEST_TIMEOUT
        litellm_request["num_retries"] = settings.MAX_RETRIES

        return litellm_request

    @staticmethod
    def convert_litellm_to_anthropic(litellm_response, original_request: MessagesRequest) -> MessagesResponse:
        """将LiteLLM响应转换为Anthropic API格式"""
        try:
            response_id = f"msg_{uuid.uuid4()}"
            content_text = ""
            tool_calls = None
            finish_reason = "stop"
            prompt_tokens = 0
            completion_tokens = 0

            # 处理LiteLLM ModelResponse对象格式
            if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
                choices = litellm_response.choices
                message = choices[0].message if choices else None
                content_text = getattr(message, 'content', "") or ""
                tool_calls = getattr(message, 'tool_calls', None)
                finish_reason = choices[0].finish_reason if choices else "stop"
                response_id = getattr(litellm_response, 'id', response_id)

                if hasattr(litellm_response, 'usage'):
                    usage = litellm_response.usage
                    prompt_tokens = getattr(usage, "prompt_tokens", 0)
                    completion_tokens = getattr(usage, "completion_tokens", 0)

            # 处理字典响应格式
            elif isinstance(litellm_response, dict):
                choices = litellm_response.get("choices", [])
                message = choices[0].get("message", {}) if choices else {}
                content_text = message.get("content", "") or ""
                tool_calls = message.get("tool_calls")
                finish_reason = choices[0].get("finish_reason", "stop") if choices else "stop"
                usage = litellm_response.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                response_id = litellm_response.get("id", response_id)

            # 构建内容块
            content_blocks = []

            # 添加文本内容
            if content_text:
                content_blocks.append(ContentBlockText(type="text", text=content_text))

            # 处理工具调用
            if tool_calls:
                if not isinstance(tool_calls, list):
                    tool_calls = [tool_calls]

                for tool_call in tool_calls:
                    try:
                        if isinstance(tool_call, dict):
                            tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                            function_data = tool_call.get("function", {})
                            name = function_data.get("name", "")
                            arguments_str = function_data.get("arguments", "{}")
                        elif hasattr(tool_call, "id") and hasattr(tool_call, "function"):
                            tool_id = tool_call.id
                            name = tool_call.function.name
                            arguments_str = tool_call.function.arguments
                        else:
                            continue

                        if not name:
                            continue

                        try:
                            arguments_dict = json.loads(arguments_str)
                        except json.JSONDecodeError:
                            arguments_dict = {"raw_arguments": arguments_str}

                        content_blocks.append(ContentBlockToolUse(
                            type="tool_use",
                            id=tool_id,
                            name=name,
                            input=arguments_dict
                        ))
                    except Exception as e:
                        logger.warning(f"处理工具调用时出错: {e}")
                        continue

            # 确保至少有一个内容块
            if not content_blocks:
                content_blocks.append(ContentBlockText(type="text", text=""))

            # 映射停止原因
            if finish_reason == "length":
                stop_reason = "max_tokens"
            elif finish_reason == "tool_calls":
                stop_reason = "tool_use"
            elif finish_reason is None and tool_calls:
                stop_reason = "tool_use"
            else:
                stop_reason = "end_turn"

            return MessagesResponse(
                id=response_id,
                model=original_request.original_model or original_request.model,
                role="assistant",
                content=content_blocks,
                stop_reason=stop_reason,
                stop_sequence=None,
                usage=Usage(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens
                )
            )

        except Exception as e:
            logger.error(f"响应转换错误: {e}")
            return MessagesResponse(
                id=f"msg_error_{uuid.uuid4()}",
                model=original_request.original_model or original_request.model,
                role="assistant",
                content=[ContentBlockText(type="text", text="响应转换错误")],
                stop_reason="error",
                usage=Usage(input_tokens=0, output_tokens=0)
            )


# ===== 应用生命周期管理 =====
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期管理"""
    logger.info("🚀 启动增强版Claude Code Gemini代理服务...")
    
    try:
        # 验证配置
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY环境变量未设置")
        
        logger.info(f"✅ 配置验证通过")
        logger.info(f"   服务地址: {settings.effective_host}:{settings.effective_port}")
        logger.info(f"   大模型: {settings.BIG_MODEL}")
        logger.info(f"   小模型: {settings.SMALL_MODEL}")
        logger.info(f"   可用Gemini模型: {len(model_manager.gemini_models)}个")
        logger.info(f"   代理配置: {settings.effective_proxy_url or '未启用'}")
        logger.info(f"   流式配置: 强制禁用={settings.FORCE_DISABLE_STREAMING}, 紧急禁用={settings.EMERGENCY_DISABLE_STREAMING}")
        
    except Exception as e:
        logger.error(f"❌ 启动失败: {e}")
        raise
    
    yield
    
    logger.info("🛑 正在关闭服务...")


# ===== 创建FastAPI应用 =====
app = FastAPI(
    title="增强版Claude Code Gemini代理",
    description="基于server.py实现的高性能Claude Code代理服务，支持完整的API端点和流式处理",
    version="2.5.0",
    lifespan=lifespan
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ===== 请求日志中间件 =====
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件"""
    start_time = time.time()
    method = request.method
    path = request.url.path
    
    logger.debug(f"📥 请求: {method} {path}")
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    logger.debug(f"📤 响应: {method} {path} - {response.status_code} ({duration:.2f}s)")
    
    return response


# ===== API端点 =====
@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "service": "增强版Claude Code Gemini代理",
        "version": "2.5.0",
        "status": "running",
        "description": "基于server.py实现的高性能代理服务",
        "endpoints": {
            "messages": "/v1/messages",
            "count_tokens": "/v1/messages/count_tokens",
            "health": "/health",
            "test_connection": "/test-connection",
            "models": "/models"
        },
        "configuration": {
            "big_model": settings.BIG_MODEL,
            "small_model": settings.SMALL_MODEL,
            "available_models": model_manager.gemini_models,
            "proxy_enabled": bool(settings.effective_proxy_url),
            "streaming_config": {
                "force_disabled": settings.FORCE_DISABLE_STREAMING,
                "emergency_disabled": settings.EMERGENCY_DISABLE_STREAMING,
                "max_retries": settings.MAX_STREAMING_RETRIES
            }
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        health_status = await HealthChecker.check_system_health()
        
        if health_status["healthy"]:
            return health_status
        else:
            return JSONResponse(
                status_code=503,
                content=health_status
            )
            
    except Exception as e:
        logger.error(f"健康检查错误: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.get("/models")
async def get_available_models():
    """获取可用的模型列表"""
    return {
        "models": model_manager.gemini_models,
        "big_model": settings.BIG_MODEL,
        "small_model": settings.SMALL_MODEL,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/test-connection")
async def test_connection():
    """测试Gemini API连接"""
    try:
        result = await HealthChecker.test_gemini_api()
        return result
    except Exception as e:
        logger.error(f"连接测试错误: {e}")
        error_info = error_handler.get_error_solution(
            error_handler.classify_error(str(e)), str(e)
        )
        return JSONResponse(
            status_code=503,
            content={
                "status": "failed",
                "error_type": error_info["error_type"],
                "message": error_info["message"],
                "solution": error_info["solution"],
                "timestamp": datetime.now().isoformat()
            }
        )


@app.post("/v1/messages")
async def create_message(request: MessagesRequest, raw_request: Request):
    """
    主消息处理端点
    兼容Anthropic Messages API格式
    """
    request_id = create_request_id()
    
    try:
        logger.info(f"📊 处理请求 - ID: {request_id}, 模型: {request.original_model}, 流式: {request.stream}")

        # 检查流式配置
        if request.stream and settings.EMERGENCY_DISABLE_STREAMING:
            logger.warning("⚠️ 流式响应通过EMERGENCY_DISABLE_STREAMING被禁用")
            request.stream = False

        if request.stream and settings.FORCE_DISABLE_STREAMING:
            logger.info("ℹ️ 流式响应通过FORCE_DISABLE_STREAMING被禁用")
            request.stream = False
        # 转换请求
        litellm_request = RequestConverter.convert_anthropic_to_litellm(request)
        
        # 日志记录
        num_tools = len(request.tools) if request.tools else 0
        LoggingManager.log_api_request(
            request_id, "POST", raw_request.url.path,
            request.original_model or request.model,
            len(litellm_request['messages']),
            num_tools
        )

        # logger.info(f"Converted request: {litellm_request.}")

        # 处理流式响应
        if request.stream:
            return await handle_streaming_response(litellm_request, request, request_id)
        else:
            # 处理非流式响应
            return await handle_non_streaming_response(litellm_request, request, request_id)

    except litellm.exceptions.APIError as e:
        logger.error(f"🔴 LiteLLM API错误 - ID: {request_id}: {e}")
        error_msg = error_handler.get_error_solution(
            error_handler.classify_error(str(e)), str(e)
        )
        raise HTTPException(status_code=getattr(e, 'status_code', 500), detail=error_msg["message"])
    
    except ConnectionError as e:
        logger.error(f"🔌 连接错误 - ID: {request_id}: {e}")
        raise HTTPException(status_code=503, detail="连接错误，请检查网络连接")
    
    except TimeoutError as e:
        logger.error(f"⏰ 超时错误 - ID: {request_id}: {e}")
        raise HTTPException(status_code=504, detail="请求超时，请重试")
    
    except Exception as e:
        logger.error(f"❌ 处理请求时出错 - ID: {request_id}: {e}")
        error_msg = error_handler.get_error_solution(
            error_handler.classify_error(str(e)), str(e)
        )
        raise HTTPException(status_code=500, detail=error_msg["message"])


async def handle_streaming_response(litellm_request: Dict[str, Any], 
                                  original_request: MessagesRequest,
                                  request_id: str):
    """处理流式响应"""
    streaming_retry_count = 0
    max_retries = settings.MAX_STREAMING_RETRIES

    while streaming_retry_count <= max_retries:
        try:
            if streaming_retry_count > 0:
                delay = min(0.5 * (2 ** streaming_retry_count), 2.0)
                logger.debug(f"⏳ 重试等待: {delay}s (尝试 {streaming_retry_count}/{max_retries})")
                await asyncio.sleep(delay)

            response_generator = await litellm.acompletion(**litellm_request)

            return StreamingResponse(
                streaming_handler.handle_streaming_with_recovery(response_generator, request_id, original_request.original_model or original_request.model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                    "X-Request-ID": request_id
                }
            )

        except Exception as streaming_error:
            streaming_retry_count += 1
            logger.warning(f"🔄 流式响应重试 {streaming_retry_count}/{max_retries}: {streaming_error}")

            if streaming_retry_count > max_retries:
                logger.error("❌ 流式响应失败，回退到非流式模式")
                break

    # 回退到非流式响应
    logger.info("📥 回退到非流式模式")
    litellm_request["stream"] = False
    return await handle_non_streaming_response(litellm_request, original_request, request_id)


async def handle_non_streaming_response(litellm_request: Dict[str, Any],
                                     original_request: MessagesRequest,
                                     request_id: str):
    """处理非流式响应"""
    start_time = time.time()
    
    litellm_response = await retry_manager.execute_with_retry(
        litellm.acompletion, **litellm_request
    )
    
    duration = time.time() - start_time
    anthropic_response = RequestConverter.convert_litellm_to_anthropic(litellm_response, original_request)
    
    # 日志记录
    LoggingManager.log_api_response(
        request_id, duration,
        {
            "input_tokens": anthropic_response.usage.input_tokens,
            "output_tokens": anthropic_response.usage.output_tokens
        },
        "success"
    )
    
    return anthropic_response


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: TokenCountRequest, raw_request: Request):
    """Token计数端点"""
    request_id = create_request_id()
    
    try:
        logger.info(f"📊 Token计数 - ID: {request_id}, 模型: {request.original_model}")

        # 创建临时请求进行转换
        temp_request = MessagesRequest(
            model=request.model,
            max_tokens=1,
            messages=request.messages,
            system=request.system,
            tools=request.tools,
        )

        litellm_data = RequestConverter.convert_anthropic_to_litellm(temp_request)

        # 日志记录
        num_tools = len(request.tools) if request.tools else 0
        LoggingManager.log_api_request(
            request_id, "POST", raw_request.url.path,
            request.original_model or request.model,
            len(litellm_data['messages']),
            num_tools
        )

        # 计算token数
        token_count = litellm.token_counter(
            model=litellm_data["model"],
            messages=litellm_data["messages"],
        )

        response = TokenCountResponse(input_tokens=token_count)
        
        LoggingManager.log_api_response(
            request_id, 0.1, {"input_tokens": token_count}, "success"
        )
        
        return response

    except Exception as e:
        logger.error(f"❌ Token计数错误 - ID: {request_id}: {e}")
        error_msg = error_handler.get_error_solution(
            error_handler.classify_error(str(e)), str(e)
        )
        raise HTTPException(status_code=500, detail=f"Token计数错误: {error_msg['message']}")


# ===== 启动函数 =====
def validate_startup():
    """启动时验证配置"""
    print("🔍 验证启动配置...")
    
    # 检查API密钥
    if not settings.GEMINI_API_KEY:
        print("❌ 致命错误: GEMINI_API_KEY未设置")
        return False
    
    if not settings.validate_api_key():
        print("⚠️ 警告: API密钥格式验证失败")
    
    # 检查网络连接
    try:
        import socket
        socket.create_connection(("8.8.8.8", 53), timeout=10)
        print("✅ 网络连接正常")
    except OSError:
        print("⚠️ 警告: 网络连接检查失败")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("增强版Claude Code Gemini代理 v2.5.0")
        print("")
        print("使用方式: uvicorn src.main:app --reload --host 0.0.0.0 --port 8080")
        print("")
        print("必需环境变量:")
        print("  GEMINI_API_KEY - Google Gemini API密钥")
        print("")
        print("可选环境变量:")
        print(f"  BIG_MODEL - 大模型名称 (默认: {settings.BIG_MODEL})")
        print(f"  SMALL_MODEL - 小模型名称 (默认: {settings.SMALL_MODEL})")
        print(f"  HOST - 服务器地址 (默认: {settings.HOST})")
        print(f"  PORT - 服务器端口 (默认: {settings.PORT})")
        print(f"  LOG_LEVEL - 日志级别 (默认: {settings.LOG_LEVEL})")
        print(f"  PROXY_ENABLED - 是否启用代理 (默认: {settings.PROXY_ENABLED})")
        print(f"  PROXY_URL - 代理地址 (默认: {settings.PROXY_URL})")
        sys.exit(0)

    # 验证启动配置
    if not validate_startup():
        print("❌ 启动验证失败，请检查配置")
        sys.exit(1)

    # 配置摘要
    print("🚀 增强版Claude Code Gemini代理 v2.5.0")
    print("✅ 配置验证通过")
    print(f"   大模型: {settings.BIG_MODEL}")
    print(f"   小模型: {settings.SMALL_MODEL}")
    print(f"   可用模型: {len(model_manager.gemini_models)}个")
    print(f"   服务地址: {settings.effective_host}:{settings.effective_port}")
    print(f"   代理配置: {settings.effective_proxy_url or '未启用'}")
    print(f"   流式配置: 重试次数={settings.MAX_STREAMING_RETRIES}")
    print("")

    # 启动服务器
    uvicorn.run(
        "src.main:app",
        host=settings.effective_host,
        port=settings.effective_port,
        log_level=settings.LOG_LEVEL.lower(),
        reload=True
    )