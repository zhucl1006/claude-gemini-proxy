"""
增强版工具函数模块
基于server.py的成熟实现，整合错误处理、代理支持和流式处理
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union
import traceback

import aiohttp
import litellm
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from .config import settings


# ===== 日志配置 =====
logger = logging.getLogger(__name__)


# ===== 常量定义 =====
class Constants:
    """常量定义，与server.py保持一致"""
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_SYSTEM = "system"
    ROLE_TOOL = "tool"

    CONTENT_TEXT = "text"
    CONTENT_IMAGE = "image"
    CONTENT_TOOL_USE = "tool_use"
    CONTENT_TOOL_RESULT = "tool_result"

    TOOL_FUNCTION = "function"

    STOP_END_TURN = "end_turn"
    STOP_MAX_TOKENS = "max_tokens"
    STOP_TOOL_USE = "tool_use"
    STOP_ERROR = "error"

    EVENT_MESSAGE_START = "message_start"
    EVENT_MESSAGE_STOP = "message_stop"
    EVENT_MESSAGE_DELTA = "message_delta"
    EVENT_CONTENT_BLOCK_START = "content_block_start"
    EVENT_CONTENT_BLOCK_STOP = "content_block_stop"
    EVENT_CONTENT_BLOCK_DELTA = "content_block_delta"
    EVENT_PING = "ping"

    DELTA_TEXT = "text_delta"
    DELTA_INPUT_JSON = "input_json_delta"


# ===== 配置管理 =====
class LiteLLMConfig:
    """LiteLLM配置管理"""
    
    @staticmethod
    def setup_litellm():
        """配置LiteLLM"""
        litellm.drop_params = True
        litellm.set_verbose = False
        litellm.request_timeout = settings.REQUEST_TIMEOUT
        litellm.num_retries = settings.MAX_RETRIES
        
        # 配置代理
        proxy_url = settings.effective_proxy_url
        if proxy_url:
            litellm.proxy_url = proxy_url
            litellm.proxy_config = {"all://": proxy_url}
            logger.info(f"已配置代理: {proxy_url}")


# ===== 错误处理 =====
class EnhancedErrorHandler:
    """增强版错误处理类，基于server.py实现"""
    
    ERROR_PATTERNS = {
        "invalid_api_key": [
            "api_key", "invalid", "unauthorized", "permission denied", "authentication"
        ],
        "rate_limit": [
            "rate limit", "quota exceeded", "too many requests", "429", "throttle"
        ],
        "model_unavailable": [
            "model not found", "model unavailable", "does not exist", "not supported"
        ],
        "content_violation": [
            "safety", "blocked", "content policy", "violation", "inappropriate"
        ],
        "network_error": [
            "network", "timeout", "connection", "resolve", "dns", "connection error"
        ],
        "invalid_request": [
            "bad request", "validation", "invalid parameter", "malformed", "schema"
        ],
        "streaming_error": [
            "error parsing chunk", "malformed json", "streaming", "chunk"
        ],
        "tool_schema_error": [
            "function_declarations", "format", "schema", "tool", "parameter"
        ]
    }
    
    @classmethod
    def classify_error(cls, error_message: str) -> str:
        """分类错误类型"""
        error_lower = str(error_message).lower()
        
        # 特殊错误类型检测
        if "error parsing chunk" in error_lower and "expecting property name" in error_lower:
            return "streaming_parsing_error"
        
        if "function_declarations" in error_lower and "format" in error_lower:
            if "only 'enum' and 'date-time' are supported" in error_lower:
                return "tool_schema_format_error"
            return "tool_schema_error"
        
        # 常规错误分类
        for error_type, patterns in cls.ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern in error_lower:
                    return error_type
        
        return "unknown_error"
    
    @classmethod
    def get_error_solution(cls, error_type: str, error_message: str) -> Dict[str, Any]:
        """获取详细的错误解决方案"""
        solutions = {
            "invalid_api_key": {
                "message": "API密钥无效或权限不足",
                "solution": "请检查您的GEMINI_API_KEY是否正确设置。可以通过访问Google AI Studio获取有效的API密钥。",
                "action": "更新API密钥",
                "docs_link": "https://ai.google.dev/gemini-api/docs/api-key"
            },
            "rate_limit": {
                "message": "达到API调用频率限制",
                "solution": "您已达到API调用频率限制。请稍后再试，或考虑升级到更高级别的API套餐。",
                "action": "等待重试或升级套餐",
                "retry_after": 60
            },
            "model_unavailable": {
                "message": "请求的模型当前不可用",
                "solution": "请尝试使用其他Gemini模型，如gemini-1.5-pro或gemini-1.5-flash。",
                "action": "更换模型",
                "available_models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
            },
            "content_violation": {
                "message": "内容被安全过滤器阻止",
                "solution": "内容被安全过滤器阻止。请检查输入内容是否符合使用政策。",
                "action": "修改输入内容",
                "policy_link": "https://ai.google.dev/gemini-api/docs/safety-settings"
            },
            "network_error": {
                "message": "网络连接问题",
                "solution": "网络连接问题。请检查您的网络连接，或验证代理配置是否正确。",
                "action": "检查网络/代理设置",
                "proxy_check": True
            },
            "streaming_parsing_error": {
                "message": "Gemini流式解析错误",
                "solution": "这是已知的Gemini API间歇性问题。请重试请求或禁用流式响应。",
                "action": "重试或禁用流式",
                "disable_streaming": True
            },
            "tool_schema_error": {
                "message": "工具架构验证错误",
                "solution": "检查工具参数定义，Gemini只支持'enum'和'date-time'格式的字符串参数。",
                "action": "修正工具定义"
            }
        }
        
        result = solutions.get(error_type, {
            "message": f"未知错误: {error_message}",
            "solution": "发生未知错误，请查看日志获取详细信息。",
            "action": "查看日志"
        })
        
        result["error_type"] = error_type
        result["original_message"] = str(error_message)
        result["timestamp"] = datetime.now().isoformat()
        
        return result


# ===== 重试管理 =====
class EnhancedRetryManager:
    """增强版重试管理器"""
    
    def __init__(self, max_retries: int = None, base_delay: float = None, max_delay: float = None):
        self.max_retries = max_retries or settings.MAX_RETRIES
        self.base_delay = base_delay or 1.0
        self.max_delay = max_delay or 60.0
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs):
        """执行带重试的函数"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_type = EnhancedErrorHandler.classify_error(str(e))
                
                # 某些错误不重试
                non_retryable_errors = ["invalid_api_key", "content_violation", "invalid_request"]
                if error_type in non_retryable_errors:
                    logger.error(f"不可重试的错误类型 {error_type}: {e}")
                    raise
                
                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (2 ** attempt),
                        self.max_delay
                    )
                    
                    logger.warning(f"第{attempt + 1}次尝试失败，等待{delay}秒后重试: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"重试{self.max_retries}次后仍然失败: {e}")
        
        raise last_exception


# ===== 代理管理 =====
class ProxyManager:
    """代理配置管理器"""
    
    @staticmethod
    def get_proxy_config() -> Dict[str, str]:
        """获取代理配置"""
        proxy_url = settings.effective_proxy_url
        if proxy_url:
            return {
                "http": proxy_url,
                "https": proxy_url,
                "all://": proxy_url
            }
        return {}
    
    @staticmethod
    def configure_litellm_proxy():
        """配置LiteLLM代理"""
        proxy_config = ProxyManager.get_proxy_config()
        if proxy_config:
            litellm.proxy_config = proxy_config
            logger.info("已配置LiteLLM代理")


# ===== 模型管理 =====
class EnhancedModelManager:
    """增强版模型管理器，基于server.py实现"""
    
    def __init__(self):
        self.base_gemini_models = [
            "gemini-1.5-pro-latest",
            "gemini-1.5-pro-preview-0514",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-preview-0514",
            "gemini-pro",
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.0-flash-exp",
            "gemini-exp-1206"
        ]
        
        self._gemini_models = set(self.base_gemini_models)
        self._add_env_models()
    
    def _add_env_models(self):
        """添加环境变量中的模型"""
        for model in [settings.BIG_MODEL, settings.SMALL_MODEL]:
            if model.startswith("gemini") and model not in self._gemini_models:
                self._gemini_models.add(model)
    
    @property
    def gemini_models(self) -> List[str]:
        """获取可用的Gemini模型列表"""
        return sorted(list(self._gemini_models))
    
    def validate_and_map_model(self, original_model: str) -> tuple[str, bool]:
        """验证并映射模型"""
        clean_model = self._clean_model_name(original_model)
        mapped_model = self._map_model_alias(clean_model)
        
        if mapped_model != clean_model:
            return f"gemini/{mapped_model}", True
        elif clean_model in self._gemini_models:
            return f"gemini/{clean_model}", True
        elif not original_model.startswith('gemini/'):
            return f"gemini/{original_model}", False
        else:
            return original_model, False
    
    def _clean_model_name(self, model: str) -> str:
        """清理模型名称"""
        if model.startswith('gemini/'):
            return model[7:]
        elif model.startswith('anthropic/'):
            return model[10:]
        elif model.startswith('openai/'):
            return model[7:]
        return model
    
    def _map_model_alias(self, clean_model: str) -> str:
        """映射模型别名"""
        model_lower = clean_model.lower()
        
        if 'haiku' in model_lower:
            return settings.SMALL_MODEL
        elif 'sonnet' in model_lower or 'opus' in model_lower:
            return settings.BIG_MODEL
        
        return clean_model


# ===== 流式处理 =====
class StreamingHandler:
    """流式处理管理器，基于server.py实现"""
    
    def __init__(self):
        self.max_consecutive_errors = 10
        self.max_malformed_chunks = 20
        self.chunk_timeout = 90.0
    
    @staticmethod
    def clean_gemini_schema(schema: Any) -> Any:
        """清理Gemini架构，增强对VertexAI JSON schema的兼容性"""
        if isinstance(schema, dict):
            # 移除不支持的字段
            schema.pop("additionalProperties", None)
            schema.pop("default", None)
            schema.pop("$schema", None)
            schema.pop("$ref", None)
            schema.pop("definitions", None)
            
            # 处理字符串格式限制
            if schema.get("type") == "string" and "format" in schema:
                allowed_formats = {"enum", "date-time"}
                if schema["format"] not in allowed_formats:
                    logger.debug(f"移除不支持的格式: {schema['format']}")
                    schema.pop("format")
            
            # 处理嵌套的properties中的type字段
            if "properties" in schema and isinstance(schema["properties"], dict):
                for prop_name, prop_schema in list(schema["properties"].items()):
                    if isinstance(prop_schema, dict):
                        # 移除properties中嵌套的type字段（除了根type）
                        if "type" in prop_schema and prop_schema["type"] not in ["string", "number", "integer", "boolean", "array", "object"]:
                            logger.debug(f"移除不支持的嵌套type: {prop_schema['type']} in property {prop_name}")
                            prop_schema.pop("type", None)
                        
                        # 递归清理子schema
                        schema["properties"][prop_name] = StreamingHandler.clean_gemini_schema(prop_schema)
            
            # 清理items（数组的schema）
            if "items" in schema:
                schema["items"] = StreamingHandler.clean_gemini_schema(schema["items"])
            
            # 清理所有嵌套结构
            for key, value in list(schema.items()):
                if key not in ["type", "properties", "items", "required", "enum", "description"]:
                    schema[key] = StreamingHandler.clean_gemini_schema(value)
                elif key in ["properties", "items"]:
                    # 这些已经单独处理过了
                    continue
                else:
                    schema[key] = StreamingHandler.clean_gemini_schema(value)
        
        elif isinstance(schema, list):
            return [StreamingHandler.clean_gemini_schema(item) for item in schema]
        
        return schema
    
    @staticmethod
    def is_malformed_chunk(chunk_str: str) -> bool:
        """检测畸形数据块"""
        if not chunk_str or not isinstance(chunk_str, str):
            return True
        
        chunk_stripped = chunk_str.strip()
        
        if not chunk_stripped or chunk_stripped in ["{", "}", "[", "]", ",", ":", '"', "'"]:
            return True
        
        malformed_patterns = [
            '{"', '"}', "[{", "}]", "{}", "[]",
            "null", '""', "''", " ", "",
            "{,", ",}", "[,", ",]"
        ]
        
        if chunk_stripped in malformed_patterns:
            return True
        
        return False
    
    @staticmethod
    def try_parse_buffered_chunk(buffer: str) -> tuple[Optional[Dict], str]:
        """尝试解析缓冲的数据块"""
        if not buffer.strip():
            return None, ""
        
        brace_count = 0
        start_pos = -1
        
        for i, char in enumerate(buffer):
            if char == '{':
                if start_pos == -1:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    json_str = buffer[start_pos:i+1]
                    try:
                        parsed = json.loads(json_str)
                        remaining_buffer = buffer[i+1:]
                        return parsed, remaining_buffer
                    except json.JSONDecodeError:
                        continue
        
        return None, buffer
    
    async def handle_streaming_with_recovery(self, response_generator, request_id: str, original_model: str):
        """处理流式响应，带错误恢复"""
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        
        # 初始事件
        yield f"event: {Constants.EVENT_MESSAGE_START}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_START, 'message': {'id': message_id, 'type': 'message', 'role': Constants.ROLE_ASSISTANT, 'model': original_model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
        
        yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': 0, 'content_block': {'type': Constants.CONTENT_TEXT, 'text': ''}})}\n\n"
        
        yield f"event: {Constants.EVENT_PING}\ndata: {json.dumps({'type': Constants.EVENT_PING})}\n\n"
        
        # 状态管理
        accumulated_text = ""
        text_block_index = 0
        consecutive_errors = 0
        malformed_chunks_count = 0
        chunk_buffer = ""
        input_tokens = 0
        output_tokens = 0
        final_stop_reason = Constants.STOP_END_TURN
        stream_terminated_early = False
        
        try:
            stream_iterator = aiter(response_generator)
            
            while True:
                try:
                    chunk = await asyncio.wait_for(anext(stream_iterator), timeout=self.chunk_timeout)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    logger.warning("流式响应超时")
                    stream_terminated_early = True
                    break
                
                consecutive_errors = 0
                
                if isinstance(chunk, str):
                    if chunk.strip() == "[DONE]":
                        break
                    
                    if self.is_malformed_chunk(chunk):
                        malformed_chunks_count += 1
                        if malformed_chunks_count > self.max_malformed_chunks:
                            logger.error(f"畸形数据块过多({malformed_chunks_count})，终止流")
                            stream_terminated_early = True
                            break
                        continue
                    
                    chunk_buffer += chunk
                    parsed_chunk, chunk_buffer = self.try_parse_buffered_chunk(chunk_buffer)
                    
                    if parsed_chunk is None:
                        if len(chunk_buffer) > 10000:
                            chunk_buffer = ""
                        continue
                    
                    chunk = parsed_chunk
                
                try:
                    delta_content_text = None
                    chunk_finish_reason = None
                    
                    if hasattr(chunk, 'choices') and chunk.choices:
                        choice = chunk.choices[0]
                        if hasattr(choice, 'delta') and choice.delta:
                            delta = choice.delta
                            delta_content_text = getattr(delta, 'content', None)
                        chunk_finish_reason = getattr(choice, 'finish_reason', None)
                    elif isinstance(chunk, dict):
                        choices = chunk.get("choices", [])
                        if choices:
                            choice = choices[0]
                            delta = choice.get("delta", {})
                            delta_content_text = delta.get("content")
                            chunk_finish_reason = choice.get("finish_reason")
                    
                    if hasattr(chunk, 'usage') and chunk.usage:
                        input_tokens = getattr(chunk.usage, 'prompt_tokens', 0)
                        output_tokens = getattr(chunk.usage, 'completion_tokens', 0)
                    elif isinstance(chunk, dict) and "usage" in chunk:
                        usage = chunk["usage"]
                        input_tokens = usage.get("prompt_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0)
                    
                    if delta_content_text:
                        accumulated_text += delta_content_text
                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': text_block_index, 'delta': {'type': Constants.DELTA_TEXT, 'text': delta_content_text}})}\n\n"
                    
                    if chunk_finish_reason:
                        if chunk_finish_reason == "length":
                            final_stop_reason = Constants.STOP_MAX_TOKENS
                        elif chunk_finish_reason == "tool_calls":
                            final_stop_reason = Constants.STOP_TOOL_USE
                        elif chunk_finish_reason == "stop":
                            final_stop_reason = Constants.STOP_END_TURN
                        else:
                            final_stop_reason = Constants.STOP_END_TURN
                        break
                
                except Exception as e:
                    consecutive_errors += 1
                    if consecutive_errors >= self.max_consecutive_errors:
                        logger.error(f"连续错误过多({consecutive_errors})，终止流")
                        stream_terminated_early = True
                        break
                    
                    await asyncio.sleep(0.1)
                    continue
        
        except Exception as e:
            logger.error(f"流式处理致命错误: {e}")
            stream_terminated_early = True
        
        # 发送结束事件
        try:
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': text_block_index})}\n\n"
            
            if stream_terminated_early and final_stop_reason == Constants.STOP_END_TURN:
                final_stop_reason = Constants.STOP_ERROR
            
            usage_data = {"input_tokens": input_tokens, "output_tokens": output_tokens}
            yield f"event: {Constants.EVENT_MESSAGE_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_DELTA, 'delta': {'stop_reason': final_stop_reason, 'stop_sequence': None}, 'usage': usage_data})}\n\n"
            yield f"event: {Constants.EVENT_MESSAGE_STOP}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_STOP})}\n\n"
            
            if malformed_chunks_count > 0:
                logger.info(f"流完成，处理了{malformed_chunks_count}个畸形数据块")
        
        except Exception as e:
            logger.error(f"发送结束事件错误: {e}")


# ===== 健康检查 =====
class HealthChecker:
    """增强版健康检查器"""
    
    @staticmethod
    async def test_gemini_api() -> Dict[str, Any]:
        """测试Gemini API连接"""
        try:
            proxy_config = ProxyManager.get_proxy_config()
            
            response = await litellm.acompletion(
                model="gemini/gemini-1.5-flash-latest",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                api_key=settings.GEMINI_API_KEY,
                proxies=proxy_config if proxy_config else None,
                timeout=settings.REQUEST_TIMEOUT
            )
            
            return {
                "status": "success",
                "message": "Gemini API连接正常",
                "model": "gemini-1.5-flash-latest",
                "proxy_enabled": bool(settings.effective_proxy_url),
                "proxy_url": settings.effective_proxy_url,
                "timestamp": datetime.now().isoformat(),
                "response_id": getattr(response, 'id', 'unknown')
            }
            
        except Exception as e:
            error_info = EnhancedErrorHandler.get_error_solution(
                EnhancedErrorHandler.classify_error(str(e)), str(e)
            )
            
            return {
                "status": "error",
                "message": str(e),
                "error_type": error_info["error_type"],
                "solution": error_info["solution"],
                "proxy_enabled": bool(settings.effective_proxy_url),
                "proxy_url": settings.effective_proxy_url,
                "timestamp": datetime.now().isoformat()
            }
    
    @staticmethod
    async def check_system_health() -> Dict[str, Any]:
        """检查系统健康状态"""
        checks = {
            "config_valid": False,
            "api_key_valid": False,
            "gemini_connection": False,
            "memory_usage": 0,
            "uptime": 0,
            "proxy_status": "disabled"
        }
        
        try:
            from .config import validate_config
            checks["config_valid"] = validate_config()
            
            checks["api_key_valid"] = bool(settings.GEMINI_API_KEY and len(settings.GEMINI_API_KEY) > 10)
            
            gemini_test = await HealthChecker.test_gemini_api()
            checks["gemini_connection"] = gemini_test["status"] == "success"
            
            try:
                import psutil
                checks["memory_usage"] = psutil.virtual_memory().percent
            except ImportError:
                checks["memory_usage"] = "unavailable"
            
            checks["uptime"] = time.time()
            
            if settings.effective_proxy_url:
                checks["proxy_status"] = "enabled"
            else:
                checks["proxy_status"] = "disabled"
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
        
        overall_health = all([
            checks["config_valid"],
            checks["api_key_valid"],
            checks["gemini_connection"]
        ])
        
        return {
            "healthy": overall_health,
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        }


# ===== 日志工具 =====
class LoggingManager:
    """日志管理器"""
    
    @staticmethod
    def setup_logging():
        """设置日志配置"""
        log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        
        handlers = [logging.StreamHandler()]
        
        if hasattr(settings, 'LOG_FILE') and settings.LOG_FILE:
            handlers.append(logging.FileHandler(settings.LOG_FILE))
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        
        # 配置uvicorn日志
        for uvicorn_logger in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
            logging.getLogger(uvicorn_logger).setLevel(logging.WARNING)
    
    @staticmethod
    def log_api_request(request_id: str, method: str, path: str, model: str, messages: int, tools: int = 0):
        """记录API请求"""
        logger.info(f"API请求: {method} {path} - 模型: {model}, 消息数: {messages}, 工具数: {tools}, 请求ID: {request_id}")
    
    @staticmethod
    def log_api_response(request_id: str, duration: float, tokens: Dict[str, int], status: str):
        """记录API响应"""
        logger.info(f"API响应: 请求ID: {request_id}, 耗时: {duration:.2f}s, token: {tokens}, 状态: {status}")


# ===== 工具函数 =====
def create_error_response(error_type: str, message: str, status_code: int = 500) -> Dict[str, Any]:
    """创建错误响应"""
    error_info = EnhancedErrorHandler.get_error_solution(error_type, message)
    
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
            "solution": error_info["solution"],
            "action": error_info["action"],
            "timestamp": datetime.now().isoformat(),
            "status_code": status_code
        }
    }


def sanitize_json_for_sse(data: Dict[str, Any]) -> str:
    """为SSE清理JSON数据"""
    return json.dumps(data, ensure_ascii=False, separators=(',', ':'))


def create_request_id() -> str:
    """创建唯一的请求ID"""
    return str(uuid.uuid4())


def parse_tool_result_content(content: Any) -> str:
    """解析工具结果内容"""
    if content is None:
        return "No content provided"
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        result_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                result_parts.append(item)
            elif isinstance(item, dict):
                try:
                    result_parts.append(json.dumps(item))
                except:
                    result_parts.append(str(item))
        return "\n".join(result_parts).strip()
    
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)
    
    try:
        return str(content)
    except:
        return "Unparseable content"


# ===== 全局实例 =====
error_handler = EnhancedErrorHandler()
retry_manager = EnhancedRetryManager()
model_manager = EnhancedModelManager()
streaming_handler = StreamingHandler()


# ===== 初始化 =====
def initialize_utils():
    """初始化工具模块"""
    try:
        LiteLLMConfig.setup_litellm()
        ProxyManager.configure_litellm_proxy()
        LoggingManager.setup_logging()
        logger.info("✅ 工具模块初始化完成")
    except Exception as e:
        logger.error(f"❌ 工具模块初始化失败: {e}")


if __name__ == "__main__":
    initialize_utils()