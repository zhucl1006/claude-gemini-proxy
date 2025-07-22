"""
API格式转换器
处理Claude API格式与Gemini API格式之间的转换
"""

import json
import logging
from typing import Dict, Any, List, Optional
import uuid

import litellm

from models import (
    ClaudeRequest, ClaudeResponse, ClaudeStreamResponse,
    GeminiRequest, GeminiResponse, Message, Role, 
    ContentBlock, ToolUse, ToolResult
)
from .config import settings


logger = logging.getLogger(__name__)


class ClaudeToGeminiConverter:
    """Claude API到Gemini API的转换器"""
    
    def __init__(self):
        """初始化转换器"""
        self.litellm = litellm
        # 设置LiteLLM配置
        litellm.api_key = settings.GEMINI_API_KEY
        litellm.set_verbose = settings.LOG_LEVEL == "DEBUG"
    
    def convert_claude_to_gemini_request(self, claude_request: ClaudeRequest) -> GeminiRequest:
        """
        将Claude API请求转换为Gemini API请求格式
        
        Args:
            claude_request: Claude API请求对象
            
        Returns:
            GeminiRequest: Gemini API请求对象
        """
        try:
            # 转换消息格式
            contents = []
            
            # 处理系统消息
            if claude_request.system:
                system_text = ""
                if isinstance(claude_request.system, str):
                    system_text = claude_request.system
                elif isinstance(claude_request.system, list):
                    # 处理Claude Code的复杂系统消息格式
                    for sys_msg in claude_request.system:
                        if isinstance(sys_msg, dict) and "text" in sys_msg:
                            system_text += sys_msg["text"] + "\n"
                        else:
                            system_text += str(sys_msg) + "\n"
                    system_text = system_text.strip()
                
                if system_text:
                    contents.append({
                        "role": "user",
                        "parts": [{"text": system_text}]
                    })
                    contents.append({
                        "role": "model",
                        "parts": [{"text": "好的，我理解了您的要求。"}]
                    })
            
            # 转换用户和助手消息
            for message in claude_request.messages:
                role = "user" if message.role == Role.USER else "model"
                
                if isinstance(message.content, str):
                    # 简单文本内容
                    contents.append({
                        "role": role,
                        "parts": [{"text": message.content}]
                    })
                elif isinstance(message.content, list):
                    # 复杂内容（文本+图像）
                    parts = []
                    for content_block in message.content:
                        if content_block.type == "text":
                            parts.append({"text": content_block.text})
                        elif content_block.type == "image":
                            # 处理图像内容
                            if content_block.source and content_block.source.type == "base64":
                                parts.append({
                                    "inline_data": {
                                        "mime_type": content_block.source.media_type,
                                        "data": content_block.source.data
                                    }
                                })
                    
                    if parts:
                        contents.append({
                            "role": role,
                            "parts": parts
                        })
            
            # 构建生成配置
            generation_config = {
                "temperature": claude_request.temperature or settings.TEMPERATURE,
                "max_output_tokens": claude_request.max_tokens or settings.MAX_TOKENS,
                "top_p": claude_request.top_p or settings.TOP_P,
                "top_k": claude_request.top_k or settings.TOP_K,
            }
            
            # 添加停止序列
            if claude_request.stop_sequences:
                generation_config["stop_sequences"] = claude_request.stop_sequences
            
            # 构建安全设置
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            # 转换工具定义
            tools = None
            if claude_request.tools:
                tools = []
                for tool in claude_request.tools:
                    gemini_tool = {
                        "function_declarations": [{
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": self._convert_json_schema(tool.input_schema.dict())
                        }]
                    }
                    tools.append(gemini_tool)
            
            return GeminiRequest(
                contents=contents,
                generation_config=generation_config,
                safety_settings=safety_settings,
                tools=tools
            )
            
        except Exception as e:
            logger.error(f"转换请求格式时出错: {e}")
            raise
    
    def convert_gemini_to_claude_response(self, gemini_response: Dict[str, Any], 
                                        original_request: ClaudeRequest) -> ClaudeResponse:
        """
        将Gemini API响应转换为Claude API响应格式
        
        Args:
            gemini_response: Gemini API响应数据
            original_request: 原始Claude请求
            
        Returns:
            ClaudeResponse: Claude API响应对象
        """
        try:
            # 检查是否有候选响应
            if not gemini_response.get("candidates"):
                raise ValueError("Gemini API没有返回任何候选响应")
            
            candidate = gemini_response["candidates"][0]
            
            # 检查内容是否被阻止
            if "finishReason" in candidate and candidate["finishReason"] in ["SAFETY", "BLOCKED"]:
                raise ValueError("内容被安全过滤器阻止")
            
            # 提取响应内容
            content_parts = []
            
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    if "text" in part:
                        content_parts.append(
                            ContentBlock(type="text", text=part["text"])
                        )
                    elif "functionCall" in part:
                        # 处理工具调用
                        function_call = part["functionCall"]
                        tool_use = ToolUse(
                            name=function_call["name"],
                            input=function_call["args"]
                        )
                        content_parts.append(
                            ContentBlock(type="tool_use", text=json.dumps(tool_use.dict()))
                        )
            
            if not content_parts:
                content_parts.append(
                    ContentBlock(type="text", text="对不起，我无法生成有意义的响应。")
                )
            
            # 确定停止原因
            stop_reason = None
            if "finishReason" in candidate:
                reason_map = {
                    "STOP": "stop_sequence",
                    "MAX_TOKENS": "max_tokens",
                    "FINISH_REASON_UNSPECIFIED": None
                }
                stop_reason = reason_map.get(candidate["finishReason"])
            
            # 提取使用情况
            usage = gemini_response.get("usageMetadata", {})
            usage_dict = {
                "input_tokens": usage.get("promptTokenCount", 0),
                "output_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0)
            }
            
            return ClaudeResponse(
                id=str(uuid.uuid4()),
                content=content_parts,
                model=original_request.model,
                stop_reason=stop_reason,
                usage=usage_dict
            )
            
        except Exception as e:
            logger.error(f"转换响应格式时出错: {e}")
            raise
    
    def _convert_json_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换JSON Schema格式以适应Gemini要求
        
        Args:
            schema: 原始JSON Schema
            
        Returns:
            转换后的JSON Schema
        """
        try:
            # Gemini对JSON Schema有特定要求
            cleaned_schema = {}
            
            # 只保留Gemini支持的字段
            supported_fields = ["type", "description", "enum", "properties", "required", 
                              "items", "default", "minimum", "maximum", "minLength", 
                              "maxLength", "pattern"]
            
            for field in supported_fields:
                if field in schema and schema[field] is not None:
                    cleaned_schema[field] = schema[field]
            
            # 处理type字段
            if "type" not in cleaned_schema:
                cleaned_schema["type"] = "object"
            elif isinstance(cleaned_schema["type"], list):
                # 如果type是数组，取第一个字符串值
                type_list = [t for t in cleaned_schema["type"] if isinstance(t, str)]
                cleaned_schema["type"] = type_list[0] if type_list else "object"
            elif not isinstance(cleaned_schema["type"], str):
                cleaned_schema["type"] = "object"  # 默认类型
            
            # 处理properties中的嵌套schema
            if "properties" in cleaned_schema:
                if isinstance(cleaned_schema["properties"], dict):
                    cleaned_properties = {}
                    for prop_name, prop_schema in cleaned_schema["properties"].items():
                        if isinstance(prop_schema, dict):
                            cleaned_properties[prop_name] = self._convert_json_schema(prop_schema)
                        else:
                            cleaned_properties[prop_name] = {"type": "string"}
                    cleaned_schema["properties"] = cleaned_properties
                else:
                    del cleaned_schema["properties"]
            
            # 处理items
            if "items" in cleaned_schema:
                if isinstance(cleaned_schema["items"], dict):
                    cleaned_schema["items"] = self._convert_json_schema(cleaned_schema["items"])
                else:
                    cleaned_schema["items"] = {"type": "string"}
            
            # 处理required字段
            if "required" in cleaned_schema and not isinstance(cleaned_schema["required"], list):
                if isinstance(cleaned_schema["required"], str):
                    cleaned_schema["required"] = [cleaned_schema["required"]]
                else:
                    cleaned_schema["required"] = []
            
            return cleaned_schema
            
        except Exception as e:
            logger.error(f"清理JSON Schema时出错: {e}")
            return {"type": "object"}  # 返回安全默认值
    
    async def convert_and_send(self, claude_request: ClaudeRequest) -> ClaudeResponse:
        """
        转换请求并发送到Gemini API
        
        Args:
            claude_request: Claude API请求
            
        Returns:
            ClaudeResponse: 转换后的响应
        """
        try:
            # 转换请求格式
            gemini_request = self.convert_claude_to_gemini_request(claude_request)
            
            # 准备LiteLLM调用参数
            messages = []
            for content in gemini_request.contents:
                role = "user" if content["role"] == "user" else "assistant"
                
                # 正确提取文本内容
                content_parts = []
                for part in content["parts"]:
                    if isinstance(part, dict) and "text" in part:
                        content_parts.append(part["text"])
                    else:
                        content_parts.append(str(part))
                
                message_content = "".join(content_parts)
                messages.append({"role": role, "content": message_content})
            
            # 准备生成配置，避免重复参数
            generation_config = gemini_request.generation_config.copy()
            # 移除已经在上面显式传递的参数
            generation_config.pop("max_output_tokens", None)
            generation_config.pop("temperature", None)
            generation_config.pop("top_p", None)
            generation_config.pop("top_k", None)
            
            # 确定正确的模型名称
            model_name = claude_request.model
            
            # 模型名称映射表
            model_mapping = {
                "gemini-2.5-pro": "gemini/gemini-2.0-flash-exp",
                "gemini-2.5-flash": "gemini/gemini-1.5-flash",
                "gemini-1.5-pro": "gemini/gemini-1.5-pro",
                "gemini-1.5-flash": "gemini/gemini-1.5-flash",
                "gemini-1.0-pro": "gemini/gemini-1.0-pro",
                "gemini-pro": "gemini/gemini-1.0-pro",
                "gemini-pro-vision": "gemini/gemini-pro-vision"
            }
            
            if model_name in model_mapping:
                model_name = model_mapping[model_name]
            elif not model_name.startswith("gemini/"):
                if model_name.startswith("gemini-"):
                    model_name = f"gemini/{model_name}"
                else:
                    model_name = f"gemini/{settings.DEFAULT_MODEL}"
            
            # 调用Gemini API - 使用正确的异步调用
            response = await litellm.acompletion(
                model=model_name,
                messages=messages,
                max_tokens=gemini_request.generation_config.get("max_output_tokens"),
                temperature=gemini_request.generation_config.get("temperature"),
                top_p=gemini_request.generation_config.get("top_p"),
                top_k=gemini_request.generation_config.get("top_k"),
                stream=False,
                **generation_config
            )
            
            # 转换响应格式
            if hasattr(response, 'model_dump'):
                response_dict = response.model_dump()
            elif hasattr(response, 'dict'):
                response_dict = response.dict()
            else:
                response_dict = response
                
            claude_response = self.convert_gemini_to_claude_response(
                response_dict, 
                claude_request
            )
            
            return claude_response
            
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            error_message = str(e)
            
            # 检查是否是API限制错误
            if "429" in error_message or "rate limit" in error_message.lower():
                raise RuntimeError("API配额已用完，请稍后重试或检查您的Google AI Studio配额")
            elif "invalid_api_key" in error_message.lower() or "unauthorized" in error_message.lower():
                raise RuntimeError("API密钥无效，请检查GEMINI_API_KEY设置")
            else:
                raise RuntimeError(f"Gemini API调用失败: {error_message}")
    
    async def convert_and_send_streaming(self, claude_request: ClaudeRequest):
        """
        转换请求并发送流式响应到Gemini API
        
        Args:
            claude_request: Claude API请求
            
        Yields:
            流式响应块
        """
        try:
            # 转换请求格式
            gemini_request = self.convert_claude_to_gemini_request(claude_request)
            
            # 准备LiteLLM调用参数
            messages = []
            for content in gemini_request.contents:
                role = "user" if content["role"] == "user" else "assistant"
                content_parts = []
                for part in content["parts"]:
                    if isinstance(part, dict) and "text" in part:
                        content_parts.append(part["text"])
                    else:
                        content_parts.append(str(part))
                
                message_content = "".join(content_parts)
                messages.append({"role": role, "content": message_content})
            
            # 准备生成配置，避免重复参数
            generation_config = gemini_request.generation_config.copy()
            # 移除已经在上面显式传递的参数
            generation_config.pop("max_output_tokens", None)
            generation_config.pop("temperature", None)
            generation_config.pop("top_p", None)
            generation_config.pop("top_k", None)
            
            # 确定正确的模型名称
            model_name = claude_request.model
            
            # 模型名称映射表
            model_mapping = {
                "gemini-2.5-pro": "gemini/gemini-2.0-flash-exp",
                "gemini-2.5-flash": "gemini/gemini-1.5-flash",
                "gemini-1.5-pro": "gemini/gemini-1.5-pro",
                "gemini-1.5-flash": "gemini/gemini-1.5-flash",
                "gemini-1.0-pro": "gemini/gemini-1.0-pro",
                "gemini-pro": "gemini/gemini-1.0-pro",
                "gemini-pro-vision": "gemini/gemini-pro-vision"
            }
            
            if model_name in model_mapping:
                model_name = model_mapping[model_name]
            elif not model_name.startswith("gemini/"):
                if model_name.startswith("gemini-"):
                    model_name = f"gemini/{model_name}"
                else:
                    model_name = f"gemini/{settings.DEFAULT_MODEL}"
            
            # 调用Gemini API获取流式响应
            response = await litellm.acompletion(
                model=model_name,
                messages=messages,
                max_tokens=gemini_request.generation_config.get("max_output_tokens"),
                temperature=gemini_request.generation_config.get("temperature"),
                top_p=gemini_request.generation_config.get("top_p"),
                top_k=gemini_request.generation_config.get("top_k"),
                stream=True,
                **generation_config
            )
            
            # 生成流式响应
            message_id = str(uuid.uuid4())
            
            # 发送消息开始
            start_data = {
                'type': 'message_start',
                'message': {
                    'id': message_id,
                    'type': 'message',
                    'role': 'assistant',
                    'content': [],
                    'model': claude_request.model
                }
            }
            yield f"data: {json.dumps(start_data)}\n\n"
            
            # 发送内容块
            content_index = 0
            
            # 正确处理LiteLLM的流式响应
            try:
                # 使用async for处理异步迭代器
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        
                        delta_data = {
                            'type': 'content_block_delta',
                            'index': content_index,
                            'delta': {
                                'type': 'text_delta',
                                'text': content
                            }
                        }
                        yield f"data: {json.dumps(delta_data)}\n\n"
                        
            except TypeError:
                # 处理同步迭代器的情况
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        
                        delta_data = {
                            'type': 'content_block_delta',
                            'index': content_index,
                            'delta': {
                                'type': 'text_delta',
                                'text': content
                            }
                        }
                        yield f"data: {json.dumps(delta_data)}\n\n"
            
            # 发送消息结束
            stop_data = {'type': 'message_stop'}
            yield f"data: {json.dumps(stop_data)}\n\n"
            
        except Exception as e:
            logger.error(f"流式响应失败: {e}")
            error_message = str(e)
            
            # 更详细的错误信息
            if "429" in error_message or "rate limit" in error_message.lower():
                user_message = "API配额已用完，请稍后重试或检查您的Google AI Studio配额"
            elif "invalid_api_key" in error_message.lower() or "unauthorized" in error_message.lower():
                user_message = "API密钥无效，请检查GEMINI_API_KEY设置"
            else:
                user_message = f"Gemini API调用失败: {error_message}"
                
            error_data = {
                'type': 'error',
                'error': {
                    'type': 'api_error',
                    'message': user_message
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"