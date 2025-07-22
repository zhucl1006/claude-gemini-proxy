"""
Google Gemini提供商实现
"""

import json
import logging
from typing import Dict, Any, AsyncGenerator, List
import litellm
from litellm import completion

from .base import BaseProvider, ProviderRequest, ProviderResponse, ProviderStreamChunk


logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """Google Gemini AI提供商"""
    
    def __init__(self, api_key: str, base_url: str = None):
        super().__init__("gemini", api_key, base_url)
        litellm.api_key = api_key
        if base_url:
            litellm.base_url = base_url
    
    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        """生成非流式响应"""
        try:
            # 转换消息格式
            messages = self._convert_messages(request.messages)
            
            # 调用Gemini API
            response = await completion(
                model=f"gemini/{request.model}",
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stop_sequences=request.stop_sequences,
                stream=False
            )
            
            # 提取响应内容
            content = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return ProviderResponse(
                content=content,
                model=response.model,
                usage=usage,
                finish_reason=response.choices[0].finish_reason
            )
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    async def generate_stream(self, request: ProviderRequest) -> AsyncGenerator[ProviderStreamChunk, None]:
        """生成流式响应"""
        try:
            messages = self._convert_messages(request.messages)
            
            response = await completion(
                model=f"gemini/{request.model}",
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stop_sequences=request.stop_sequences,
                stream=True
            )
            
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield ProviderStreamChunk(
                        content=chunk.choices[0].delta.content,
                        delta=chunk.choices[0].delta.content
                    )
                
                # 最后一块包含使用统计
                if chunk.usage:
                    yield ProviderStreamChunk(
                        content="",
                        usage={
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens
                        },
                        is_final=True
                    )
                    
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise
    
    def validate_model(self, model: str) -> bool:
        """验证模型名称"""
        valid_models = [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
            "gemini-pro",
            "gemini-pro-vision"
        ]
        return model in valid_models or model.startswith("gemini/")
    
    def get_supported_models(self) -> List[str]:
        """获取支持的模型列表"""
        return [
            "gemini-1.5-pro",
            "gemini-1.5-flash", 
            "gemini-1.0-pro",
            "gemini-pro",
            "gemini-pro-vision"
        ]
    
    async def test_connection(self) -> Dict[str, Any]:
        """测试连接"""
        try:
            response = await completion(
                model="gemini/gemini-1.5-flash",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            return {
                "status": "success",
                "message": "Gemini API connection successful",
                "model": response.model
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """转换消息格式"""
        return messages
    
    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """转换工具定义格式"""
        converted_tools = []
        for tool in tools:
            if "function" in tool:
                converted_tools.append(tool)
            else:
                # 转换为Gemini格式
                converted_tools.append({
                    "type": "function",
                    "function": tool
                })
        return converted_tools