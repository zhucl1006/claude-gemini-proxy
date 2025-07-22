"""
OpenAI提供商实现
"""

import logging
from typing import Dict, Any, AsyncGenerator, List
import litellm
from litellm import completion

from .base import BaseProvider, ProviderRequest, ProviderResponse, ProviderStreamChunk


logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI提供商"""
    
    def __init__(self, api_key: str, base_url: str = None):
        super().__init__("openai", api_key, base_url)
        litellm.api_key = api_key
        if base_url:
            litellm.base_url = base_url
    
    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        """生成非流式响应"""
        try:
            response = await completion(
                model=f"openai/{request.model}",
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop_sequences,
                tools=request.tools,
                stream=False
            )
            
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
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def generate_stream(self, request: ProviderRequest) -> AsyncGenerator[ProviderStreamChunk, None]:
        """生成流式响应"""
        try:
            response = await completion(
                model=f"openai/{request.model}",
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop_sequences,
                tools=request.tools,
                stream=True
            )
            
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield ProviderStreamChunk(
                        content=chunk.choices[0].delta.content,
                        delta=chunk.choices[0].delta.content
                    )
                
                if chunk.usage:
                    yield ProviderStreamChunk(
                        usage={
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens
                        },
                        is_final=True
                    )
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise
    
    def validate_model(self, model: str) -> bool:
        """验证模型名称"""
        valid_models = [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ]
        return model in valid_models or model.startswith("gpt-")
    
    def get_supported_models(self) -> List[str]:
        """获取支持的模型列表"""
        return [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ]
    
    async def test_connection(self) -> Dict[str, Any]:
        """测试连接"""
        try:
            response = await completion(
                model="openai/gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            return {
                "status": "success",
                "message": "OpenAI API connection successful",
                "model": response.model
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }