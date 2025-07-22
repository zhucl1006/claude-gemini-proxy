"""
Anthropic Claude提供商实现
"""

import logging
from typing import Dict, Any, AsyncGenerator, List
import litellm
from litellm import completion

from .base import BaseProvider, ProviderRequest, ProviderResponse, ProviderStreamChunk


logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Anthropic Claude提供商"""
    
    def __init__(self, api_key: str, base_url: str = None):
        super().__init__("anthropic", api_key, base_url)
        litellm.api_key = api_key
        if base_url:
            litellm.base_url = base_url
    
    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        """生成非流式响应"""
        try:
            response = await completion(
                model=f"anthropic/{request.model}",
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stop_sequences=request.stop_sequences,
                tools=request.tools,
                system=request.system,
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
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def generate_stream(self, request: ProviderRequest) -> AsyncGenerator[ProviderStreamChunk, None]:
        """生成流式响应"""
        try:
            response = await completion(
                model=f"anthropic/{request.model}",
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stop_sequences=request.stop_sequences,
                tools=request.tools,
                system=request.system,
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
            logger.error(f"Anthropic streaming error: {e}")
            raise
    
    def validate_model(self, model: str) -> bool:
        """验证模型名称"""
        valid_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
        return model in valid_models or model.startswith("claude-")
    
    def get_supported_models(self) -> List[str]:
        """获取支持的模型列表"""
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
    
    async def test_connection(self) -> Dict[str, Any]:
        """测试连接"""
        try:
            response = await completion(
                model="anthropic/claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            return {
                "status": "success",
                "message": "Anthropic API connection successful",
                "model": response.model
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }