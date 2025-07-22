"""
基础提供商接口
定义所有AI提供商必须实现的通用接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncGenerator, List, Optional
from dataclasses import dataclass
import uuid
from datetime import datetime


@dataclass
class ProviderRequest:
    """提供商请求数据结构"""
    messages: List[Dict[str, str]]
    model: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    stream: bool = False
    system: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            k: v for k, v in self.__dict__.items() 
            if v is not None
        }


@dataclass
class ProviderResponse:
    """提供商响应数据结构"""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    id: str = None
    created: datetime = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created is None:
            self.created = datetime.now()


@dataclass
class ProviderStreamChunk:
    """流式响应数据块"""
    content: str = ""
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    delta: Optional[str] = None
    is_final: bool = False


class BaseProvider(ABC):
    """基础AI提供商接口"""
    
    def __init__(self, name: str, api_key: str, base_url: Optional[str] = None):
        """初始化提供商
        
        Args:
            name: 提供商名称
            api_key: API密钥
            base_url: 自定义基础URL（可选）
        """
        self.name = name
        self.api_key = api_key
        self.base_url = base_url
    
    @abstractmethod
    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        """生成非流式响应
        
        Args:
            request: 请求对象
            
        Returns:
            响应对象
        """
        pass
    
    @abstractmethod
    async def generate_stream(self, request: ProviderRequest) -> AsyncGenerator[ProviderStreamChunk, None]:
        """生成流式响应
        
        Args:
            request: 请求对象
            
        Yields:
            流式数据块
        """
        pass
    
    @abstractmethod
    def validate_model(self, model: str) -> bool:
        """验证模型名称是否有效
        
        Args:
            model: 模型名称
            
        Returns:
            是否有效
        """
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """获取支持的模型列表
        
        Returns:
            模型名称列表
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """测试与提供商的连接
        
        Returns:
            测试结果字典
        """
        pass
    
    def get_provider_info(self) -> Dict[str, Any]:
        """获取提供商信息"""
        return {
            "name": self.name,
            "supported_models": self.get_supported_models(),
            "base_url": self.base_url
        }


class ProviderRegistry:
    """提供商注册中心"""
    
    _providers: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: type):
        """注册提供商类
        
        Args:
            name: 提供商名称
            provider_class: 提供商类
        """
        if not issubclass(provider_class, BaseProvider):
            raise ValueError(f"Provider class must inherit from BaseProvider")
        cls._providers[name] = provider_class
    
    @classmethod
    def get_provider(cls, name: str, api_key: str, **kwargs) -> BaseProvider:
        """获取提供商实例
        
        Args:
            name: 提供商名称
            api_key: API密钥
            **kwargs: 额外参数
            
        Returns:
            提供商实例
        """
        if name not in cls._providers:
            raise ValueError(f"Unknown provider: {name}")
        
        provider_class = cls._providers[name]
        return provider_class(api_key=api_key, **kwargs)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """列出所有注册的提供商"""
        return list(cls._providers.keys())


# 注册内置提供商
from . import providers

# 自动注册所有提供商类
ProviderRegistry.register("gemini", providers.GeminiProvider)
ProviderRegistry.register("openai", providers.OpenAIProvider)
ProviderRegistry.register("anthropic", providers.AnthropicProvider)
ProviderRegistry.register("azure", providers.AzureOpenAIProvider)
ProviderRegistry.register("ollama", providers.OllamaProvider)