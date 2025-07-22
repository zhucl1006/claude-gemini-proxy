"""
配置管理模块
基于server.py的成熟配置管理实现
"""

import os
import re
import sys
from typing import Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    
    # ===== API配置 =====
    GEMINI_API_KEY: str = Field(..., description="Google Gemini API密钥")
    
    # 模型配置
    BIG_MODEL: str = Field(default="gemini-2.5-pro", description="大型模型")
    SMALL_MODEL: str = Field(default="gemini-2.5-flash", description="小型模型")
    
    # ===== 服务器配置 =====
    HOST: str = Field(default="0.0.0.0", description="服务器监听地址")
    PORT: int = Field(default=8080, description="服务器监听端口")
    LOG_LEVEL: str = Field(default="WARNING", description="日志级别")
    
    # ===== 连接配置 =====
    REQUEST_TIMEOUT: int = Field(default=120, description="请求超时时间(秒)")
    MAX_RETRIES: int = Field(default=3, description="最大重试次数")
    
    # ===== 流式配置 =====
    MAX_STREAMING_RETRIES: int = Field(default=12, description="流式重试次数")
    FORCE_DISABLE_STREAMING: bool = Field(default=True, description="强制禁用流式")
    EMERGENCY_DISABLE_STREAMING: bool = Field(default=True, description="紧急禁用流式")
    
    # ===== 代理配置 =====
    HTTPS_PROXY: Optional[str] = Field(default=None, description="HTTPS代理地址")
    HTTP_PROXY: Optional[str] = Field(default=None, description="HTTP代理地址")
    
    # ===== 内容限制 =====
    MAX_TOKENS_LIMIT: int = Field(default=8192, description="最大token限制")
    
    # ===== 传统兼容配置 =====
    # 保持向后兼容的配置项
    PROXY_HOST: str = Field(default="0.0.0.0", description="代理服务器监听地址(兼容)")
    PROXY_PORT: int = Field(default=3456, description="代理服务器监听端口(兼容)")
    DEFAULT_MODEL: str = Field(default="gemini-2.5-pro", description="默认模型(兼容)")
    ENABLE_STREAMING: bool = Field(default=True, description="是否启用流式响应(兼容)")
    MAX_TOKENS: int = Field(default=4000, description="最大响应token数(兼容)")
    TEMPERATURE: float = Field(default=0.7, description="模型温度参数(兼容)")
    TOP_P: float = Field(default=0.8, description="top_p采样参数(兼容)")
    TOP_K: int = Field(default=40, description="top_k采样参数(兼容)")
    TIMEOUT_SECONDS: int = Field(default=120, description="请求超时时间(兼容)")
    RETRY_DELAY: float = Field(default=1.0, description="重试延迟(兼容)")
    
    # 网络配置
    CONNECTION_POOL_SIZE: int = Field(default=100, description="连接池大小")
    PROXY_ENABLED: bool = Field(default=False, description="是否启用代理")
    PROXY_URL: str = Field(default="http://127.0.0.1:7890", description="默认代理地址")
    
    # 安全配置
    ALLOWED_ORIGINS: List[str] = Field(default=["*"], description="允许的CORS源")
    API_RATE_LIMIT: str = Field(default="100/minute", description="API速率限制")
    
    # 监控配置
    ENABLE_METRICS: bool = Field(default=True, description="是否启用性能监控")
    HEALTH_CHECK_INTERVAL: int = Field(default=30, description="健康检查间隔")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # 忽略未定义的环境变量
    
    @validator("GEMINI_API_KEY")
    def validate_gemini_api_key(cls, v):
        """验证Gemini API密钥格式"""
        if not v:
            raise ValueError("GEMINI_API_KEY 不能为空")
        
        # 基本格式验证
        if not (v.startswith('AIza') and len(v) == 39):
            # 发出警告但不阻止，因为可能有新的密钥格式
            print("⚠️ 警告: API密钥格式可能不正确，请检查Google AI Studio获取的密钥")
        
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """验证日志级别"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"日志级别必须是 {valid_levels} 之一")
        return v.upper()
    
    @property
    def effective_host(self) -> str:
        """获取实际使用的主机地址"""
        return self.HOST or self.PROXY_HOST
    
    @property
    def effective_port(self) -> int:
        """获取实际使用的端口"""
        return self.PORT or self.PROXY_PORT
    
    @property
    def effective_proxy_url(self) -> Optional[str]:
        """获取实际使用的代理地址"""
        if self.HTTPS_PROXY:
            return self.HTTPS_PROXY
        elif self.HTTP_PROXY:
            return self.HTTP_PROXY
        elif self.PROXY_ENABLED:
            return self.PROXY_URL
        return None
    
    def get_gemini_models(self) -> List[str]:
        """获取可用的Gemini模型列表"""
        return [
            "gemini-2.5-pro",
            "gemini-2.5-flash", 
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-2.5-pro-preview-05-20",
            "gemini-2.5-flash-preview-05-20"
        ]
    
    def validate_api_key(self) -> bool:
        """验证API密钥的有效性"""
        if not self.GEMINI_API_KEY:
            return False
        
        # 基本格式检查
        if not (self.GEMINI_API_KEY.startswith('AIza') and len(self.GEMINI_API_KEY) == 39):
            return False
        
        return True


# 全局设置实例
settings = Settings()


def validate_config():
    """验证配置的有效性"""
    required_vars = ["GEMINI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"缺少必需的环境变量: {', '.join(missing_vars)}")
    
    return True


def get_gemini_models():
    """获取可用的Gemini模型列表"""
    return [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.0-pro",
        "gemini-pro",
        "gemini-pro-vision"
    ]


def is_valid_model(model: str) -> bool:
    """检查模型名称是否有效"""
    return model in get_gemini_models()


# 创建配置验证
if __name__ == "__main__":
    try:
        validate_config()
        print("✅ 配置验证通过")
        print(f"📍 服务地址: {settings.PROXY_HOST}:{settings.PROXY_PORT}")
        print(f"🤖 默认模型: {settings.DEFAULT_MODEL}")
        print(f"🌊 流式响应: {'启用' if settings.ENABLE_STREAMING else '禁用'}")
    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        exit(1)