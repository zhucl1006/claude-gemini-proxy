"""
Pydantic数据模型
定义Claude API和Gemini API的数据结构
"""

from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class Role(str, Enum):
    """消息角色枚举"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ContentType(str, Enum):
    """内容类型枚举"""
    TEXT = "text"
    IMAGE = "image"


class ImageSource(BaseModel):
    """图像源信息"""
    type: str = Field(default="base64", description="图像源类型")
    media_type: str = Field(description="图像MIME类型，如image/jpeg")
    data: str = Field(description="base64编码的图像数据")


class ContentBlock(BaseModel):
    """内容块模型"""
    type: str = Field(description="内容类型: text, image")
    text: Optional[str] = Field(None, description="文本内容")
    source: Optional[ImageSource] = Field(None, description="图像源信息")


class Message(BaseModel):
    """消息模型"""
    role: Role = Field(description="消息角色")
    content: Union[str, List[ContentBlock]] = Field(description="消息内容")


class ToolParameter(BaseModel):
    """工具参数模型 - 兼容JSON Schema格式"""
    type: Optional[Union[str, List[str]]] = Field(None, description="参数类型: string, number, boolean, array, object")
    description: Optional[str] = Field(None, description="参数描述")
    enum: Optional[List[Any]] = Field(None, description="枚举值")
    properties: Optional[Dict[str, 'ToolParameter']] = Field(None, description="对象属性")
    required: Optional[List[str]] = Field(None, description="必需属性列表")
    items: Optional['ToolParameter'] = Field(None, description="数组项定义")
    default: Optional[Any] = Field(None, description="默认值")
    minimum: Optional[float] = Field(None, description="最小值")
    maximum: Optional[float] = Field(None, description="最大值")
    minLength: Optional[int] = Field(None, description="最小长度")
    maxLength: Optional[int] = Field(None, description="最大长度")
    pattern: Optional[str] = Field(None, description="正则表达式模式")
    additionalProperties: Optional[Union[bool, Dict[str, Any]]] = Field(None, description="是否允许额外属性或额外属性schema")
    schema_: Optional[str] = Field(None, alias="$schema", description="JSON Schema版本")
    ref_: Optional[str] = Field(None, alias="$ref", description="引用定义")
    
    # 新增JSON Schema兼容字段
    oneOf: Optional[List['ToolParameter']] = Field(None, description="oneOf验证")
    anyOf: Optional[List['ToolParameter']] = Field(None, description="anyOf验证")
    allOf: Optional[List['ToolParameter']] = Field(None, description="allOf验证")
    not_: Optional['ToolParameter'] = Field(None, description="not验证", alias="not")
    
    # 对象验证字段
    minProperties: Optional[int] = Field(None, description="最小属性数")
    maxProperties: Optional[int] = Field(None, description="最大属性数")
    
    # 数组验证字段
    minItems: Optional[int] = Field(None, description="最小项目数")
    maxItems: Optional[int] = Field(None, description="最大项目数")
    uniqueItems: Optional[bool] = Field(None, description="项目是否唯一")
    
    # 字符串验证字段
    format: Optional[str] = Field(None, description="字符串格式")
    
    # 数字验证字段
    exclusiveMinimum: Optional[float] = Field(None, description="排除最小值")
    exclusiveMaximum: Optional[float] = Field(None, description="排除最大值")
    multipleOf: Optional[float] = Field(None, description="倍数验证")
    
    # 允许任意额外字段以支持完整的JSON Schema
    class Config:
        extra = "allow"


class ToolDefinition(BaseModel):
    """工具定义模型"""
    name: str = Field(description="工具名称")
    description: str = Field(description="工具描述")
    input_schema: ToolParameter = Field(description="输入模式")


class ToolUse(BaseModel):
    """工具使用模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="工具调用ID")
    type: str = Field(default="tool_use", description="类型标识")
    name: str = Field(description="工具名称")
    input: Dict[str, Any] = Field(description="工具输入参数")


class ToolResult(BaseModel):
    """工具结果模型"""
    type: str = Field(default="tool_result", description="类型标识")
    tool_use_id: str = Field(description="对应的工具调用ID")
    content: Union[str, Dict[str, Any]] = Field(description="工具执行结果")
    is_error: bool = Field(default=False, description="是否为错误结果")


class ClaudeRequest(BaseModel):
    """Claude API请求模型"""
    model: str = Field(description="模型名称")
    messages: List[Message] = Field(description="消息列表")
    max_tokens: int = Field(default=4000, description="最大token数")
    temperature: Optional[float] = Field(default=0.7, description="温度参数")
    top_p: Optional[float] = Field(default=0.8, description="top_p参数")
    top_k: Optional[int] = Field(default=40, description="top_k参数")
    stop_sequences: Optional[List[str]] = Field(None, description="停止序列")
    tools: Optional[List[ToolDefinition]] = Field(None, description="可用工具列表")
    stream: bool = Field(default=False, description="是否启用流式响应")
    system: Optional[Union[str, List[Dict[str, Any]]]] = Field(None, description="系统提示或系统消息列表")

    @validator('temperature')
    def validate_temperature(cls, v):
        if v is not None and (v < 0 or v > 2):
            raise ValueError('温度参数必须在0到2之间')
        return v

    @validator('top_p')
    def validate_top_p(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('top_p参数必须在0到1之间')
        return v

    @validator('top_k')
    def validate_top_k(cls, v):
        if v is not None and v < 1:
            raise ValueError('top_k参数必须大于0')
        return v


class ClaudeResponse(BaseModel):
    """Claude API响应模型"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="响应ID")
    type: str = Field(default="message", description="响应类型")
    role: str = Field(default="assistant", description="角色")
    content: List[ContentBlock] = Field(description="响应内容")
    model: str = Field(description="使用的模型")
    stop_reason: Optional[str] = Field(None, description="停止原因")
    stop_sequence: Optional[str] = Field(None, description="停止序列")
    usage: Dict[str, int] = Field(description="token使用情况")


class ClaudeStreamResponse(BaseModel):
    """Claude流式响应模型"""
    type: str = Field(description="响应类型: message_start, content_block_delta, message_delta, message_stop")
    message: Optional[Dict[str, Any]] = Field(None, description="消息信息")
    index: Optional[int] = Field(None, description="内容块索引")
    delta: Optional[Dict[str, Any]] = Field(None, description="增量内容")
    usage: Optional[Dict[str, int]] = Field(None, description="token使用情况")


class GeminiRequest(BaseModel):
    """Gemini API请求模型"""
    contents: List[Dict[str, Any]] = Field(description="内容列表")
    generation_config: Dict[str, Any] = Field(description="生成配置")
    safety_settings: Optional[List[Dict[str, Any]]] = Field(None, description="安全设置")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="工具定义")


class GeminiResponse(BaseModel):
    """Gemini API响应模型"""
    candidates: List[Dict[str, Any]] = Field(description="候选响应")
    usage_metadata: Dict[str, int] = Field(description="使用情况元数据")
    model_version: str = Field(description="模型版本")


class ErrorResponse(BaseModel):
    """错误响应模型"""
    type: str = Field(default="error", description="错误类型")
    error: Dict[str, Any] = Field(description="错误详情")


class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(description="状态")
    timestamp: float = Field(description="时间戳")
    service: str = Field(description="服务名称")
    version: Optional[str] = Field(None, description="版本信息")


# 工具参数模型的前向引用
ToolParameter.update_forward_refs()