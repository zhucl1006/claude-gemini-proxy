"""
测试数据模型
"""

import pytest
from src.models import ClaudeRequest, Role, Message, ContentBlock, ToolDefinition


class TestModels:
    """测试模型类"""
    
    def test_claude_request_creation(self):
        """测试ClaudeRequest创建"""
        request = ClaudeRequest(
            model="gemini-1.5-pro",
            messages=[
                Message(role=Role.USER, content="Hello, world!")
            ],
            max_tokens=100
        )
        
        assert request.model == "gemini-1.5-pro"
        assert len(request.messages) == 1
        assert request.messages[0].role == Role.USER
        assert request.messages[0].content == "Hello, world!"
    
    def test_temperature_validation(self):
        """测试温度参数验证"""
        with pytest.raises(ValueError):
            ClaudeRequest(
                model="gemini-1.5-pro",
                messages=[Message(role=Role.USER, content="test")],
                temperature=3.0  # 超出范围
            )
    
    def test_content_block_creation(self):
        """测试内容块创建"""
        text_block = ContentBlock(type="text", text="Hello")
        assert text_block.type == "text"
        assert text_block.text == "Hello"