#!/usr/bin/env python3
"""
调试测试脚本
"""

import asyncio
import json
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import ClaudeRequest, Role
from converter import ClaudeToGeminiConverter

async def test_conversion():
    """测试请求转换"""
    converter = ClaudeToGeminiConverter()
    
    # 创建测试请求
    request = ClaudeRequest(
        model="gemini-1.5-pro",
        messages=[
            {"role": Role.USER, "content": "测试gemini 2.5 pro"}
        ]
    )
    
    print("=== 原始Claude请求 ===")
    print(json.dumps(request.dict(), indent=2, ensure_ascii=False))
    
    # 转换为Gemini格式
    gemini_request = converter.convert_claude_to_gemini_request(request)
    
    print("\n=== 转换后的Gemini请求 ===")
    print(json.dumps(gemini_request.dict(), indent=2, ensure_ascii=False))
    
    # 检查消息转换
    for content in gemini_request.contents:
        print(f"\n角色: {content['role']}")
        print(f"内容: {content['parts']}")

if __name__ == "__main__":
    asyncio.run(test_conversion())