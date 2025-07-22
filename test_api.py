#!/usr/bin/env python3
"""
直接测试LiteLLM API调用
"""

import asyncio
import litellm
import os

# 设置API密钥
litellm.api_key = os.getenv("GEMINI_API_KEY")

async def test_gemini_direct():
    """直接测试Gemini API"""
    try:
        print("=== 直接测试Gemini API ===")
        
        # 测试非流式调用
        response = await litellm.acompletion(
            model="gemini/gemini-1.5-pro",
            messages=[
                {"role": "user", "content": "测试gemini 2.5 pro"}
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        print("响应:")
        print(response)
        print("\n响应内容:")
        print(response.choices[0].message.content)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_gemini_direct())