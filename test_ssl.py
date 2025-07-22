#!/usr/bin/env python3
"""
测试SSL连接和证书验证
"""

import asyncio
import ssl
import httpx
import os

async def test_ssl_connection():
    """测试SSL连接"""
    try:
        print("=== 测试SSL连接 ===")
        
        # 测试Google API的SSL连接
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro"
        
        async with httpx.AsyncClient(
            verify=True,
            timeout=30.0
        ) as client:
            response = await client.get(url)
            print(f"状态码: {response.status_code}")
            print("SSL连接测试通过")
            
    except ssl.SSLError as e:
        print(f"SSL错误: {e}")
    except Exception as e:
        print(f"其他错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ssl_connection())