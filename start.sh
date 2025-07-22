#!/bin/bash
# Claude Gemini Proxy 启动脚本

# 设置环境变量
export PYTHONPATH=src

# 检查环境变量
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ 错误: GEMINI_API_KEY 环境变量未设置"
    echo "请先设置: export GEMINI_API_KEY=your-gemini-api-key"
    exit 1
fi

# 检查端口占用
PORT=${PROXY_PORT:-3456}
if lsof -i:$PORT > /dev/null 2>&1; then
    echo "⚠️  端口 $PORT 已被占用，尝试使用备用端口..."
    PORT=3457
fi

# 启动服务
echo "🚀 启动 Claude Code Gemini 代理服务..."
echo "📍 服务将运行在 http://localhost:$PORT"
echo "🔍 API文档: http://localhost:$PORT/docs"
echo "❤️  健康检查: http://localhost:$PORT/health"
echo "🤖 默认模型: gemini-2.5-pro"
echo ""

# 使用uv运行
uv run uvicorn src.main:app --host 0.0.0.0 --port $PORT --reload