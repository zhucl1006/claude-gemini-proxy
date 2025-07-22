#!/bin/bash

# Claude Gemini Proxy 启动脚本
# 使用方法: ./scripts/start.sh

echo "🚀 启动 Claude Code Gemini Proxy..."

# 检查Python环境
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ Python 3 未安装"
    exit 1
fi

# 检查依赖
if ! python3 -c "import fastapi" >/dev/null 2>&1; then
    echo "📦 正在安装依赖..."
    pip install -r requirements.txt
fi

# 检查环境变量
if [[ -z "$GEMINI_API_KEY" ]]; then
    if [[ -f .env ]]; then
        source .env
        echo "✅ 从.env文件加载配置"
    else
        echo "⚠️  未找到.env文件，请确保GEMINI_API_KEY环境变量已设置"
    fi
fi

# 创建日志目录
mkdir -p logs

# 启动服务
echo "🌐 服务将在 http://localhost:8080 启动"
echo "📊 健康检查: http://localhost:8080/health"
echo "📚 API文档: http://localhost:8080/docs"
echo ""

python3 -m src.main