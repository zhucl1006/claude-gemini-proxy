#!/bin/bash

# Claude Gemini Proxy 停止脚本

echo "🛑 停止 Claude Code Gemini Proxy..."

# 查找并停止Python进程
PIDS=$(pgrep -f "src.main")

if [ -n "$PIDS" ]; then
    echo "找到运行中的进程: $PIDS"
    kill $PIDS
    echo "✅ 进程已停止"
else
    echo "没有运行中的进程"
fi

# 清理PID文件（如果存在）
if [ -f .service_pid ]; then
    rm .service_pid
fi

echo "服务已停止"