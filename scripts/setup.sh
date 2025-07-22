#!/bin/bash

# Claude Code Gemini Proxy 自动配置脚本
# 这个脚本会自动配置Claude Code以使用Gemini代理

set -e

echo "🚀 Claude Code Gemini Proxy 配置脚本"
echo "========================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查依赖
print_step "检查系统依赖..."

# 检查Python
if ! command -v python3 >/dev/null 2>&1; then
    print_error "Python 3 未安装"
    exit 1
fi

# 检查curl
if ! command -v curl >/dev/null 2>&1; then
    print_error "curl 未安装"
    exit 1
fi

print_status "系统依赖检查通过"

# 检查Gemini API密钥
print_step "检查Gemini API密钥..."

if [[ -z "$GEMINI_API_KEY" ]]; then
    if [[ -f .env ]]; then
        source .env
    fi
    
    if [[ -z "$GEMINI_API_KEY" ]]; then
        print_error "GEMINI_API_KEY 环境变量未设置"
        echo "请设置环境变量: export GEMINI_API_KEY=your-api-key"
        echo "或者在当前目录创建.env文件并添加: GEMINI_API_KEY=your-api-key"
        exit 1
    fi
fi

print_status "Gemini API密钥已配置"

# 检查Claude Code安装
print_step "检查Claude Code安装..."

CLAUDE_PATH=""
if command -v claude >/dev/null 2>&1; then
    CLAUDE_PATH=$(which claude)
    print_status "找到Claude Code: $CLAUDE_PATH"
elif [[ -f "/usr/local/bin/claude" ]]; then
    CLAUDE_PATH="/usr/local/bin/claude"
    print_status "找到Claude Code: $CLAUDE_PATH"
elif [[ -f "$HOME/.local/bin/claude" ]]; then
    CLAUDE_PATH="$HOME/.local/bin/claude"
    print_status "找到Claude Code: $CLAUDE_PATH"
else
    print_error "未找到Claude Code"
    echo "请确保Claude Code已正确安装"
    echo "安装指南: https://claude.ai/code"
    exit 1
fi

# 启动代理服务
print_step "启动代理服务..."

# 检查端口占用
if lsof -i :8080 >/dev/null 2>&1; then
    print_warning "端口8080已被占用"
    echo "您可以选择:"
    echo "1. 停止占用8080的服务"
    echo "2. 修改PROXY_PORT环境变量"
    read -p "按回车继续或输入其他端口号: " NEW_PORT
    
    if [[ -n "$NEW_PORT" ]]; then
        export PROXY_PORT=$NEW_PORT
        print_status "已设置端口为: $PROXY_PORT"
    else
        print_warning "使用默认端口8080"
    fi
fi

# 启动服务
print_status "正在启动代理服务..."
nohup python3 -m src.main > logs/proxy.log 2>&1 &
echo $! > logs/proxy.pid

sleep 3

# 检查服务是否启动成功
if curl -f http://localhost:${PROXY_PORT:-8080}/health >/dev/null 2>&1; then
    print_status "代理服务启动成功"
else
    print_error "代理服务启动失败"
    cat logs/proxy.log
    exit 1
fi

# 配置Claude Code
print_step "配置Claude Code..."

PROXY_URL="http://localhost:${PROXY_PORT:-8080}"

# 配置Claude Code API URL
if $CLAUDE_PATH config set api_url "$PROXY_URL/v1/messages" 2>/dev/null; then
    print_status "已配置Claude Code API URL: $PROXY_URL/v1/messages"
else
    # 如果config命令不存在，使用环境变量方式
    print_status "使用环境变量配置方式"
    echo "export ANTHROPIC_BASE_URL=$PROXY_URL"
    echo "export ANTHROPIC_API_KEY=dummy-key"
fi

# 创建启动脚本
cat > start_proxy.sh << 'EOF'
#!/bin/bash
echo "启动Claude Code Gemini代理..."
python3 -m src.main
EOF

chmod +x start_proxy.sh

# 创建停止脚本
cat > stop_proxy.sh << 'EOF'
#!/bin/bash
echo "停止Claude Code Gemini代理..."
if [[ -f logs/proxy.pid ]]; then
    kill $(cat logs/proxy.pid) 2>/dev/null || true
    rm logs/proxy.pid
    echo "代理服务已停止"
else
    echo "代理服务未运行"
fi
EOF

chmod +x stop_proxy.sh

# 创建环境变量配置脚本
cat > set_env.sh << 'EOF'
#!/bin/bash
# 设置环境变量
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_API_KEY=dummy-key

echo "环境变量已设置:"
echo "ANTHROPIC_BASE_URL=$ANTHROPIC_BASE_URL"
echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"
EOF

chmod +x set_env.sh

# 测试连接
print_step "测试连接..."

if curl -s -X POST "$PROXY_URL/v1/messages" \
    -H "Content-Type: application/json" \
    -H "x-api-key: dummy-key" \
    -d '{"model":"gemini-1.5-pro","messages":[{"role":"user","content":"Hello"}],"max_tokens":10}' \
    >/dev/null 2>&1; then
    print_status "✅ 连接测试通过"
else
    print_warning "⚠️ 连接测试失败，请检查日志"
fi

# 创建日志目录
mkdir -p logs

# 显示配置信息
echo ""
echo "🎉 配置完成！"
echo "=================="
echo ""
echo "服务信息:"
echo "  代理地址: $PROXY_URL"
echo "  健康检查: $PROXY_URL/health"
echo "  日志文件: logs/proxy.log"
echo ""
echo "使用方法:"
echo "  1. 启动代理: ./start_proxy.sh"
echo "  2. 停止代理: ./stop_proxy.sh"
echo "  3. 设置环境: source set_env.sh"
echo ""
echo "测试Claude Code:"
echo "  claude '你好，请介绍一下这个项目'"
echo ""
echo "手动配置（如果需要）:"
echo "  export ANTHROPIC_BASE_URL=$PROXY_URL"
echo "  export ANTHROPIC_API_KEY=dummy-key"
echo ""
print_status "配置完成！开始享受使用Gemini的Claude Code吧！"