# Claude Gemini Proxy

🚀 **让Claude Code支持任意AI模型的通用代理服务**

一个基于Python + FastAPI + LiteLLM的高性能代理服务，让Claude Code能够无缝使用Gemini、GPT-4、Claude等50+种AI模型。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com)
[![LiteLLM](https://img.shields.io/badge/LiteLLM-Latest-orange.svg)](https://litellm.ai)
[![uv](https://img.shields.io/badge/uv-Latest-purple.svg)](https://github.com/astral-sh/uv)

## ✨ 核心特性

- 🎯 **完全兼容Claude Code** - 无需修改任何客户端代码
- 🌐 **支持50+AI模型** - Gemini、GPT-4、Claude、本地模型等
- ⚡ **高性能异步处理** - 支持流式响应和并发请求
- 🛠️ **智能错误恢复** - 自动重试和错误处理机制
- 🔧 **简单配置** - 一个环境变量即可开始使用
- 📊 **完整监控** - 健康检查和性能监控
- 🐳 **多种部署方式** - 本地、Docker、云服务

## 🚀 快速开始（5分钟上手）

### 方式1: 使用uv（推荐，更快）

```bash
# 安装uv（如果还没安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆项目
git clone https://github.com/zhucl1006/claude-gemini-proxy.git
cd claude-gemini-proxy

# 自动创建虚拟环境并安装依赖
uv sync
```

### 方式2: 使用传统pip

```bash
# 克隆项目
git clone https://github.com/zhucl1006/claude-gemini-proxy.git
cd claude-gemini-proxy

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制配置文件
cp .env.example .env

# 编辑配置（至少设置一个AI服务的API密钥）
export GEMINI_API_KEY="your-gemini-api-key"
# 或者
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. 启动服务

```bash
# 使用uv启动（推荐）
uv run python src/main.py

# 或使用传统方式
python src/main.py
```

服务将在 `http://localhost:8080` 启动

### 4. 配置Claude Code

```bash
# 方式1: 环境变量（推荐）
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_API_KEY=dummy-key

# 方式2: Claude配置文件
claude config set api_url http://localhost:8080
claude config set api_key dummy-key
```

### 5. 开始使用

```bash
claude "请帮我分析这段Python代码的性能"
```

🎉 **恭喜！现在Claude Code已经在使用您配置的AI模型了！**

## 📋 支持的AI服务

### 🌟 主要AI服务

| 服务商 | 模型示例 | 环境变量 | 说明 |
|--------|----------|----------|------|
| **Google** | gemini-1.5-pro, gemini-1.5-flash | `GEMINI_API_KEY` | 推荐，性价比高 |
| **OpenAI** | gpt-4, gpt-4o, gpt-3.5-turbo | `OPENAI_API_KEY` | 功能强大 |
| **Anthropic** | claude-3-5-sonnet, claude-3-haiku | `ANTHROPIC_API_KEY` | 原生Claude |
| **Azure OpenAI** | gpt-4, gpt-35-turbo | `AZURE_API_KEY` | 企业级 |
| **AWS Bedrock** | claude-v2, titan-text | `AWS_ACCESS_KEY_ID` | 云原生 |
| **Ollama** | llama2, codellama, mistral | - | 本地部署 |

### 🔧 模型配置示例

```bash
# 使用Gemini（推荐）
export GEMINI_API_KEY="AIza..."
export DEFAULT_MODEL="gemini-1.5-pro"

# 使用GPT-4
export OPENAI_API_KEY="sk-..."
export DEFAULT_MODEL="gpt-4"

# 使用本地Ollama
export DEFAULT_MODEL="ollama/llama2"
```

### 代理配置示例

```bash
# 启用代理（适用于国内网络环境）
export PROXY_ENABLED=true
export PROXY_URL=http://127.0.0.1:7890

# 或者使用自定义代理地址
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

# 验证代理连接
curl http://localhost:8080/test-connection
```

## ⚙️ 详细配置

### 环境变量说明

```bash
# === AI服务配置 ===
GEMINI_API_KEY=""           # Google Gemini API密钥
OPENAI_API_KEY=""           # OpenAI API密钥
ANTHROPIC_API_KEY=""        # Anthropic API密钥
AZURE_API_KEY=""            # Azure OpenAI API密钥

# === 服务配置 ===
PROXY_HOST="0.0.0.0"        # 服务器地址
PROXY_PORT="8080"           # 服务器端口
DEFAULT_MODEL="gemini-1.5-pro"  # 默认AI模型

# === 代理配置 ===
PROXY_ENABLED="false"       # 是否启用代理
PROXY_URL="http://127.0.0.1:7890"  # 默认代理地址
HTTP_PROXY=""               # 自定义HTTP代理
HTTPS_PROXY=""              # 自定义HTTPS代理

# === 功能配置 ===
ENABLE_STREAMING="true"     # 启用流式响应
MAX_TOKENS="8192"          # 最大token数
TIMEOUT_SECONDS="120"      # 请求超时时间
LOG_LEVEL="INFO"           # 日志级别

# === 高级配置 ===
ENABLE_MULTI_MODEL="false" # 启用多模型支持
LOAD_BALANCING="false"     # 启用负载均衡
COST_OPTIMIZATION="false"  # 启用成本优化
```

### 多模型配置

```bash
# 启用多模型支持
export ENABLE_MULTI_MODEL="true"

# 配置模型映射
export MODEL_MAPPING='{
  "haiku": "gemini-1.5-flash",
  "sonnet": "gemini-1.5-pro", 
  "opus": "gpt-4",
  "coding": "ollama/codellama"
}'
```

## 🔗 Claude Code集成详解

### 方式1: 环境变量配置（推荐）

```bash
# 1. 启动代理服务
uv run python src/main.py  # 或 python src/main.py

# 2. 设置环境变量
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_API_KEY=dummy-key

# 3. 验证连接
curl http://localhost:8080/health

# 4. 使用Claude Code
claude "Hello, world!"
```

### 方式2: 配置文件

```bash
# 1. 修改Claude配置
claude config set api_url http://localhost:8080
claude config set api_key dummy-key

# 2. 验证配置
claude config show

# 3. 测试连接
claude "测试连接"
```

### 方式3: 临时使用

```bash
# 临时设置（仅当前会话有效）
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_API_KEY=dummy claude "临时测试"
```

### 一键启动脚本

```bash
# 创建启动脚本
cat > start.sh << 'EOF'
#!/bin/bash
export GEMINI_API_KEY="your-api-key-here"
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_API_KEY=dummy-key

echo "启动Claude Gemini Proxy..."
uv run python src/main.py &
PROXY_PID=$!

echo "等待服务启动..."
sleep 3

echo "测试连接..."
curl -s http://localhost:8080/health

echo "代理服务已启动，PID: $PROXY_PID"
echo "现在可以使用Claude Code了！"
EOF

chmod +x start.sh
./start.sh
```

## 🐳 部署方式

### Docker部署

```bash
# 构建镜像
docker build -t claude-gemini-proxy .

# 运行容器
docker run -d \
  -p 8080:8080 \
  -e GEMINI_API_KEY="your-api-key" \
  --name claude-gemini-proxy \
  claude-gemini-proxy
```

### Docker Compose

```yaml
version: '3.8'
services:
  claude-gemini-proxy:
    build: .
    ports:
      - "8080:8080"
    environment:
      - GEMINI_API_KEY=your-api-key
      - DEFAULT_MODEL=gemini-1.5-pro
    restart: unless-stopped
```

### 代理配置

```yaml
# docker-compose.yml 带代理配置
version: '3.8'
services:
  claude-gemini-proxy:
    build: .
    ports:
      - "8080:8080"
    environment:
      - GEMINI_API_KEY=your-api-key
      - DEFAULT_MODEL=gemini-1.5-pro
      - PROXY_ENABLED=true
      - PROXY_URL=http://127.0.0.1:7890
    restart: unless-stopped
```

### 云服务部署

```bash
# Railway
railway deploy

# Render
render deploy

# Heroku
heroku create your-app-name
git push heroku main
```

## 📊 API文档

### 核心端点

- `POST /v1/messages` - 消息处理（兼容Anthropic API）
- `POST /v1/messages/count_tokens` - Token计数
- `GET /health` - 健康检查
- `GET /test-connection` - 连接测试
- `GET /models` - 支持的模型列表

### 请求示例

```bash
# 发送消息
curl -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-1.5-pro",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'

# 健康检查
curl http://localhost:8080/health
```

## 🛠️ 故障排除

### 常见问题

**Q: Claude Code连接失败**
```bash
# 检查服务状态
curl http://localhost:8080/health

# 检查环境变量
echo $ANTHROPIC_BASE_URL
echo $ANTHROPIC_API_KEY
```

**Q: API密钥错误**
```bash
# 测试API连接
curl http://localhost:8080/test-connection
```

**Q: 代理连接问题**
```bash
# 检查代理配置
echo $PROXY_ENABLED
echo $PROXY_URL

# 测试代理连接
curl http://localhost:8080/test-connection
```

**Q: 流式响应中断**
```bash
# 禁用流式响应
export ENABLE_STREAMING="false"
```

### 日志调试

```bash
# 启用详细日志
export LOG_LEVEL="DEBUG"
python src/main.py

# 查看日志文件
tail -f logs/proxy.log
```

## 🚀 高级功能

### 智能模型选择

```python
# 根据任务类型自动选择模型
TASK_MODEL_MAPPING = {
    "code": "gpt-4",           # 代码任务使用GPT-4
    "chat": "gemini-1.5-pro",  # 对话使用Gemini
    "quick": "gemini-1.5-flash" # 快速任务使用Flash
}
```

### 负载均衡

```bash
# 启用负载均衡
export LOAD_BALANCING="true"
export BACKUP_MODELS="gpt-4,claude-3-sonnet"
```

### 成本优化

```bash
# 启用成本优化
export COST_OPTIMIZATION="true"
export COST_THRESHOLD="0.01"  # 每请求成本阈值
```

## 📈 性能监控

### 监控端点

- `GET /metrics` - Prometheus指标
- `GET /stats` - 使用统计
- `GET /logs` - 实时日志

### 性能指标

- 请求延迟
- 成功率
- Token使用量
- 成本统计

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [LiteLLM](https://litellm.ai) - 统一的LLM API接口
- [FastAPI](https://fastapi.tiangolo.com) - 现代化的Python Web框架
- [Claude Code](https://claude.ai) - 强大的AI编程助手

---

⭐ **如果这个项目对您有帮助，请给我们一个Star！**

## 🔮 未来扩展计划

### 即将支持的功能
- [ ] **多模态支持** - 图像、音频、视频处理
- [ ] **插件系统** - 自定义AI服务集成
- [ ] **Web界面** - 图形化配置和监控
- [ ] **集群部署** - 高可用性和水平扩展
- [ ] **智能缓存** - 减少API调用成本
- [ ] **A/B测试** - 模型效果对比

### 扩展到其他AI服务

本项目的架构设计支持轻松扩展到任何AI服务：

```python
# 添加新的AI服务只需要：
# 1. 在LiteLLM中注册新服务
# 2. 添加环境变量配置
# 3. 更新模型映射

# 示例：添加新的AI服务
NEW_AI_SERVICES = {
    "cohere": "cohere/command-r-plus",
    "huggingface": "huggingface/microsoft/DialoGPT-large",
    "replicate": "replicate/meta/llama-2-70b-chat",
    "together": "together_ai/togethercomputer/llama-2-7b-chat"
}
```

## 🏗️ 架构设计

### 系统架构图

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Claude Code   │───▶│  Proxy Server   │───▶│   AI Services   │
│                 │    │                 │    │                 │
│ - 发送请求      │    │ - API转换       │    │ - Gemini        │
│ - 接收响应      │    │ - 流式处理      │    │ - GPT-4         │
│ - 工具调用      │    │ - 错误处理      │    │ - Claude        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 核心组件

1. **API转换器** (`converter.py`)
   - Anthropic ↔ LiteLLM 格式转换
   - 工具调用格式处理
   - 流式响应转换

2. **配置管理** (`config.py`)
   - 环境变量处理
   - 模型映射管理
   - 服务发现

3. **流式处理** (`streaming.py`)
   - SSE事件生成
   - 错误恢复机制
   - 连接管理

4. **监控系统** (`monitoring.py`)
   - 健康检查
   - 性能指标
   - 日志管理

## 📚 开发文档

### 本地开发

```bash
# 使用uv开发模式（推荐）
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8080

# 或传统方式
uvicorn src.main:app --reload --host 0.0.0.0 --port 8080

# 运行测试
uv run pytest tests/  # 或 pytest tests/

# 代码格式化
uv run black src/     # 或 black src/
uv run isort src/     # 或 isort src/

# 类型检查
uv run mypy src/      # 或 mypy src/

# 添加新依赖
uv add package-name

# 移除依赖
uv remove package-name
```

### 项目结构

```
claude-gemini-proxy/
├── src/
│   ├── main.py              # FastAPI应用入口
│   ├── config.py            # 配置管理
│   ├── models.py            # 数据模型
│   ├── converter.py         # API转换器
│   ├── streaming.py         # 流式处理
│   ├── monitoring.py        # 监控系统
│   └── utils.py             # 工具函数
├── tests/                   # 测试文件
├── docs/                    # 文档
├── scripts/                 # 脚本文件
├── pyproject.toml          # uv项目配置
├── requirements.txt         # 传统依赖列表
├── uv.lock                 # uv锁定文件
├── Dockerfile              # Docker配置
├── docker-compose.yml      # Docker Compose
└── README.md               # 项目说明
```

📧 **问题反馈**: [Issues](https://github.com/zhucl1006/claude-gemini-proxy/issues)
💬 **讨论交流**: [Discussions](https://github.com/zhucl1006/claude-gemini-proxy/discussions)
