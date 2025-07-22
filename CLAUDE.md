# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Claude Code Gemini Proxy** - A Python-based proxy service that enables Claude Code to use Google Gemini and other AI models through a unified API interface.

## Quick Start Commands

### Development Setup
```bash
# Install dependencies with uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or with pip
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="your-gemini-api-key"

# Run development server
uv run python src/main.py
# Or with hot reload
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8080
```

### Testing & Quality
```bash
# Run tests
uv run pytest tests/

# Code formatting
uv run black src/
uv run isort src/

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/
```

### Configuration
```bash
# Copy and edit environment file
cp .env.example .env

# Validate configuration
uv run python -c "from src.config import validate_config; validate_config()"

# Test Gemini connection
curl http://localhost:8080/test-connection
```

## Architecture Overview

### Core Components
- **main.py**: FastAPI application entry point with endpoint definitions
- **config.py**: Environment-based configuration management using Pydantic
- **models.py**: Pydantic models for Claude API and Gemini API data structures
- **converter.py**: API format conversion between Claude and Gemini formats
- **streaming.py**: Server-Sent Events (SSE) handling for streaming responses
- **utils.py**: Utility functions for error handling, validation, and monitoring

### Key Features
- **Full Claude API Compatibility**: Compatible with Anthropic Messages API
- **Streaming Support**: Real-time SSE streaming responses
- **Error Recovery**: Intelligent retry mechanisms and error handling
- **Multi-Modal**: Support for text and image inputs
- **Tool Calling**: Function calling support
- **Health Monitoring**: Built-in health checks and diagnostics

### API Endpoints
- `POST /v1/messages` - Claude-compatible messages endpoint
- `GET /health` - Health check endpoint
- `GET /test-connection` - Gemini API connectivity test
- `GET /docs` - FastAPI auto-generated documentation

## Environment Variables

### Required
```bash
GEMINI_API_KEY=your-google-ai-studio-api-key
```

### Optional Configuration
```bash
# Server settings
PROXY_HOST=0.0.0.0
PROXY_PORT=8080

# Model settings
DEFAULT_MODEL=gemini-1.5-pro
ENABLE_STREAMING=true

# Request limits
MAX_TOKENS=4000
TEMPERATURE=0.7
TOP_P=0.8
TOP_K=40

# Retry settings
MAX_RETRIES=3
RETRY_DELAY=1.0

# Logging
LOG_LEVEL=INFO
```

## Claude Code Integration

### Quick Setup
```bash
# 1. Start the proxy
uv run python src/main.py

# 2. Configure Claude Code
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_API_KEY=dummy-key

# 3. Test connection
claude "Hello, are you using Gemini?"
```

### Alternative Configuration
```bash
# Using Claude config
claude config set api_url http://localhost:8080
claude config set api_key dummy-key
```

## Project Structure

```
src/
├── main.py           # FastAPI application
├── config.py         # Configuration management
├── models.py         # Data models
├── converter.py      # API format conversion
├── streaming.py      # SSE streaming
├── utils.py          # Utilities and helpers
└── __init__.py

tests/                # Test suite
docs/                 # Documentation
scripts/              # Utility scripts
monitoring/           # Health monitoring
```

## Model Support

### Supported Gemini Models
- gemini-1.5-pro (default)
- gemini-1.5-flash
- gemini-1.0-pro
- gemini-pro
- gemini-pro-vision

### Configuration Examples
```bash
# Use Gemini Flash for faster responses
export DEFAULT_MODEL=gemini-1.5-flash

# Set custom parameters
export MAX_TOKENS=8192
export TEMPERATURE=0.5
```

## Development Workflow

### Adding New Features
1. Update models in `src/models.py` if needed
2. Implement conversion logic in `src/converter.py`
3. Add streaming support in `src/streaming.py`
4. Update configuration in `src/config.py`
5. Add tests in `tests/`
6. Update documentation

### Testing Changes
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_models.py

# Run with coverage
uv run pytest --cov=src

# Test manually
curl -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-1.5-pro","messages":[{"role":"user","content":"Hello"}]}'
```

## Common Issues and Solutions

### API Key Issues
- **Error**: `invalid_api_key` - Check GEMINI_API_KEY is set correctly
- **Solution**: Visit Google AI Studio to generate a new API key

### Connection Issues
- **Error**: `network_error` - Check network connectivity
- **Solution**: Test with `curl http://localhost:8080/health`

### Rate Limiting
- **Error**: `rate_limit` - Too many requests
- **Solution**: Wait and retry, or upgrade API tier

### Memory Issues
- **High memory usage**: Reduce MAX_TOKENS or use streaming
- **Solution**: Monitor with logs and adjust parameters

## Docker Usage

```bash
# Build image
docker build -t claude-gemini-proxy .

# Run container
docker run -d \
  -p 8080:8080 \
  -e GEMINI_API_KEY=your-key \
  claude-gemini-proxy

# Or use Docker Compose
docker-compose up -d
```

## Production Deployment

### Environment Variables for Production
```bash
PROXY_HOST=0.0.0.0
PROXY_PORT=8080
LOG_LEVEL=WARNING
API_RATE_LIMIT=100/minute
ALLOWED_ORIGINS=["https://your-domain.com"]
```

### Health Monitoring
```bash
# Check health
curl http://localhost:8080/health

# Monitor logs
tail -f claude-gemini-proxy.log
```

## Extension Points

### Adding New AI Providers
1. Extend `src/converter.py` with new provider conversion
2. Add new models to `src/config.py`
3. Update validation in `src/utils.py`
4. Add tests for new provider

### Custom Middleware
- Add authentication middleware in `src/main.py`
- Implement custom logging in `src/utils.py`
- Add rate limiting as needed

## Performance Optimization

### Streaming Best Practices
- Use streaming for large responses
- Set appropriate timeout values
- Monitor memory usage
- Implement proper cleanup

### Resource Management
- Configure MAX_TOKENS based on use case
- Use connection pooling if needed
- Monitor error rates and adjust retry settings