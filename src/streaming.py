"""
增强版流式响应处理模块
基于server.py的成熟实现，处理Server-Sent Events (SSE) 和流式数据转换
包含完整的错误恢复、重试机制和JSON解析修复
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Dict, Any, Optional, Union
import uuid

from models import ClaudeRequest, ClaudeStreamResponse
from .converter import ClaudeToGeminiConverter
from .config import settings


logger = logging.getLogger(__name__)


class StreamingHandler:
    def __init__(self):
        """初始化流式处理器"""
        self.converter = ClaudeToGeminiConverter()
        self.active_streams = {}  # 跟踪活跃的流
        
        # 基于server.py的常量定义
        self.EVENT_MESSAGE_START = "message_start"
        self.EVENT_MESSAGE_STOP = "message_stop"
        self.EVENT_MESSAGE_DELTA = "message_delta"
        self.EVENT_CONTENT_BLOCK_START = "content_block_start"
        self.EVENT_CONTENT_BLOCK_STOP = "content_block_stop"
        self.EVENT_CONTENT_BLOCK_DELTA = "content_block_delta"
        self.EVENT_PING = "ping"
        
        self.DELTA_TEXT = "text_delta"
        self.DELTA_INPUT_JSON = "input_json_delta"
    
    async def handle_streaming_with_recovery(self, response_generator, request_id: str, original_model: str) -> AsyncGenerator[str, None]:
        """
        基于server.py的增强版流式处理，包含完整的错误恢复机制
        
        Args:
            response_generator: LiteLLM响应生成器
            request_id: 请求ID
            original_model: 原始模型名称
            
        Yields:
            SSE格式的数据块
        """
        message_id = f"msg_{request_id}"
        
        # 发送初始SSE事件
        yield f"event: {self.EVENT_MESSAGE_START}\ndata: {json.dumps({
            'type': self.EVENT_MESSAGE_START,
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {'input_tokens': 0, 'output_tokens': 0}
            }
        })}\n\n"

        yield f"event: {self.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({
            'type': self.EVENT_CONTENT_BLOCK_START,
            'index': 0,
            'content_block': {'type': 'text', 'text': ''}
        })}\n\n"

        yield f"event: {self.EVENT_PING}\ndata: {json.dumps({'type': self.EVENT_PING})}\n\n"

        # 流式状态管理
        accumulated_text = ""
        text_block_index = 0
        tool_block_counter = 0
        current_tool_calls = {}
        input_tokens = 0
        output_tokens = 0
        final_stop_reason = "end_turn"

        # 基于server.py的错误恢复跟踪
        consecutive_errors = 0
        max_consecutive_errors = 10
        stream_terminated_early = False
        malformed_chunks_count = 0
        max_malformed_chunks = 20

        # 缓冲区用于处理不完整的chunk
        chunk_buffer = ""

        try:
            # 包装整个流式处理过程
            stream_iterator = aiter(response_generator)

            while True:
                try:
                    # 获取下一个chunk
                    try:
                        chunk = await asyncio.wait_for(anext(stream_iterator), timeout=90.0)
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        logger.warning("流式响应超时，终止")
                        stream_terminated_early = True
                        break

                    # 重置连续错误计数器
                    consecutive_errors = 0

                    # 处理字符串chunk
                    if isinstance(chunk, str):
                        if chunk.strip() == "[DONE]":
                            break

                        if self._is_malformed_chunk(chunk):
                            malformed_chunks_count += 1
                            logger.debug(f"跳过格式错误的chunk #{malformed_chunks_count}: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}'")

                            if malformed_chunks_count > max_malformed_chunks:
                                logger.error(f"格式错误的chunk过多 ({malformed_chunks_count})，终止流")
                                stream_terminated_early = True
                                break
                            continue

                        # 添加到缓冲区并尝试解析
                        chunk_buffer += chunk
                        parsed_chunk, chunk_buffer = self._try_parse_buffered_chunk(chunk_buffer)

                        if parsed_chunk is None:
                            # 继续缓冲，如果没有完整的chunk
                            if len(chunk_buffer) > 10000:
                                logger.warning("chunk缓冲区过大，清空")
                                chunk_buffer = ""
                            continue

                        chunk = parsed_chunk

                    # 处理字典或对象chunk
                    if isinstance(chunk, dict) or hasattr(chunk, 'choices'):
                        await self._process_chunk(chunk, message_id, text_block_index, 
                                               tool_block_counter, current_tool_calls, 
                                               accumulated_text, output_tokens)

                except (json.JSONDecodeError, ValueError) as parse_error:
                    consecutive_errors += 1
                    logger.debug(f"JSON解析错误 (尝试 {consecutive_errors}/{max_consecutive_errors}): {parse_error}")

                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"连续解析错误过多 ({consecutive_errors})，终止流")
                        stream_terminated_early = True
                        break
                    continue

                except Exception as general_error:
                    consecutive_errors += 1
                    logger.error(f"流式处理意外错误 (尝试 {consecutive_errors}/{max_consecutive_errors}): {general_error}")

                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"连续错误过多 ({consecutive_errors})，终止流")
                        stream_terminated_early = True
                        break

                    await asyncio.sleep(0.1)
                    continue

        except Exception as outer_error:
            logger.error(f"致命流式错误: {outer_error}")
            stream_terminated_early = True

        # 始终发送最终的SSE事件
        await self._send_final_events(message_id, text_block_index, current_tool_calls, 
                                    stream_terminated_early, input_tokens, output_tokens)

    async def _process_chunk(self, chunk, message_id: str, text_block_index: int, 
                           tool_block_counter: int, current_tool_calls: dict, 
                           accumulated_text: str, output_tokens: int):
        """处理单个chunk"""
        delta_content_text = None
        delta_tool_calls = None
        chunk_finish_reason = None

        # 提取chunk数据
        if hasattr(chunk, 'choices') and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, 'delta') and choice.delta:
                delta = choice.delta
                delta_content_text = getattr(delta, 'content', None)
                if hasattr(delta, 'tool_calls'):
                    delta_tool_calls = delta.tool_calls
            chunk_finish_reason = getattr(choice, 'finish_reason', None)
        elif isinstance(chunk, dict):
            choices = chunk.get("choices", [])
            if choices:
                choice = choices[0]
                delta = choice.get("delta", {})
                delta_content_text = delta.get("content")
                delta_tool_calls = delta.get("tool_calls")
                chunk_finish_reason = choice.get("finish_reason")

        # 处理文本增量
        if delta_content_text:
            yield f"event: {self.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({
                'type': self.EVENT_CONTENT_BLOCK_DELTA,
                'index': text_block_index,
                'delta': {
                    'type': self.DELTA_TEXT,
                    'text': delta_content_text
                }
            })}\n\n"

        # 处理工具调用
        if delta_tool_calls:
            for tc_chunk in delta_tool_calls:
                if not (hasattr(tc_chunk, 'function') and tc_chunk.function and
                       hasattr(tc_chunk.function, 'name') and tc_chunk.function.name):
                    continue

                tool_call_id = tc_chunk.id

                if tool_call_id not in current_tool_calls:
                    tool_block_counter += 1
                    tool_index = text_block_index + tool_block_counter

                    current_tool_calls[tool_call_id] = {
                        "index": tool_index,
                        "name": tc_chunk.function.name or "",
                        "args_buffer": tc_chunk.function.arguments or ""
                    }

                    yield f"event: {self.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({
                        'type': self.EVENT_CONTENT_BLOCK_START,
                        'index': tool_index,
                        'content_block': {
                            'type': 'tool_use',
                            'id': tool_call_id,
                            'name': current_tool_calls[tool_call_id]['name'],
                            'input': {}
                        }
                    })}\n\n"

                if tc_chunk.function.arguments:
                    current_tool_calls[tool_call_id]["args_buffer"] += tc_chunk.function.arguments
                    yield f"event: {self.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({
                        'type': self.EVENT_CONTENT_BLOCK_DELTA,
                        'index': current_tool_calls[tool_call_id]['index'],
                        'delta': {
                            'type': self.DELTA_INPUT_JSON,
                            'partial_json': tc_chunk.function.arguments
                        }
                    })}\n\n"

    async def _send_final_events(self, message_id: str, text_block_index: int, 
                               current_tool_calls: dict, stream_terminated_early: bool, 
                               input_tokens: int, output_tokens: int):
        """发送最终的SSE事件"""
        try:
            yield f"event: {self.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({
                'type': self.EVENT_CONTENT_BLOCK_STOP,
                'index': text_block_index
            })}\n\n"

            for tool_data in current_tool_calls.values():
                yield f"event: {self.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({
                    'type': self.EVENT_CONTENT_BLOCK_STOP,
                    'index': tool_data['index']
                })}\n\n"

            if stream_terminated_early:
                final_stop_reason = "error"
            else:
                final_stop_reason = "end_turn"

            usage_data = {"input_tokens": input_tokens, "output_tokens": output_tokens}
            yield f"event: {self.EVENT_MESSAGE_DELTA}\ndata: {json.dumps({
                'type': self.EVENT_MESSAGE_DELTA,
                'delta': {
                    'stop_reason': final_stop_reason,
                    'stop_sequence': None
                },
                'usage': usage_data
            })}\n\n"

            yield f"event: {self.EVENT_MESSAGE_STOP}\ndata: {json.dumps({
                'type': self.EVENT_MESSAGE_STOP
            })}\n\n"

        except Exception as final_error:
            logger.error(f"发送最终SSE事件时出错: {final_error}")

    def _is_malformed_chunk(self, chunk_str: str) -> bool:
        """
        检测格式错误的chunk，基于server.py实现
        
        Args:
            chunk_str: 要检查的chunk字符串
            
        Returns:
            是否为格式错误的chunk
        """
        if not chunk_str or not isinstance(chunk_str, str):
            return True

        chunk_stripped = chunk_str.strip()

        # 空或空白
        if not chunk_stripped:
            return True

        # 单个字符
        malformed_singles = ["{", "}", "[", "]", ",", ":", '"', "'"]
        if chunk_stripped in malformed_singles:
            return True

        # 常见格式错误模式
        malformed_patterns = [
            '{"', '"}', "[{", "}]", "{}", "[]",
            "null", '""', "''", " ", "",
            "{,", ",}", "[,", ",]"
        ]
        if chunk_stripped in malformed_patterns:
            return True

        # 不完整的JSON结构
        if chunk_stripped.startswith('{') and not chunk_stripped.endswith('}'):
            if len(chunk_stripped) < 15:
                return True

        if chunk_stripped.startswith('[') and not chunk_stripped.endswith(']'):
            if len(chunk_stripped) < 10:
                return True

        # 括号不匹配
        if chunk_stripped.count('{') != chunk_stripped.count('}'):
            if len(chunk_stripped) < 20:
                return True

        if chunk_stripped.count('[') != chunk_stripped.count(']'):
            if len(chunk_stripped) < 20:
                return True

        return False

    def _try_parse_buffered_chunk(self, buffer: str) -> tuple[Optional[Dict], str]:
        """
        尝试解析缓冲区中的chunk，基于server.py实现
        
        Args:
            buffer: 缓冲区字符串
            
        Returns:
            (解析后的chunk, 剩余的缓冲区)
        """
        if not buffer.strip():
            return None, ""

        # 尝试查找完整的JSON对象
        brace_count = 0
        start_pos = -1

        for i, char in enumerate(buffer):
            if char == '{':
                if start_pos == -1:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    # 找到完整的JSON对象
                    json_str = buffer[start_pos:i+1]
                    try:
                        parsed = json.loads(json_str)
                        remaining_buffer = buffer[i+1:]
                        return parsed, remaining_buffer
                    except json.JSONDecodeError:
                        continue

        # 没有找到完整的JSON
        return None, buffer
    
    async def _process_streaming_response(self, claude_request: ClaudeRequest, stream_id: str) -> AsyncGenerator[str, None]:
        """
        处理流式响应的核心逻辑
        
        Args:
            claude_request: Claude请求
            stream_id: 流ID
            
        Yields:
            SSE格式的数据块
        """
        try:
            # 发送消息开始事件
            message_id = str(uuid.uuid4())
            start_event = {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": claude_request.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0
                    }
                }
            }
            
            yield f"data: {json.dumps(start_event)}\n\n"
            
            # 发送内容块开始事件
            content_block_start = {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "text",
                    "text": ""
                }
            }
            
            yield f"data: {json.dumps(content_block_start)}\n\n"
            
            # 获取Gemini流式响应
            chunk_count = 0
            total_text = ""
            
            try:
                async for chunk in self.converter.convert_and_send_streaming(claude_request):
                    if chunk.startswith("data: "):
                        try:
                            # 解析chunk数据
                            chunk_data = json.loads(chunk[6:])
                            
                            # 处理错误
                            if chunk_data.get("type") == "error":
                                yield chunk
                                return
                            
                            # 处理文本增量
                            if chunk_data.get("type") == "content_block_delta" and chunk_data.get("delta", {}).get("text"):
                                text_delta = chunk_data["delta"]["text"]
                                total_text += text_delta
                                chunk_count += 1
                                
                                # 发送文本增量事件
                                delta_event = {
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": text_delta
                                    }
                                }
                                
                                yield f"data: {json.dumps(delta_event)}\n\n"
                                
                                # 添加小延迟以防止过载
                                if chunk_count % 10 == 0:
                                    await asyncio.sleep(0.001)
                                    
                        except json.JSONDecodeError:
                            logger.warning(f"无法解析chunk数据: {chunk}")
                            continue
                        except Exception as e:
                            logger.error(f"处理chunk时出错: {e}")
                            # 发送错误响应
                            error_event = {
                                "type": "error",
                                "error": {
                                    "type": "stream_error",
                                    "message": str(e)
                                }
                            }
                            yield f"data: {json.dumps(error_event)}\n\n"
                            return
            except Exception as e:
                logger.error(f"流式响应错误: {e}")
                # 发送错误响应
                error_event = {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": str(e)
                    }
                }
                yield f"data: {json.dumps(error_event)}\n\n"
            
            # 发送内容块停止事件
            content_block_stop = {
                "type": "content_block_stop",
                "index": 0
            }
            
            yield f"data: {json.dumps(content_block_stop)}\n\n"
            
            # 发送消息增量事件（包含使用情况）
            usage_event = {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "end_turn",
                    "stop_sequence": None
                },
                "usage": {
                    "output_tokens": len(total_text.split())
                }
            }
            
            yield f"data: {json.dumps(usage_event)}\n\n"
            
            # 发送消息停止事件
            message_stop = {
                "type": "message_stop"
            }
            
            yield f"data: {json.dumps(message_stop)}\n\n"
            
        except Exception as e:
            logger.error(f"处理流式响应时出错: {e}")
            raise
    
    def _create_error_chunk(self, error_type: str, message: str) -> str:
        """
        创建错误响应块
        
        Args:
            error_type: 错误类型
            message: 错误消息
            
        Returns:
            SSE格式的错误块
        """
        error_event = {
            "type": "error",
            "error": {
                "type": error_type,
                "message": message
            }
        }
        
        return f"data: {json.dumps(error_event)}\n\n"
    
    def _create_heartbeat_chunk(self) -> str:
        """
        创建心跳包
        
        Returns:
            SSE格式的心跳包
        """
        return f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
    
    def get_active_streams(self) -> Dict[str, Any]:
        """
        获取当前活跃的流信息
        
        Returns:
            活跃流的状态信息
        """
        return {
            stream_id: {
                "status": info["status"],
                "duration": time.time() - info["start_time"],
                "model": info["request"].model
            }
            for stream_id, info in self.active_streams.items()
        }
    
    async def cleanup_expired_streams(self, max_age: int = 300):
        """
        清理过期的流
        
        Args:
            max_age: 最大存活时间（秒）
        """
        current_time = time.time()
        expired_streams = []
        
        for stream_id, info in self.active_streams.items():
            if current_time - info["start_time"] > max_age:
                expired_streams.append(stream_id)
        
        for stream_id in expired_streams:
            self.active_streams.pop(stream_id, None)
            logger.info(f"清理过期流: {stream_id}")


class StreamRetryHandler:
    """流式重试处理器"""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        """初始化重试处理器"""
        self.max_retries = max_retries
        self.delay = delay
    
    async def retry_with_backoff(self, func, *args, **kwargs):
        """
        带有指数退避的重试机制
        
        Args:
            func: 要重试的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数结果
        """
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries:
                    logger.error(f"重试{self.max_retries}次后仍然失败: {e}")
                    raise
                
                wait_time = self.delay * (2 ** attempt)
                logger.warning(f"第{attempt + 1}次尝试失败，等待{wait_time}秒后重试: {e}")
                await asyncio.sleep(wait_time)


class StreamBuffer:
    """流式缓冲区管理"""
    
    def __init__(self, max_size: int = 1000):
        """初始化缓冲区"""
        self.buffer = []
        self.max_size = max_size
        self.lock = asyncio.Lock()
    
    async def add_chunk(self, chunk: str):
        """添加数据块到缓冲区"""
        async with self.lock:
            if len(self.buffer) >= self.max_size:
                self.buffer.pop(0)  # 移除最旧的数据
            self.buffer.append(chunk)
    
    async def get_chunks(self) -> list:
        """获取所有缓冲的数据块"""
        async with self.lock:
            chunks = self.buffer.copy()
            self.buffer.clear()
            return chunks
    
    async def size(self) -> int:
        """获取当前缓冲区大小"""
        async with self.lock:
            return len(self.buffer)


# 全局流式处理器实例
streaming_handler = StreamingHandler()


async def periodic_cleanup():
    """定期清理过期的流"""
    while True:
        await asyncio.sleep(60)  # 每分钟检查一次
        try:
            await streaming_handler.cleanup_expired_streams()
        except Exception as e:
            logger.error(f"清理过期流时出错: {e}")