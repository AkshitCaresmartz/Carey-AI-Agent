"""GROQ LLM client with MCP integration."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from groq import Groq
from pydantic import BaseModel

from .mcp_client import MCPClient
from .config import get_settings

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str  # "user", "assistant"
    content: str


class ConversationManager:
    """Manages conversation state and message history."""

    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.conversation_id: Optional[str] = None

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append({"role": "user", "content": content})
        logger.debug(f"ConversationManager: added user message (len={len(content)} chars)")

    def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation."""
        # System messages should come before user messages; prepend if empty
        # or insert at the beginning to ensure they are seen by the model.
        # We'll insert at index 0 to keep a single authoritative system message.
        # If multiple system messages are added, they will appear in order.
        self.messages.insert(0, {"role": "system", "content": content})
        logger.debug("ConversationManager: added system message")

    def add_assistant_message(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add an assistant message to the conversation."""
        message = {"role": "assistant", "content": content}
        if tool_calls:
            # sanitize tool_calls to ensure each has a function object with 'name' and 'arguments' (string)
            sanitized = []
            for tc in tool_calls:
                try:
                    tc_copy = dict(tc)
                    func = tc_copy.get('function') or {}

                    # find arguments either at top-level or inside function
                    raw_args = None
                    if 'arguments' in tc_copy and tc_copy['arguments'] is not None:
                        raw_args = tc_copy['arguments']
                    elif isinstance(func, dict) and 'arguments' in func and func['arguments'] is not None:
                        raw_args = func['arguments']

                    # stringify arguments
                    if raw_args is None:
                        args_str = '{}'
                    elif isinstance(raw_args, str):
                        args_str = raw_args
                    else:
                        try:
                            args_str = json.dumps(raw_args)
                        except Exception:
                            args_str = str(raw_args)

                    sanitized.append({
                        'id': tc_copy.get('id'),
                        'type': 'function',
                        'function': {
                            'name': func.get('name') if isinstance(func, dict) else func,
                            'arguments': args_str
                        }
                    })
                except Exception:
                    # fallback: keep original
                    sanitized.append(tc)

            message["tool_calls"] = sanitized
            logger.debug(f"ConversationManager: added assistant message with {len(sanitized)} tool_calls")
        self.messages.append(message)
        logger.debug("ConversationManager: appended assistant message")

    def add_tool_message(self, name: str, content: str) -> None:
        """Add a tool execution result to the conversation.

        Use the 'name' key so LLMs/GROQ can associate the tool output with the
        function name (OpenAI/GROQ expect tool messages like {role: 'tool', name: 'fn', content: '...'}).
        """
        self.messages.append({
            "role": "tool",
            "name": name,
            "content": content
        })
        logger.debug(f"ConversationManager: added tool message for '{name}' (len={len(str(content))} chars)")

    def add_tool_message_with_id(self, name: str, content: str, tool_call_id: Optional[str] = None) -> None:
        """Add a tool execution result to the conversation and include the
        `tool_call_id` field when available so GROQ/OpenAI can associate the
        tool message with the corresponding assistant tool call.
        """
        msg = {
            "role": "tool",
            "name": name,
            "content": content
        }
        if tool_call_id is not None:
            msg["tool_call_id"] = tool_call_id
        self.messages.append(msg)
        logger.debug(f"ConversationManager: added tool message for '{name}' with tool_call_id='{tool_call_id}'")

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in the conversation."""
        msgs = self.messages.copy()
        logger.debug(f"ConversationManager: get_messages called (count={len(msgs)})")
        return msgs

    def clear_messages(self) -> None:
        """Clear all messages from the conversation."""
        count = len(self.messages)
        self.messages.clear()
        logger.info(f"ConversationManager: cleared messages (removed {count} messages)")


class GroqClient:
    """GROQ LLM client with optional MCP tools integration."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        mcp_client: Optional[MCPClient] = None
    ):
        """Initialize GROQ client.
        
        Args:
            api_key: GROQ API key
            model: GROQ model to use
            mcp_client: Optional MCP client for tool integration
        """
        # Load settings when values are not explicitly provided
        settings = get_settings()

        if api_key is None:
            api_key = settings.groq_api_key
            if api_key:
                logger.info("GROQ API key loaded from environment settings")
            else:
                logger.warning("No GROQ API key provided; GROQ client may fail to authenticate")

        if model is None:
            model = settings.groq_model
            logger.info(f"Using GROQ model from settings: {model}")
        else:
            logger.info(f"Using GROQ model provided to GroqClient: {model}")

        self.client = Groq(api_key=api_key)
        self.model = model
        self.mcp_client = mcp_client
        self.tools: List[Dict[str, Any]] = []
        logger.info(f"GroqClient initialized with model={self.model} mcp_client_present={bool(self.mcp_client)} tools_count={len(self.tools)}")

    async def initialize_tools(self) -> None:
        """Initialize MCP tools if MCP client is available."""
        if not self.mcp_client:
            logger.info("No MCP client provided, running in simple LLM mode")
            return

        try:
            mcp_tools = await self.mcp_client.list_tools()
            
            # Convert MCP tools to a function-calling format expected by the
            # LLM client. Ensure the `parameters` field is a JSON Schema dict.
            for tool in mcp_tools:
                # Normalize inputSchema into a dict (JSON Schema). Some MCP
                # servers may return a stringified JSON schema or None.
                parameters = None
                try:
                    if tool.inputSchema is None:
                        parameters = {"type": "object", "properties": {}, "additionalProperties": True}
                    elif isinstance(tool.inputSchema, str):
                        parameters = json.loads(tool.inputSchema)
                    elif isinstance(tool.inputSchema, dict):
                        parameters = tool.inputSchema
                    else:
                        # Fallback to permissive object schema
                        parameters = {"type": "object", "properties": {}, "additionalProperties": True}
                except Exception as e:
                    logger.warning(f"Failed to parse inputSchema for tool {tool.name}: {e}; using permissive schema")
                    parameters = {"type": "object", "properties": {}, "additionalProperties": True}

                function_def = {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": parameters
                }

                logger.debug(f"Prepared function def for tool '{tool.name}': {json.dumps(function_def, default=str)[:1000]}")

                # Wrap into the expected GROQ/OpenAI function format
                wrapped = {
                    "type": "function",
                    "function": function_def
                }

                self.tools.append(wrapped)

            # Extract names for logging (wrapped format)
            tool_names = []
            for t in self.tools:
                try:
                    if isinstance(t, dict) and 'function' in t:
                        tool_names.append(t['function'].get('name'))
                    else:
                        tool_names.append(t.get('name'))
                except Exception:
                    tool_names.append(str(t))

            logger.info(f"Initialized {len(self.tools)} MCP tools: {tool_names}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP tools: {e}")
            raise

    async def chat_completion(
        self,
        conversation_manager: ConversationManager,
        message: str
    ) -> Tuple[str, bool, Optional[List[Dict[str, Any]]]]:
        """Generate chat completion with optional tool usage.
        
        Args:
            conversation_manager: Conversation state manager
            message: User message
            
        Returns:
            Tuple of (response_content, used_tools, tool_calls_info)
        """
        logger.info(f"GroqClient.chat_completion called for message (len={len(message)}): {message[:80]}...")
        conversation_manager.add_user_message(message)
        
        # If no tools available, use simple completion
        if not self.tools:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversation_manager.get_messages(),
                temperature=0.7,
                max_tokens=1000
            )
            response_content = response.choices[0].message.content or "I don't have a response to that."
            # If the model returned an inline function-like string, try to execute it
            # Pattern: '<function=NAME>{JSON_ARGS}' or '<function=NAME>JSON' or '<function=NAME>{...}'
            try:
                rt = str(response_content)
                if '<function=' in rt and self.mcp_client:
                    idx = rt.find('<function=')
                    start = idx + len('<function=')
                    end_name = rt.find('>', start)
                    brace_idx = rt.find('{', start)

                    if end_name == -1:
                        if brace_idx != -1:
                            func_name = rt[start:brace_idx]
                            args_start = brace_idx
                        else:
                            # no args, name until whitespace
                            ws = rt.find(' ', start)
                            if ws == -1:
                                func_name = rt[start:]
                                args_start = -1
                            else:
                                func_name = rt[start:ws]
                                args_start = -1
                    else:
                        func_name = rt[start:end_name]
                        args_start = rt.find('{', end_name)

                    args_part = '{}'
                    if args_start != -1 and args_start is not None:
                        depth = 0
                        for i in range(args_start, len(rt)):
                            if rt[i] == '{':
                                depth += 1
                            elif rt[i] == '}':
                                depth -= 1
                                if depth == 0:
                                    args_part = rt[args_start:i+1]
                                    break

                    try:
                        func_args = json.loads(args_part)
                    except Exception:
                        func_args = {}

                    logger.info(f"Detected inline function (no-tools mode): {func_name} args: {func_args}")
                    # execute tool
                    result = await self.mcp_client.call_tool(func_name, func_args)
                    # extract content
                    tool_result = None
                    try:
                        if getattr(result, 'structuredContent', None) is not None:
                            tool_result = result.structuredContent
                        elif hasattr(result, 'content') and result.content:
                            c = result.content[0]
                            tool_result = c.text if getattr(c, 'text', None) else c
                        else:
                            tool_result = result
                    except Exception:
                        tool_result = str(result)

                    # add tool message and synthesize; include tool_call_id to match assistant tool call
                    conversation_manager.add_tool_message_with_id(func_name, str(tool_result), tool_call_id=func_name)
                    outgoing = []
                    for m in conversation_manager.get_messages():
                        msg = dict(m)
                        if msg.get('role') == 'tool' and isinstance(msg.get('content'), (list, tuple)):
                            try:
                                parts = []
                                for c in msg['content']:
                                    if isinstance(c, dict) and c.get('text'):
                                        parts.append(c.get('text'))
                                    else:
                                        parts.append(str(c))
                                msg['content'] = '\n'.join(parts)
                            except Exception:
                                msg['content'] = str(msg['content'])
                        outgoing.append(msg)

                    final_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=outgoing,
                        temperature=0.7,
                        max_tokens=1000
                    )
                    final_content = final_response.choices[0].message.content or "I processed your request."
                    conversation_manager.add_assistant_message(final_content)
                    return final_content, True, [{"function": func_name, "arguments": func_args, "result": tool_result}]
            except Exception:
                logger.debug('Inline function detection/execution failed in no-tools path', exc_info=True)

            conversation_manager.add_assistant_message(response_content)
            return response_content, False, None

        # Use tools if available
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversation_manager.get_messages(),
                tools=self.tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1000
            )
            # Log the full raw model response for debugging (truncated)
            try:
                logger.debug(f"Raw model response: {str(response)[:4000]}")
            except Exception:
                logger.debug("Raw model response could not be serialized")

            response_message = response.choices[0].message
            tool_calls_info = []
            used_tools = False

            # Handle tool calls if present
            if getattr(response_message, 'tool_calls', None):
                logger.info(f"Model requested tool calls: {[tc.function.name for tc in response_message.tool_calls]}")
                used_tools = True
                # Log the tool call intents
                tool_call_summary = []
                # Ensure each tool_call has an id; if missing, generate one so we
                # can attach the same id to the subsequent tool message
                for idx, tc in enumerate(response_message.tool_calls):
                    try:
                        # tc.function.arguments may be a JSON string or an object
                        raw_args = None
                        try:
                            raw_args = tc.function.arguments
                        except Exception:
                            raw_args = None

                        # Ensure arguments are a JSON string when sending back to GROQ
                        if raw_args is None:
                            args_str = "{}"
                        elif isinstance(raw_args, str):
                            args_str = raw_args
                        else:
                            try:
                                args_str = json.dumps(raw_args)
                            except Exception:
                                args_str = str(raw_args)

                    except Exception:
                        args_str = "{}"

                    # function must be an object (not a string) and must include 'arguments' inside
                    tc_id = None
                    try:
                        tc_id = tc.id
                    except Exception:
                        tc_id = None
                    if not tc_id:
                        # generate a predictable id when model does not supply one
                        tc_id = f"{tc.function.name}_{idx}"

                    tool_call_summary.append({
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": args_str}
                    })

                logger.info(f"Model requested tool calls: {tool_call_summary}")
                conversation_manager.add_assistant_message(
                    response_message.content or "",
                    tool_call_summary
                )

                # Execute each tool call, using generated ids from tool_call_summary
                for i, tool_call in enumerate(response_message.tool_calls):
                    # corresponding id generated above
                    try:
                        tc_id = tool_call_summary[i].get('id')
                    except Exception:
                        tc_id = None
                    try:
                        function_name = tool_call.function.name
                        # tool_call.function.arguments may be a JSON string or dict
                        raw_args = None
                        try:
                            raw_args = tool_call.function.arguments
                        except Exception:
                            raw_args = None

                        if raw_args is None:
                            function_args = {}
                        elif isinstance(raw_args, str):
                            try:
                                function_args = json.loads(raw_args)
                            except Exception:
                                # Fallback: keep as raw string
                                function_args = raw_args
                        else:
                            function_args = raw_args

                        logger.info(f"Calling MCP tool: {function_name} with args: {function_args}")
                        
                        # Call the MCP tool
                        result = await self.mcp_client.call_tool(function_name, function_args)
                        logger.debug(f"Raw result object for tool '{function_name}': {str(result)[:2000]}")
                        
                        # Extract content from the result
                        # Prefer structuredContent if available, then content
                        tool_result = None
                        try:
                            if getattr(result, 'structuredContent', None) is not None:
                                tool_result = result.structuredContent
                            elif hasattr(result, 'content') and result.content:
                                # Some MCP servers return content as list with text
                                c = result.content[0]
                                tool_result = c.text if getattr(c, 'text', None) else c
                            else:
                                tool_result = result
                        except Exception:
                            tool_result = str(result)
                        
                        # Prepare a human-friendly automatic summary where possible
                        def _auto_summarize(result_obj: Any) -> Tuple[str, Any]:
                            # returns (summary_text, raw_for_response)
                            try:
                                parsed = result_obj
                                if isinstance(result_obj, str):
                                    try:
                                        parsed = json.loads(result_obj)
                                    except Exception:
                                        parsed = result_obj

                                # If it's a list, count items
                                if isinstance(parsed, list):
                                    count = len(parsed)
                                    sample = parsed[0] if count > 0 else None
                                    if count == 0:
                                        return ("No results found.", parsed)
                                    # build sample summary fields
                                    if isinstance(sample, dict):
                                        keys = list(sample.keys())[:5]
                                        sample_str = ', '.join([f"{k}: {sample.get(k)}" for k in keys])
                                        return (f"Found {count} result(s). Example: {sample_str}", parsed)
                                    else:
                                        return (f"Found {count} result(s). Example: {str(sample)[:200]}", parsed)

                                # If it's a dict, try to summarize some common fields
                                if isinstance(parsed, dict):
                                    # If a TotalRecords-like field exists, use it
                                    total = parsed.get('TotalRecords') or parsed.get('Total') or parsed.get('Count')
                                    if total:
                                        return (f"Found {total} result(s).", parsed)
                                    # Otherwise, present a short selection of keys
                                    keys = list(parsed.keys())[:6]
                                    brief = ', '.join([f"{k}: {parsed.get(k)}" for k in keys])
                                    return (f"Found 1 result. {brief}", parsed)

                                # fallback: short text summary
                                return (str(parsed)[:1000], parsed)
                            except Exception:
                                return (str(result_obj), result_obj)

                        # Prepare the tool output as a tool message (stringified).
                        # Ensure raw_for_response is defined (attempt to parse JSON if the
                        # tool returned a JSON string).
                        raw_for_response = tool_result
                        try:
                            if isinstance(tool_result, str):
                                try:
                                    parsed = json.loads(tool_result)
                                    raw_for_response = parsed
                                except Exception:
                                    raw_for_response = tool_result
                        except Exception:
                            raw_for_response = tool_result

                        try:
                            raw_text = json.dumps(raw_for_response, default=str, indent=2) if not isinstance(raw_for_response, str) else raw_for_response
                        except Exception:
                            raw_text = str(raw_for_response)

                        # Add the tool message (with id) so the model can read the tool output
                        conversation_manager.add_tool_message_with_id(function_name, raw_text, tool_call_id=tc_id)

                        tool_calls_info.append({
                            "function": function_name,
                            "arguments": function_args,
                            "result": tool_result
                        })
                        # DO NOT return here; let the final LLM synthesis step below
                        # read the tool output and generate a natural-language reply.
                        
                    except Exception as e:
                        error_msg = f"Tool execution failed: {str(e)}"
                        logger.error(f"Tool {function_name} failed: {e}", exc_info=True)
                        conversation_manager.add_tool_message_with_id(function_name, error_msg, tool_call_id=tc_id)
                        
                        tool_calls_info.append({
                            "function": function_name,
                            "arguments": function_args,
                            "result": error_msg,
                            "error": True
                        })

                # Generate final response after tool execution
                # Before sending final request to GROQ, normalize messages to ensure
                # 'assistant' messages with tool_calls include function.arguments as a string
                outgoing_messages = []
                for m in conversation_manager.get_messages():
                    # shallow copy
                    msg = dict(m)
                    if msg.get('role') == 'assistant' and 'tool_calls' in msg:
                        normalized_tool_calls = []
                        for tc in msg['tool_calls']:
                            # tc may have 'arguments' at top-level or inside function
                            tc_copy = dict(tc)
                            func = tc_copy.get('function') or {}
                            # determine raw arguments
                            raw_args = None
                            if 'arguments' in tc_copy and tc_copy['arguments'] is not None:
                                raw_args = tc_copy['arguments']
                            elif 'arguments' in func and func['arguments'] is not None:
                                raw_args = func['arguments']

                            # stringify if needed
                            if raw_args is None:
                                args_str = '{}'
                            elif isinstance(raw_args, str):
                                args_str = raw_args
                            else:
                                try:
                                    args_str = json.dumps(raw_args)
                                except Exception:
                                    args_str = str(raw_args)

                            normalized_tool_calls.append({
                                'id': tc_copy.get('id'),
                                'type': 'function',
                                'function': {
                                    'name': func.get('name') if isinstance(func, dict) else func,
                                    'arguments': args_str
                                }
                            })

                        msg['tool_calls'] = normalized_tool_calls

                    # Normalize tool role content: if content is a list-like structured
                    # convert to text where Groq expects text. We keep as-is if string.
                    if msg.get('role') == 'tool' and isinstance(msg.get('content'), (list, tuple)):
                        try:
                            # join text fields if possible
                            parts = []
                            for c in msg['content']:
                                if isinstance(c, dict) and c.get('text'):
                                    parts.append(c.get('text'))
                                else:
                                    parts.append(str(c))
                            msg['content'] = '\n'.join(parts)
                        except Exception:
                            msg['content'] = str(msg['content'])

                    # Append every normalized message so the final synthesis has full context
                    outgoing_messages.append(msg)

                    # Ensure tool messages include tool_call_id matching assistant tool_calls
                    try:
                        # build a mapping from function name -> last seen tool_call id
                        name_to_id = {}
                        for o in outgoing_messages:
                            if o.get('role') == 'assistant' and 'tool_calls' in o:
                                for tc in o['tool_calls']:
                                    try:
                                        fn = tc.get('function', {}).get('name') if isinstance(tc.get('function'), dict) else tc.get('function')
                                        tid = tc.get('id')
                                        if fn and tid:
                                            name_to_id[fn] = tid
                                    except Exception:
                                        continue

                        for o in outgoing_messages:
                            if o.get('role') == 'tool' and 'tool_call_id' not in o:
                                try:
                                    nm = o.get('name')
                                    if nm and nm in name_to_id:
                                        o['tool_call_id'] = name_to_id[nm]
                                    elif nm:
                                        # fallback: use the tool name as id
                                        o['tool_call_id'] = nm
                                except Exception:
                                    continue
                    except Exception:
                        pass

                # Insert a one-off system instruction at the start of the outgoing
                # messages to ask the model to synthesize a human-readable answer
                synth_prompt = {
                    'role': 'system',
                    'content': (
                        'You are an assistant that converts tool outputs into a concise, human-readable '
                        'summary for the user. Use the tool output(s) present in the conversation and do not '
                        'repeat function call syntax or tool names. Answer directly and clearly.'
                    )
                }
                final_messages = [synth_prompt] + outgoing_messages
                try:
                    import json as _json
                    logger.debug("Outgoing messages for final synthesis: %s", _json.dumps(final_messages, default=str)[:4000])
                except Exception:
                    logger.debug("Outgoing messages for final synthesis (truncated)")

                # Allow the model to call tools during final synthesis. If the model
                # calls tools, execute them and re-run synthesis until the model
                # returns a non-tool response.
                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=final_messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=0.2,
                    max_tokens=1000
                )

                # Loop to handle any tool_calls the model requests during synthesis.
                while getattr(final_response.choices[0].message, 'tool_calls', None):
                    synth_tool_calls = final_response.choices[0].message.tool_calls
                    logger.info(f"Synthesis pass requested tool calls: {[tc.function.name for tc in synth_tool_calls]}")

                    # Execute each requested tool and append tool messages
                    for stc in synth_tool_calls:
                        try:
                            # ensure id
                            stc_id = getattr(stc, 'id', None) or f"{stc.function.name}_synth"
                            # parse args (may be string or object)
                            raw_args = None
                            try:
                                raw_args = stc.function.arguments
                            except Exception:
                                raw_args = None
                            if raw_args is None:
                                stc_args = {}
                            elif isinstance(raw_args, str):
                                try:
                                    stc_args = json.loads(raw_args)
                                except Exception:
                                    stc_args = raw_args
                            else:
                                stc_args = raw_args

                            logger.info(f"Calling synthesis-time tool: {stc.function.name} with args: {stc_args}")
                            res = await self.mcp_client.call_tool(stc.function.name, stc_args)
                            # extract tool result
                            try:
                                if getattr(res, 'structuredContent', None) is not None:
                                    stc_result = res.structuredContent
                                elif hasattr(res, 'content') and res.content:
                                    c = res.content[0]
                                    stc_result = c.text if getattr(c, 'text', None) else c
                                else:
                                    stc_result = res
                            except Exception:
                                stc_result = str(res)

                            # add assistant tool_call marker and tool message
                            assistant_tool_call = [{
                                'id': stc_id,
                                'type': 'function',
                                'function': {'name': stc.function.name, 'arguments': json.dumps(stc_args)}
                            }]
                            conversation_manager.add_assistant_message('', assistant_tool_call)
                            # stringify stc_result
                            try:
                                stc_raw = json.dumps(stc_result, default=str, indent=2) if not isinstance(stc_result, str) else stc_result
                            except Exception:
                                stc_raw = str(stc_result)
                            conversation_manager.add_tool_message_with_id(stc.function.name, stc_raw, tool_call_id=stc_id)

                            tool_calls_info.append({
                                'function': stc.function.name,
                                'arguments': stc_args,
                                'result': stc_result
                            })
                        except Exception as e:
                            logger.error(f"Synthesis-time tool {getattr(stc, 'function', {}).get('name', 'unknown')} failed: {e}", exc_info=True)
                            # attach error message as tool output
                            conversation_manager.add_tool_message_with_id(getattr(stc, 'function', {}).get('name', 'tool'), f"Tool execution error: {e}", tool_call_id=getattr(stc, 'id', None))

                    # Rebuild outgoing messages for next synthesis pass
                    outgoing_messages = []
                    for m in conversation_manager.get_messages():
                        msg = dict(m)
                        if msg.get('role') == 'assistant' and 'tool_calls' in msg:
                            normalized_tool_calls = []
                            for tc in msg['tool_calls']:
                                tc_copy = dict(tc)
                                func = tc_copy.get('function') or {}
                                raw_args = None
                                if 'arguments' in tc_copy and tc_copy['arguments'] is not None:
                                    raw_args = tc_copy['arguments']
                                elif 'arguments' in func and func['arguments'] is not None:
                                    raw_args = func['arguments']
                                if raw_args is None:
                                    args_str = '{}'
                                elif isinstance(raw_args, str):
                                    args_str = raw_args
                                else:
                                    try:
                                        args_str = json.dumps(raw_args)
                                    except Exception:
                                        args_str = str(raw_args)
                                normalized_tool_calls.append({
                                    'id': tc_copy.get('id'),
                                    'type': 'function',
                                    'function': {
                                        'name': func.get('name') if isinstance(func, dict) else func,
                                        'arguments': args_str
                                    }
                                })
                            msg['tool_calls'] = normalized_tool_calls

                        if msg.get('role') == 'tool' and isinstance(msg.get('content'), (list, tuple)):
                            try:
                                parts = []
                                for c in msg['content']:
                                    if isinstance(c, dict) and c.get('text'):
                                        parts.append(c.get('text'))
                                    else:
                                        parts.append(str(c))
                                msg['content'] = '\n'.join(parts)
                            except Exception:
                                msg['content'] = str(msg['content'])

                        outgoing_messages.append(msg)

                    final_messages = [synth_prompt] + outgoing_messages
                    try:
                        logger.debug("Outgoing messages for final synthesis (loop): %s", json.dumps(final_messages, default=str)[:4000])
                    except Exception:
                        logger.debug("Outgoing messages for final synthesis (loop) could not be serialized")

                    final_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=final_messages,
                        tools=self.tools,
                        tool_choice="auto",
                        temperature=0.2,
                        max_tokens=1000
                    )

                # When loop exits, final_response contains natural language reply
                final_content = final_response.choices[0].message.content or "I processed your request."
                conversation_manager.add_assistant_message(final_content)
                return final_content, used_tools, tool_calls_info

            else:
                # No tool calls, log the model response to help debugging why
                # the model didn't choose to call tools.
                response_content = response_message.content or "I don't have a response to that."
                logger.info(f"Model responded without tool calls: {response_content}")

                # Try to detect an inline function intent of the form
                # '<function=NAME>{JSON_ARGS}' anywhere in the returned text.
                detected = False
                try:
                    rt = str(response_content)
                    if '<function=' in rt:
                        detected = True
                        idx = rt.find('<function=')
                        start = idx + len('<function=')
                        # Look for '>' after the name. If not present, the model may emit
                        # '<function=name>{...}' (no '>'). In that case find the '{' and
                        # consider the name to stop just before the brace.
                        end_name = rt.find('>', start)
                        brace_idx_after_name = rt.find('{', start)

                        if end_name == -1:
                            # No '>' found. If there's a '{' right after the name, use that as delimiter.
                            if brace_idx_after_name != -1:
                                func_name = rt[start:brace_idx_after_name]
                                args_start = brace_idx_after_name
                            else:
                                # No '>' or '{' — take the rest as name and assume no args
                                # stop at whitespace if present
                                next_space = rt.find(' ', start)
                                if next_space == -1:
                                    func_name = rt[start:]
                                    args_start = -1
                                else:
                                    func_name = rt[start:next_space]
                                    args_start = -1
                        else:
                            func_name = rt[start:end_name]
                            # start searching for args after the closing '>'
                            args_start = rt.find('{', end_name)

                        # Find JSON args if present by matching braces from args_start
                        args_part = '{}'
                        if args_start != -1 and args_start is not None:
                            depth = 0
                            for i in range(args_start, len(rt)):
                                if rt[i] == '{':
                                    depth += 1
                                elif rt[i] == '}':
                                    depth -= 1
                                    if depth == 0:
                                        args_part = rt[args_start:i+1]
                                        break
                        try:
                            func_args = json.loads(args_part)
                        except Exception:
                            logger.debug('Failed to parse inline function args as JSON, using empty dict', exc_info=True)
                            func_args = {}

                        logger.info(f"Detected inline function intent from model: {func_name} with args: {func_args}")

                        if not self.mcp_client:
                            logger.error('Model requested tool execution but no MCP client is available')
                            raise RuntimeError('No MCP client available')

                        # Execute the tool
                        logger.info(f"Executing MCP tool '{func_name}' with args: {func_args}")
                        result = await self.mcp_client.call_tool(func_name, func_args)
                        logger.info(f"MCP tool '{func_name}' execution completed")

                        # Extract tool content
                        try:
                            if getattr(result, 'structuredContent', None) is not None:
                                tool_result = result.structuredContent
                            elif hasattr(result, 'content') and result.content:
                                c = result.content[0]
                                tool_result = c.text if getattr(c, 'text', None) else c
                            else:
                                tool_result = result
                        except Exception:
                            tool_result = str(result)

                        logger.debug(f"Tool result (truncated): {str(tool_result)[:2000]}")

                        # Auto-summarize the tool result and return a human-friendly
                        # summary immediately (so we don't rely on a second LLM pass).
                        def _auto_summarize_inline(result_obj: Any) -> Tuple[str, Any]:
                            try:
                                parsed = result_obj
                                if isinstance(result_obj, str):
                                    try:
                                        parsed = json.loads(result_obj)
                                    except Exception:
                                        parsed = result_obj

                                if isinstance(parsed, list):
                                    count = len(parsed)
                                    sample = parsed[0] if count > 0 else None
                                    if count == 0:
                                        return ("No results found.", parsed)
                                    if isinstance(sample, dict):
                                        keys = list(sample.keys())[:5]
                                        sample_str = ', '.join([f"{k}: {sample.get(k)}" for k in keys])
                                        return (f"Found {count} result(s). Example: {sample_str}", parsed)
                                    return (f"Found {count} result(s). Example: {str(sample)[:200]}", parsed)

                                if isinstance(parsed, dict):
                                    total = parsed.get('TotalRecords') or parsed.get('Total') or parsed.get('Count')
                                    if total:
                                        return (f"Found {total} result(s).", parsed)
                                    keys = list(parsed.keys())[:6]
                                    brief = ', '.join([f"{k}: {parsed.get(k)}" for k in keys])
                                    return (f"Found 1 result. {brief}", parsed)

                                return (str(parsed)[:1000], parsed)
                            except Exception:
                                return (str(result_obj), result_obj)

                        summary_text, raw_for_response = _auto_summarize_inline(tool_result)

                        try:
                            raw_text = json.dumps(raw_for_response, default=str, indent=2) if not isinstance(raw_for_response, str) else raw_for_response
                        except Exception:
                            raw_text = str(raw_for_response)

                        # Add assistant tool call entry so GROQ sees the function call
                        assistant_tool_call = [{
                            'id': func_name,
                            'type': 'function',
                            'function': {'name': func_name, 'arguments': json.dumps(func_args)}
                        }]
                        conversation_manager.add_assistant_message('', assistant_tool_call)

                        # Add the raw tool output as a tool message with tool_call_id
                        conversation_manager.add_tool_message_with_id(func_name, raw_text, tool_call_id=func_name)

                        # Instead of returning an automatically generated summary here,
                        # allow the final synthesis pass below to call the LLM and
                        # produce a more natural human-readable reply.
                        logger.info("Tool output added; proceeding to final LLM synthesis for inline execution")
                        outgoing_messages = []
                        for m in conversation_manager.get_messages():
                            msg = dict(m)
                            if msg.get('role') == 'assistant' and 'tool_calls' in msg:
                                # ensure function.arguments string exists
                                normalized_tool_calls = []
                                for tc in msg['tool_calls']:
                                    tc_copy = dict(tc)
                                    func = tc_copy.get('function') or {}
                                    raw_args = None
                                    if 'arguments' in tc_copy and tc_copy['arguments'] is not None:
                                        raw_args = tc_copy['arguments']
                                    elif 'arguments' in func and func['arguments'] is not None:
                                        raw_args = func['arguments']
                                    if raw_args is None:
                                        args_s = '{}'
                                    elif isinstance(raw_args, str):
                                        args_s = raw_args
                                    else:
                                        try:
                                            args_s = json.dumps(raw_args)
                                        except Exception:
                                            args_s = str(raw_args)
                                    normalized_tool_calls.append({'id': tc_copy.get('id'), 'type': 'function', 'function': {'name': func.get('name'), 'arguments': args_s}})
                                msg['tool_calls'] = normalized_tool_calls

                            if msg.get('role') == 'tool' and isinstance(msg.get('content'), (list, tuple)):
                                try:
                                    parts = []
                                    for c in msg['content']:
                                        if isinstance(c, dict) and c.get('text'):
                                            parts.append(c.get('text'))
                                        else:
                                            parts.append(str(c))
                                    msg['content'] = '\n'.join(parts)
                                except Exception:
                                    msg['content'] = str(msg['content'])

                            outgoing_messages.append(msg)

                        try:
                            import json as _json
                            logger.debug("Outgoing messages for final synthesis: %s", _json.dumps(outgoing_messages, default=str)[:4000])
                        except Exception:
                            logger.debug(f"Outgoing messages for final synthesis (truncated): {[str(m)[:500] for m in outgoing_messages]}")

                        # Insert transient system prompt to synthesize a human-friendly reply
                        synth_prompt = {
                            'role': 'system',
                            'content': (
                                'You are an assistant that converts tool outputs into a concise, human-readable '
                                'summary for the user. Use the tool output(s) present in the conversation and do not '
                                'repeat function call syntax or tool names. Answer directly and clearly.'
                            )
                        }
                        final_messages = [synth_prompt] + outgoing_messages

                        final_response = self.client.chat.completions.create(
                            model=self.model,
                            messages=final_messages,
                            temperature=0.7,
                            max_tokens=1000
                        )

                        final_content = final_response.choices[0].message.content or "I processed your request."
                        conversation_manager.add_assistant_message(final_content)

                        logger.info("Final human-readable response generated after tool execution")
                        return final_content, True, [{"function": func_name, "arguments": func_args, "result": tool_result}]
                except Exception as e:
                    if detected:
                        logger.error(f"Failed during inline function handling: {e}", exc_info=True)
                    else:
                        logger.debug("No inline function intent detected in model response")

                # If we didn't detect or couldn't handle inline function, return original model text
                conversation_manager.add_assistant_message(response_content)
                return response_content, False, None

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise