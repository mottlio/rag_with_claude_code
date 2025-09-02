import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
- **search_course_content**: Search within course materials for specific content
- **get_course_outline**: Get course title, course link, and complete lesson list with lesson numbers and titles

Multi-Round Tool Usage:
- You can make UP TO 2 rounds of tool calls per query
- Use multiple rounds when initial tool results suggest additional searches would be valuable
- Examples of multi-round usage:
  * Round 1: Search for general topic → Round 2: Search for specific details found in Round 1
  * Round 1: Get course outline → Round 2: Search specific lessons mentioned
  * Round 1: Search one course → Round 2: Search related course for comparison

Tool Usage Guidelines:
- **Round 1**: Use tools to get initial information
- **Round 2**: Refine search or gather additional context if needed
- **Termination**: When you have sufficient information, provide final answer without additional tools
- Synthesize information from ALL tool calls when multiple rounds are used
- If no relevant information is found across all tool attempts, state this clearly

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific content questions**: Use tools strategically (1-2 rounds as needed)
- **Course outline/structure questions**: Use get_course_outline tool, then search specific content if needed
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or round analysis
 - Do not mention "based on the tool results" or "using tools"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
5. **Comprehensive** - Integrate information from multiple tool rounds when used
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_rounds: int = 2) -> str:
        """
        Generate AI response with support for sequential tool calling.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool rounds allowed (default: 2)
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently
        system_content = self._build_system_content(conversation_history)
        
        # Initialize message list
        messages = [{"role": "user", "content": query}]
        
        # Main processing loop - up to max_rounds
        for round_num in range(1, max_rounds + 1):
            try:
                # Determine if tools should be available this round
                include_tools = round_num < max_rounds
                
                # Make API call
                response = self._make_api_call(messages, system_content, tools if include_tools else None, round_num)
                
                # Check termination conditions
                if response.stop_reason != "tool_use" or not tool_manager:
                    return response.content[0].text
                
                # Execute tools and prepare for next round
                messages = self._process_tool_round(messages, response, tool_manager, round_num)
                
            except Exception as e:
                # Graceful error handling
                return f"I apologize, but I encountered a technical issue while processing your request: {str(e)}"
        
        # Final synthesis round (no tools)
        try:
            final_response = self._make_api_call(messages, system_content, None, "final")
            return final_response.content[0].text
        except Exception as e:
            return f"I apologize, but I encountered a technical issue while providing my final response: {str(e)}"
    
    def _build_system_content(self, conversation_history: Optional[str]) -> str:
        """Build system content with optional conversation history."""
        return (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
    
    def _make_api_call(self, messages: List[Dict], system_content: str, 
                       tools: Optional[List], round_num) -> Any:
        """
        Make a single API call to Claude with error handling.
        
        Args:
            messages: Current message history
            system_content: System prompt content
            tools: Available tools (None to exclude tools)
            round_num: Round number for logging
            
        Returns:
            API response object
        """
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }
        
        # Add tools only if provided (allows for tool-less final round)
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        try:
            response = self.client.messages.create(**api_params)
            return response
        except Exception as e:
            print(f"API call failed in round {round_num}: {str(e)}")
            raise
    
    def _process_tool_round(self, messages: List[Dict], response, tool_manager, round_num: int) -> List[Dict]:
        """
        Process a tool use round and prepare messages for the next round.
        
        Args:
            messages: Current message list
            response: API response containing tool use requests
            tool_manager: Manager to execute tools
            round_num: Current round number for error reporting
            
        Returns:
            Updated message list for next API call
        """
        # Add AI's tool use response to conversation
        updated_messages = messages.copy()
        updated_messages.append({"role": "assistant", "content": response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                except Exception as e:
                    # Handle tool execution errors gracefully
                    print(f"Tool execution failed in round {round_num}: {str(e)}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Tool execution failed: {str(e)}"
                    })
        
        # Add tool results to conversation
        if tool_results:
            updated_messages.append({"role": "user", "content": tool_results})
        
        return updated_messages