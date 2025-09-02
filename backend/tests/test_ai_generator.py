import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test cases for AIGenerator class"""

    def test_init(self):
        """Test AIGenerator initialization"""
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        
        assert ai_gen.model == "claude-sonnet-4-20250514"
        assert ai_gen.base_params["model"] == "claude-sonnet-4-20250514"
        assert ai_gen.base_params["temperature"] == 0
        assert ai_gen.base_params["max_tokens"] == 800

    def test_system_prompt_content(self):
        """Test that system prompt contains required instructions"""
        prompt = AIGenerator.SYSTEM_PROMPT
        
        # Check for key instructions
        assert "search_course_content" in prompt
        assert "get_course_outline" in prompt
        assert "Tool Usage" in prompt or "Multi-Round Tool Usage" in prompt
        assert "Response Protocol" in prompt
        
        # Check for tool-specific guidance (updated for multi-round)
        assert "Course-specific content questions" in prompt
        assert "Course outline/structure questions" in prompt
        assert "UP TO 2 rounds" in prompt  # New multi-round capability

    @patch('anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic_class):
        """Test basic response generation without tools"""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Basic response without tools")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        # Test
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response("What is machine learning?")
        
        assert result == "Basic response without tools"
        mock_client.messages.create.assert_called_once()
        
        # Verify API call parameters
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["model"] == "claude-sonnet-4-20250514"
        assert call_args[1]["temperature"] == 0
        assert call_args[1]["max_tokens"] == 800
        assert call_args[1]["messages"][0]["content"] == "What is machine learning?"
        assert "tools" not in call_args[1]

    @patch('anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic_class):
        """Test response generation with conversation history"""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with history")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        # Test
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        history = "Previous conversation context"
        result = ai_gen.generate_response("Follow up question", conversation_history=history)
        
        assert result == "Response with history"
        
        # Check that history was included in system prompt
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation context" in system_content

    @patch('anthropic.Anthropic')
    def test_generate_response_with_tools(self, mock_anthropic_class):
        """Test response generation with tools but no tool usage"""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response without tool usage")]
        mock_response.stop_reason = "end_turn"  # No tool usage
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        # Mock tools
        tools = [
            {
                "name": "search_course_content",
                "description": "Search course content",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        ]
        tool_manager = Mock()
        
        # Test
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response("General question", tools=tools, tool_manager=tool_manager)
        
        assert result == "Response without tool usage"
        
        # Verify tools were passed to API
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["tools"] == tools
        assert call_args[1]["tool_choice"] == {"type": "auto"}

    @patch('anthropic.Anthropic')
    def test_generate_response_with_tool_usage(self, mock_anthropic_class):
        """Test response generation with tool usage flow"""
        # Setup mocks
        mock_client = Mock()
        
        # First call - tool usage
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_content_block = Mock()
        mock_content_block.type = "tool_use"
        mock_content_block.name = "search_course_content"
        mock_content_block.input = {"query": "test query"}
        mock_content_block.id = "tool_id_123"
        mock_tool_response.content = [mock_content_block]
        
        # Second call - final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final response after tool usage")]
        mock_final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Tool execution result"
        
        tools = [{"name": "search_course_content"}]
        
        # Test
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Search for something", 
            tools=tools, 
            tool_manager=tool_manager
        )
        
        assert result == "Final response after tool usage"
        
        # Verify tool was executed
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test query"
        )
        
        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2

    @patch('anthropic.Anthropic')
    def test_handle_tool_execution_multiple_tools(self, mock_anthropic_class):
        """Test handling multiple tool calls in one response"""
        # Setup mocks
        mock_client = Mock()
        
        # Tool usage response with multiple tools
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        
        mock_content_block1 = Mock()
        mock_content_block1.type = "tool_use"
        mock_content_block1.name = "search_course_content"
        mock_content_block1.input = {"query": "search query"}
        mock_content_block1.id = "tool_id_1"
        
        mock_content_block2 = Mock()
        mock_content_block2.type = "tool_use"
        mock_content_block2.name = "get_course_outline"
        mock_content_block2.input = {"course_name": "Test Course"}
        mock_content_block2.id = "tool_id_2"
        
        mock_tool_response.content = [mock_content_block1, mock_content_block2]
        
        # Final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Response using multiple tools")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.side_effect = ["Search result", "Outline result"]
        
        tools = [
            {"name": "search_course_content"},
            {"name": "get_course_outline"}
        ]
        
        # Test
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Complex query", 
            tools=tools, 
            tool_manager=tool_manager
        )
        
        assert result == "Response using multiple tools"
        
        # Verify both tools were executed
        assert tool_manager.execute_tool.call_count == 2
        tool_manager.execute_tool.assert_any_call("search_course_content", query="search query")
        tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="Test Course")

    @patch('anthropic.Anthropic')
    def test_handle_tool_execution_error(self, mock_anthropic_class):
        """Test handling of tool execution errors"""
        # Setup mocks
        mock_client = Mock()
        
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_content_block = Mock()
        mock_content_block.type = "tool_use"
        mock_content_block.name = "search_course_content"
        mock_content_block.input = {"query": "test query"}
        mock_content_block.id = "tool_id_123"
        mock_tool_response.content = [mock_content_block]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Error handled response")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        # Mock tool manager with error
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Tool execution failed: Database error"
        
        tools = [{"name": "search_course_content"}]
        
        # Test
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Search query", 
            tools=tools, 
            tool_manager=tool_manager
        )
        
        assert result == "Error handled response"
        
        # Verify that tool error was passed to final API call
        final_call_args = mock_client.messages.create.call_args_list[1]
        messages = final_call_args[1]["messages"]
        
        # Should have 3 messages: user, assistant (tool use), user (tool results)
        assert len(messages) >= 2
        
        # Find tool result message
        tool_result_message = None
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                tool_result_message = msg
                break
        
        assert tool_result_message is not None
        assert "Tool execution failed: Database error" in str(tool_result_message["content"])

    def test_process_tool_round_message_structure(self):
        """Test the structure of messages in tool processing flow"""
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        
        # Mock response with tool use
        mock_response = Mock()
        mock_content_block = Mock()
        mock_content_block.type = "tool_use"
        mock_content_block.name = "test_tool"
        mock_content_block.input = {"param": "value"}
        mock_content_block.id = "test_id"
        mock_response.content = [mock_content_block]
        
        # Initial messages
        initial_messages = [{"role": "user", "content": "test query"}]
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Tool result"
        
        # Test the new _process_tool_round method
        updated_messages = ai_gen._process_tool_round(initial_messages, mock_response, tool_manager, 1)
        
        # Verify message structure
        # Should have: original user message, assistant tool use, user tool results
        assert len(updated_messages) == 3
        assert updated_messages[0]["role"] == "user"
        assert updated_messages[1]["role"] == "assistant" 
        assert updated_messages[2]["role"] == "user"
        
        # Tool result should be properly structured
        tool_results = updated_messages[2]["content"]
        assert isinstance(tool_results, list)
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "test_id"
        assert tool_results[0]["content"] == "Tool result"

    @patch('anthropic.Anthropic')
    def test_api_key_validation(self, mock_anthropic_class):
        """Test that API key is properly passed to Anthropic client"""
        test_key = "test_anthropic_key_123"
        ai_gen = AIGenerator(test_key, "claude-sonnet-4-20250514")
        
        # Verify Anthropic client was initialized with correct key
        mock_anthropic_class.assert_called_once_with(api_key=test_key)

    @patch('anthropic.Anthropic')
    def test_anthropic_api_exception_handling(self, mock_anthropic_class):
        """Test handling of Anthropic API exceptions"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API error")
        mock_anthropic_class.return_value = mock_client
        
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        
        # With new error handling, should return graceful error message instead of raising
        result = ai_gen.generate_response("test query")
        assert "technical issue" in result
        assert "API error" in result


class TestAIGeneratorIntegration:
    """Integration tests for AIGenerator with real tool scenarios"""

    @patch('anthropic.Anthropic')
    def test_realistic_search_tool_flow(self, mock_anthropic_class):
        """Test a realistic search tool usage flow"""
        # Setup mock responses
        mock_client = Mock()
        
        # Tool usage response
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_content_block = Mock()
        mock_content_block.type = "tool_use"
        mock_content_block.name = "search_course_content"
        mock_content_block.input = {
            "query": "What is prompt caching?",
            "course_name": "Building Towards Computer Use"
        }
        mock_content_block.id = "search_123"
        mock_tool_response.content = [mock_content_block]
        
        # Final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Prompt caching retains some of the results of processing prompts between invocations...")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        # Mock tool manager with realistic search result
        tool_manager = Mock()
        search_result = """[Building Towards Computer Use - Lesson 2]
        Prompt caching retains some of the results of processing prompts between invocation
        to the model, which can be a large cost and latency saver."""
        tool_manager.execute_tool.return_value = search_result
        
        # Mock tools
        tools = [{
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "course_name": {"type": "string"}
                },
                "required": ["query"]
            }
        }]
        
        # Test
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "What is prompt caching in the Building Towards Computer Use course?",
            tools=tools,
            tool_manager=tool_manager
        )
        
        # Verify realistic flow worked
        assert "Prompt caching retains" in result
        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="What is prompt caching?",
            course_name="Building Towards Computer Use"
        )

    @patch('anthropic.Anthropic')  
    def test_outline_tool_flow(self, mock_anthropic_class):
        """Test outline tool usage flow"""
        # Setup mock responses
        mock_client = Mock()
        
        # Tool usage response
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_content_block = Mock()
        mock_content_block.type = "tool_use"
        mock_content_block.name = "get_course_outline"
        mock_content_block.input = {"course_name": "MCP"}
        mock_content_block.id = "outline_123"
        mock_tool_response.content = [mock_content_block]
        
        # Final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="The MCP course has 4 lessons covering...")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        # Mock tool manager with outline result
        tool_manager = Mock()
        outline_result = """Course: Model Context Protocol (MCP)
Course Link: https://example.com/mcp
Instructor: Test Instructor

Lessons:
  1: Introduction to MCP
  2: Building MCP Servers
  3: MCP Client Integration
  4: Advanced MCP Patterns"""
        tool_manager.execute_tool.return_value = outline_result
        
        # Mock tools
        tools = [{
            "name": "get_course_outline",
            "description": "Get course outline",
            "input_schema": {
                "type": "object",
                "properties": {"course_name": {"type": "string"}},
                "required": ["course_name"]
            }
        }]
        
        # Test
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "What lessons are in the MCP course?",
            tools=tools,
            tool_manager=tool_manager
        )
        
        # Verify outline flow worked
        assert "The MCP course has 4 lessons" in result
        tool_manager.execute_tool.assert_called_once_with("get_course_outline", course_name="MCP")