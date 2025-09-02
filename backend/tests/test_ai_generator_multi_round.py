import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator


class TestMultiRoundAIGenerator:
    """Test cases for multi-round sequential tool calling in AIGenerator"""

    def test_system_prompt_multi_round_content(self):
        """Test that updated system prompt contains multi-round instructions"""
        prompt = AIGenerator.SYSTEM_PROMPT
        
        # Check for multi-round specific content
        assert "UP TO 2 rounds of tool calls" in prompt
        assert "Multi-Round Tool Usage" in prompt
        assert "Round 1:" in prompt
        assert "Round 2:" in prompt
        assert "Examples of multi-round usage" in prompt
        assert "Comprehensive" in prompt  # New response requirement

    @patch('anthropic.Anthropic')
    def test_single_round_sufficient(self, mock_anthropic_class):
        """Test that simple queries still work with single tool call"""
        # Setup mock client
        mock_client = Mock()
        
        # Single response without tool usage
        mock_response = Mock()
        mock_response.content = [Mock(text="Simple direct answer")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        # Test
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        tool_manager = Mock()
        
        result = ai_gen.generate_response(
            "What is machine learning?", 
            tools=tools, 
            tool_manager=tool_manager
        )
        
        assert result == "Simple direct answer"
        # Should only make one API call since no tool usage
        assert mock_client.messages.create.call_count == 1

    @patch('anthropic.Anthropic')
    def test_two_round_tool_usage(self, mock_anthropic_class):
        """Test genuine multi-round tool usage"""
        mock_client = Mock()
        
        # Round 1: Tool usage response
        mock_round1_response = Mock()
        mock_round1_response.stop_reason = "tool_use"
        mock_content_block1 = Mock()
        mock_content_block1.type = "tool_use"
        mock_content_block1.name = "get_course_outline"
        mock_content_block1.input = {"course_name": "MCP"}
        mock_content_block1.id = "tool_1"
        mock_round1_response.content = [mock_content_block1]
        
        # Round 2: Another tool usage response
        mock_round2_response = Mock()
        mock_round2_response.stop_reason = "tool_use"
        mock_content_block2 = Mock()
        mock_content_block2.type = "tool_use"
        mock_content_block2.name = "search_course_content"
        mock_content_block2.input = {"query": "authentication", "lesson_number": 3}
        mock_content_block2.id = "tool_2"
        mock_round2_response.content = [mock_content_block2]
        
        # Final response (no tools)
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Based on the course outline and lesson content, authentication is covered...")]
        mock_final_response.stop_reason = "end_turn"
        
        # Set up call sequence
        mock_client.messages.create.side_effect = [
            mock_round1_response,  # Round 1
            mock_round2_response,  # Round 2 
            mock_final_response    # Final synthesis
        ]
        mock_anthropic_class.return_value = mock_client
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.side_effect = [
            "Course outline with lesson 3: Authentication",  # Round 1 result
            "Detailed authentication content from lesson 3"  # Round 2 result
        ]
        
        tools = [
            {"name": "get_course_outline"},
            {"name": "search_course_content"}
        ]
        
        # Test
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Tell me about authentication in lesson 3 of the MCP course",
            tools=tools,
            tool_manager=tool_manager
        )
        
        # Verify result
        assert "authentication is covered" in result
        
        # Verify 3 API calls were made (2 tool rounds + 1 final)
        assert mock_client.messages.create.call_count == 3
        
        # Verify both tools were executed
        assert tool_manager.execute_tool.call_count == 2
        tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="MCP")
        tool_manager.execute_tool.assert_any_call("search_course_content", query="authentication", lesson_number=3)

    @patch('anthropic.Anthropic')
    def test_max_rounds_enforcement(self, mock_anthropic_class):
        """Test that system stops after max_rounds (default 2)"""
        mock_client = Mock()
        
        # Both rounds return tool_use
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_content_block = Mock()
        mock_content_block.type = "tool_use"
        mock_content_block.name = "search_course_content"
        mock_content_block.input = {"query": "test"}
        mock_content_block.id = "tool_id"
        mock_tool_response.content = [mock_content_block]
        
        # Final response (forced, no tools available)
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final synthesized response")]
        mock_final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [
            mock_tool_response,    # Round 1
            mock_tool_response,    # Round 2 
            mock_final_response    # Final (no tools)
        ]
        mock_anthropic_class.return_value = mock_client
        
        # Mock tool manager
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Tool result"
        
        tools = [{"name": "search_course_content"}]
        
        # Test
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Complex query requiring multiple searches",
            tools=tools,
            tool_manager=tool_manager,
            max_rounds=2
        )
        
        # Should get final response
        assert result == "Final synthesized response"
        
        # Should make exactly 3 API calls (2 tool rounds + 1 final)
        assert mock_client.messages.create.call_count == 3
        
        # Should execute tools exactly 2 times (once per round)
        assert tool_manager.execute_tool.call_count == 2

    @patch('anthropic.Anthropic')
    def test_tool_error_handling_mid_sequence(self, mock_anthropic_class):
        """Test graceful handling of tool execution failures"""
        mock_client = Mock()
        
        # Round 1: Tool usage
        mock_round1_response = Mock()
        mock_round1_response.stop_reason = "tool_use"
        mock_content_block = Mock()
        mock_content_block.type = "tool_use"
        mock_content_block.name = "search_course_content"
        mock_content_block.input = {"query": "test"}
        mock_content_block.id = "tool_1"
        mock_round1_response.content = [mock_content_block]
        
        # Round 2: Another tool usage
        mock_round2_response = Mock()
        mock_round2_response.stop_reason = "tool_use"
        mock_content_block2 = Mock()
        mock_content_block2.type = "tool_use"
        mock_content_block2.name = "get_course_outline"
        mock_content_block2.input = {"course_name": "BadCourse"}
        mock_content_block2.id = "tool_2"
        mock_round2_response.content = [mock_content_block2]
        
        # Final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Response with partial information")]
        
        mock_client.messages.create.side_effect = [
            mock_round1_response,
            mock_round2_response,
            mock_final_response
        ]
        mock_anthropic_class.return_value = mock_client
        
        # Mock tool manager with one success, one failure
        tool_manager = Mock()
        tool_manager.execute_tool.side_effect = [
            "Successful search result",  # Round 1 succeeds
            Exception("Database connection failed")  # Round 2 fails
        ]
        
        tools = [{"name": "search_course_content"}, {"name": "get_course_outline"}]
        
        # Test
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Test query",
            tools=tools,
            tool_manager=tool_manager
        )
        
        # Should still get a response despite tool error
        assert result == "Response with partial information"
        
        # Should have made 3 API calls
        assert mock_client.messages.create.call_count == 3

    @patch('anthropic.Anthropic')
    def test_api_error_handling(self, mock_anthropic_class):
        """Test handling of API errors during multi-round flow"""
        mock_client = Mock()
        
        # First call succeeds, second call fails
        mock_success_response = Mock()
        mock_success_response.stop_reason = "tool_use"
        mock_content_block = Mock()
        mock_content_block.type = "tool_use"
        mock_content_block.name = "search_course_content"
        mock_content_block.input = {"query": "test"}
        mock_content_block.id = "tool_1"
        mock_success_response.content = [mock_content_block]
        
        mock_client.messages.create.side_effect = [
            mock_success_response,  # Round 1 succeeds
            Exception("API rate limit exceeded")  # Round 2 fails
        ]
        mock_anthropic_class.return_value = mock_client
        
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Tool result"
        
        tools = [{"name": "search_course_content"}]
        
        # Test
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Test query",
            tools=tools,
            tool_manager=tool_manager
        )
        
        # Should return graceful error message
        assert "technical issue" in result
        assert "API rate limit exceeded" in result

    @patch('anthropic.Anthropic')
    def test_conversation_history_preservation(self, mock_anthropic_class):
        """Test that multi-round context is maintained"""
        mock_client = Mock()
        
        # Round 1: Tool usage
        mock_round1_response = Mock()
        mock_round1_response.stop_reason = "tool_use"
        mock_content_block = Mock()
        mock_content_block.type = "tool_use"
        mock_content_block.name = "search_course_content"
        mock_content_block.input = {"query": "test"}
        mock_content_block.id = "tool_1"
        mock_round1_response.content = [mock_content_block]
        
        # Round 2: Final response (no more tools needed)
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Response with context")]
        mock_final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [
            mock_round1_response,
            mock_final_response
        ]
        mock_anthropic_class.return_value = mock_client
        
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Tool result"
        
        tools = [{"name": "search_course_content"}]
        
        # Test
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Test query",
            tools=tools,
            tool_manager=tool_manager
        )
        
        # Verify message history was built correctly
        # Round 2 call should have more messages (includes tool exchange from Round 1)
        round2_call = mock_client.messages.create.call_args_list[1]
        round2_messages = round2_call[1]["messages"]
        
        # Should have: user query, assistant tool use, user tool results
        assert len(round2_messages) == 3
        assert round2_messages[0]["role"] == "user"    # Original query
        assert round2_messages[1]["role"] == "assistant"  # Tool use
        assert round2_messages[2]["role"] == "user"    # Tool results

    @patch('anthropic.Anthropic')
    def test_custom_max_rounds(self, mock_anthropic_class):
        """Test that custom max_rounds parameter is respected"""
        mock_client = Mock()
        
        # Single tool usage response
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_content_block = Mock()
        mock_content_block.type = "tool_use"
        mock_content_block.name = "search_course_content"
        mock_content_block.input = {"query": "test"}
        mock_content_block.id = "tool_1"
        mock_tool_response.content = [mock_content_block]
        
        # Final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Single round result")]
        
        mock_client.messages.create.side_effect = [
            mock_tool_response,     # Round 1
            mock_final_response     # Final (no more rounds allowed)
        ]
        mock_anthropic_class.return_value = mock_client
        
        tool_manager = Mock()
        tool_manager.execute_tool.return_value = "Tool result"
        
        tools = [{"name": "search_course_content"}]
        
        # Test with max_rounds=1
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Test query",
            tools=tools,
            tool_manager=tool_manager,
            max_rounds=1
        )
        
        # Should make exactly 2 API calls (1 tool round + 1 final)
        assert mock_client.messages.create.call_count == 2
        
        # Should execute tool once
        assert tool_manager.execute_tool.call_count == 1

    @patch('anthropic.Anthropic')
    def test_early_termination_no_tool_manager(self, mock_anthropic_class):
        """Test that system terminates early if no tool_manager provided"""
        mock_client = Mock()
        
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_response.content = [Mock(text="Would use tools but can't")]
        mock_client.messages.create.return_value = mock_tool_response
        mock_anthropic_class.return_value = mock_client
        
        tools = [{"name": "search_course_content"}]
        
        # Test without tool_manager
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Test query",
            tools=tools,
            tool_manager=None  # No tool manager
        )
        
        # Should return direct response without executing tools
        assert result == "Would use tools but can't"
        
        # Should only make one API call
        assert mock_client.messages.create.call_count == 1


class TestMultiRoundIntegration:
    """Integration tests for multi-round functionality"""

    @patch('anthropic.Anthropic')
    def test_realistic_course_outline_then_search(self, mock_anthropic_class):
        """Test realistic scenario: get outline, then search specific lesson"""
        mock_client = Mock()
        
        # Round 1: Get course outline
        mock_outline_response = Mock()
        mock_outline_response.stop_reason = "tool_use"
        mock_outline_block = Mock()
        mock_outline_block.type = "tool_use"
        mock_outline_block.name = "get_course_outline"
        mock_outline_block.input = {"course_name": "React Basics"}
        mock_outline_block.id = "outline_1"
        mock_outline_response.content = [mock_outline_block]
        
        # Round 2: Search specific lesson
        mock_search_response = Mock()
        mock_search_response.stop_reason = "tool_use"
        mock_search_block = Mock()
        mock_search_block.type = "tool_use"
        mock_search_block.name = "search_course_content"
        mock_search_block.input = {"query": "hooks", "course_name": "React Basics", "lesson_number": 3}
        mock_search_block.id = "search_1"
        mock_search_response.content = [mock_search_block]
        
        # Final synthesis
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="React hooks are covered in lesson 3, which focuses on state management...")]
        mock_final_response.stop_reason = "end_turn"
        
        mock_client.messages.create.side_effect = [
            mock_outline_response,
            mock_search_response,
            mock_final_response
        ]
        mock_anthropic_class.return_value = mock_client
        
        # Mock tool results
        tool_manager = Mock()
        tool_manager.execute_tool.side_effect = [
            "Course: React Basics\nLessons:\n  3: State Management with Hooks",  # Outline
            "Hooks allow you to use state in functional components..."  # Search
        ]
        
        tools = [
            {"name": "get_course_outline"},
            {"name": "search_course_content"}
        ]
        
        # Test
        ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "What does lesson 3 cover about hooks in the React Basics course?",
            tools=tools,
            tool_manager=tool_manager
        )
        
        # Verify realistic multi-round response
        assert "React hooks" in result
        assert "lesson 3" in result
        assert "state management" in result
        
        # Verify sequence of tool calls
        assert tool_manager.execute_tool.call_count == 2
        tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="React Basics")
        tool_manager.execute_tool.assert_any_call("search_course_content", query="hooks", course_name="React Basics", lesson_number=3)