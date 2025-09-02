import pytest
from unittest.mock import Mock, patch, MagicMock
from rag_system import RAGSystem
from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool
from ai_generator import AIGenerator
from vector_store import VectorStore
from session_manager import SessionManager


class TestRAGSystemInit:
    """Test RAG System initialization"""

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_init_components(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_processor, sample_config):
        """Test that all components are properly initialized"""
        rag_system = RAGSystem(sample_config)
        
        # Verify component initialization
        mock_doc_processor.assert_called_once_with(sample_config.CHUNK_SIZE, sample_config.CHUNK_OVERLAP)
        mock_vector_store.assert_called_once_with(sample_config.CHROMA_PATH, sample_config.EMBEDDING_MODEL, sample_config.MAX_RESULTS)
        mock_ai_gen.assert_called_once_with(sample_config.ANTHROPIC_API_KEY, sample_config.ANTHROPIC_MODEL)
        mock_session_mgr.assert_called_once_with(sample_config.MAX_HISTORY)
        
        # Verify tools are set up
        assert rag_system.tool_manager is not None
        assert rag_system.search_tool is not None
        assert rag_system.outline_tool is not None

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_tools_registration(self, mock_session_mgr, mock_ai_gen, mock_vector_store, mock_doc_processor, sample_config):
        """Test that tools are properly registered"""
        rag_system = RAGSystem(sample_config)
        
        # Get tool definitions to verify registration
        tool_defs = rag_system.tool_manager.get_tool_definitions()
        
        # Should have both search and outline tools
        assert len(tool_defs) == 2
        tool_names = [tool["name"] for tool in tool_defs]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names


class TestRAGSystemQuery:
    """Test RAG System query processing"""

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.SessionManager')
    def test_query_without_session(self, mock_session_mgr, mock_vector_store, mock_doc_processor, sample_config):
        """Test query processing without session ID"""
        # Setup mocks
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Test response"
        
        with patch('rag_system.AIGenerator', return_value=mock_ai_gen):
            rag_system = RAGSystem(sample_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=[])
            rag_system.tool_manager.reset_sources = Mock()
            
            response, sources = rag_system.query("What is machine learning?")
            
            # Verify response
            assert response == "Test response"
            assert sources == []
            
            # Verify AI generator was called correctly
            mock_ai_gen.generate_response.assert_called_once()
            call_args = mock_ai_gen.generate_response.call_args
            assert "Answer this question about course materials: What is machine learning?" in call_args[1]["query"]
            assert call_args[1]["conversation_history"] is None

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.SessionManager')
    def test_query_with_session(self, mock_session_mgr, mock_vector_store, mock_doc_processor, sample_config):
        """Test query processing with session ID"""
        # Setup mocks
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Test response with history"
        
        mock_session_mgr_instance = Mock()
        mock_session_mgr_instance.get_conversation_history.return_value = "Previous conversation"
        mock_session_mgr.return_value = mock_session_mgr_instance
        
        with patch('rag_system.AIGenerator', return_value=mock_ai_gen):
            rag_system = RAGSystem(sample_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=[])
            rag_system.tool_manager.reset_sources = Mock()
            
            response, sources = rag_system.query("Follow up question", session_id="session_123")
            
            # Verify session manager was used
            mock_session_mgr_instance.get_conversation_history.assert_called_once_with("session_123")
            mock_session_mgr_instance.add_exchange.assert_called_once_with("session_123", "Follow up question", "Test response with history")
            
            # Verify AI generator received history
            call_args = mock_ai_gen.generate_response.call_args
            assert call_args[1]["conversation_history"] == "Previous conversation"

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.SessionManager')
    def test_query_with_sources(self, mock_session_mgr, mock_vector_store, mock_doc_processor, sample_config):
        """Test query processing that returns sources"""
        # Setup mocks
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Response using search results"
        
        test_sources = [
            {"display": "Course A - Lesson 1", "link": "https://test1.com"},
            {"display": "Course B - Lesson 2", "link": "https://test2.com"}
        ]
        
        with patch('rag_system.AIGenerator', return_value=mock_ai_gen):
            rag_system = RAGSystem(sample_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=test_sources)
            rag_system.tool_manager.reset_sources = Mock()
            
            response, sources = rag_system.query("Search for course content")
            
            # Verify sources are returned
            assert response == "Response using search results"
            assert sources == test_sources
            
            # Verify sources are reset after retrieval
            rag_system.tool_manager.get_last_sources.assert_called_once()
            rag_system.tool_manager.reset_sources.assert_called_once()

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.SessionManager')  
    def test_query_with_tools(self, mock_session_mgr, mock_vector_store, mock_doc_processor, sample_config):
        """Test that tools are properly passed to AI generator"""
        # Setup mocks
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Tool-enhanced response"
        
        with patch('rag_system.AIGenerator', return_value=mock_ai_gen):
            rag_system = RAGSystem(sample_config)
            rag_system.tool_manager.get_last_sources = Mock(return_value=[])
            rag_system.tool_manager.reset_sources = Mock()
            
            response, sources = rag_system.query("What courses are available?")
            
            # Verify tools were passed to AI generator
            call_args = mock_ai_gen.generate_response.call_args
            assert "tools" in call_args[1]
            assert "tool_manager" in call_args[1]
            assert call_args[1]["tool_manager"] is rag_system.tool_manager
            
            # Verify tool definitions were passed
            tools = call_args[1]["tools"]
            assert isinstance(tools, list)
            assert len(tools) > 0

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.SessionManager')
    def test_query_ai_generator_exception(self, mock_session_mgr, mock_vector_store, mock_doc_processor, sample_config):
        """Test handling of AI generator exceptions"""
        # Setup mock that raises exception
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.side_effect = Exception("API error")
        
        with patch('rag_system.AIGenerator', return_value=mock_ai_gen):
            rag_system = RAGSystem(sample_config)
            
            # Should propagate exception (current behavior)
            with pytest.raises(Exception, match="API error"):
                rag_system.query("Test query")


class TestRAGSystemDocumentProcessing:
    """Test document processing functionality"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_add_course_document_success(self, mock_session_mgr, mock_ai_gen, mock_vector_store, sample_config, sample_course, sample_course_chunks):
        """Test successful course document addition"""
        # Setup mocks
        mock_doc_processor = Mock()
        mock_doc_processor.process_course_document.return_value = (sample_course, sample_course_chunks)
        
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        
        with patch('rag_system.DocumentProcessor', return_value=mock_doc_processor):
            rag_system = RAGSystem(sample_config)
            
            course, chunk_count = rag_system.add_course_document("/path/to/course.txt")
            
            # Verify document processing
            mock_doc_processor.process_course_document.assert_called_once_with("/path/to/course.txt")
            
            # Verify vector store operations
            mock_vector_store_instance.add_course_metadata.assert_called_once_with(sample_course)
            mock_vector_store_instance.add_course_content.assert_called_once_with(sample_course_chunks)
            
            # Verify return values
            assert course == sample_course
            assert chunk_count == len(sample_course_chunks)

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_add_course_document_processing_error(self, mock_session_mgr, mock_ai_gen, mock_vector_store, sample_config):
        """Test handling of document processing errors"""
        # Setup mock that raises exception
        mock_doc_processor = Mock()
        mock_doc_processor.process_course_document.side_effect = Exception("Processing error")
        
        with patch('rag_system.DocumentProcessor', return_value=mock_doc_processor):
            rag_system = RAGSystem(sample_config)
            
            course, chunk_count = rag_system.add_course_document("/path/to/bad_course.txt")
            
            # Should return None and 0 on error
            assert course is None
            assert chunk_count == 0

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_add_course_folder_success(self, mock_listdir, mock_exists, mock_session_mgr, mock_ai_gen, mock_vector_store, sample_config, sample_course, sample_course_chunks):
        """Test successful folder processing"""
        # Setup file system mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.pdf", "readme.md"]
        
        # Setup document processor mock
        mock_doc_processor = Mock()
        mock_doc_processor.process_course_document.return_value = (sample_course, sample_course_chunks)
        
        # Setup vector store mock
        mock_vector_store_instance = Mock()
        mock_vector_store_instance.get_existing_course_titles.return_value = []
        mock_vector_store.return_value = mock_vector_store_instance
        
        with patch('rag_system.DocumentProcessor', return_value=mock_doc_processor):
            rag_system = RAGSystem(sample_config)
            
            total_courses, total_chunks = rag_system.add_course_folder("/path/to/docs")
            
            # Should process .txt and .pdf files
            assert mock_doc_processor.process_course_document.call_count == 2
            assert total_courses == 2  # Both files processed as new courses
            assert total_chunks == len(sample_course_chunks) * 2

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('os.path.exists')
    def test_add_course_folder_nonexistent(self, mock_exists, mock_session_mgr, mock_ai_gen, mock_vector_store, sample_config):
        """Test handling of nonexistent folder"""
        mock_exists.return_value = False
        
        with patch('rag_system.DocumentProcessor'):
            rag_system = RAGSystem(sample_config)
            
            total_courses, total_chunks = rag_system.add_course_folder("/nonexistent/path")
            
            assert total_courses == 0
            assert total_chunks == 0

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_add_course_folder_skip_existing(self, mock_listdir, mock_exists, mock_session_mgr, mock_ai_gen, mock_vector_store, sample_config, sample_course, sample_course_chunks):
        """Test skipping existing courses"""
        # Setup file system mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt"]
        
        # Setup document processor mock
        mock_doc_processor = Mock()
        mock_doc_processor.process_course_document.return_value = (sample_course, sample_course_chunks)
        
        # Setup vector store mock with existing course
        mock_vector_store_instance = Mock()
        mock_vector_store_instance.get_existing_course_titles.return_value = [sample_course.title]
        mock_vector_store.return_value = mock_vector_store_instance
        
        with patch('rag_system.DocumentProcessor', return_value=mock_doc_processor):
            rag_system = RAGSystem(sample_config)
            
            total_courses, total_chunks = rag_system.add_course_folder("/path/to/docs")
            
            # Course should be processed but not added (already exists)
            mock_doc_processor.process_course_document.assert_called_once()
            mock_vector_store_instance.add_course_metadata.assert_not_called()
            mock_vector_store_instance.add_course_content.assert_not_called()
            assert total_courses == 0
            assert total_chunks == 0


class TestRAGSystemAnalytics:
    """Test analytics functionality"""

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_get_course_analytics(self, mock_session_mgr, mock_ai_gen, mock_doc_processor, sample_config):
        """Test course analytics retrieval"""
        # Setup vector store mock
        mock_vector_store = Mock()
        mock_vector_store.get_course_count.return_value = 3
        mock_vector_store.get_existing_course_titles.return_value = ["Course A", "Course B", "Course C"]
        
        with patch('rag_system.VectorStore', return_value=mock_vector_store):
            rag_system = RAGSystem(sample_config)
            
            analytics = rag_system.get_course_analytics()
            
            assert analytics["total_courses"] == 3
            assert analytics["course_titles"] == ["Course A", "Course B", "Course C"]


class TestRAGSystemEndToEnd:
    """End-to-end integration tests"""

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.SessionManager')
    @patch('anthropic.Anthropic')
    def test_complete_query_flow_with_search(self, mock_anthropic_class, mock_session_mgr, mock_vector_store, mock_doc_processor, sample_config):
        """Test complete query flow that uses search tool"""
        # Setup Anthropic client mock
        mock_client = Mock()
        
        # Tool usage response
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_content_block = Mock()
        mock_content_block.type = "tool_use"
        mock_content_block.name = "search_course_content"
        mock_content_block.input = {"query": "prompt caching", "course_name": "Building Towards Computer Use"}
        mock_content_block.id = "search_123"
        mock_tool_response.content = [mock_content_block]
        
        # Final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Prompt caching retains processing results between calls to reduce costs and latency.")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        # Setup vector store mock
        mock_vector_store_instance = Mock()
        search_results = Mock()
        search_results.error = None
        search_results.is_empty.return_value = False
        search_results.documents = ["Prompt caching retains some results of processing prompts between invocations..."]
        search_results.metadata = [{"course_title": "Building Towards Computer Use", "lesson_number": 2}]
        mock_vector_store_instance.search.return_value = search_results
        mock_vector_store_instance.get_lesson_link.return_value = "https://learn.example.com/lesson2"
        mock_vector_store.return_value = mock_vector_store_instance
        
        # Create RAG system
        rag_system = RAGSystem(sample_config)
        
        # Execute query
        response, sources = rag_system.query("What is prompt caching in the Building Towards Computer Use course?")
        
        # Verify end-to-end flow
        assert "Prompt caching retains processing results" in response
        assert len(sources) == 1
        assert sources[0]["display"] == "Building Towards Computer Use - Lesson 2"
        assert sources[0]["link"] == "https://learn.example.com/lesson2"
        
        # Verify vector store search was called correctly
        mock_vector_store_instance.search.assert_called_once_with(
            query="prompt caching",
            course_name="Building Towards Computer Use",
            lesson_number=None
        )

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.SessionManager')
    @patch('anthropic.Anthropic')
    def test_complete_query_flow_with_outline(self, mock_anthropic_class, mock_session_mgr, mock_vector_store, mock_doc_processor, sample_config):
        """Test complete query flow that uses outline tool"""
        # Setup Anthropic client mock
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
        mock_final_response.content = [Mock(text="The MCP course covers 4 lessons: Introduction, Building Servers, Client Integration, and Advanced Patterns.")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        # Setup vector store mock
        mock_vector_store_instance = Mock()
        outline_data = {
            'course_title': 'Model Context Protocol (MCP)',
            'course_link': 'https://example.com/mcp',
            'instructor': 'MCP Team',
            'lessons': [
                {'lesson_number': 1, 'lesson_title': 'Introduction to MCP', 'lesson_link': 'https://lesson1.com'},
                {'lesson_number': 2, 'lesson_title': 'Building MCP Servers', 'lesson_link': 'https://lesson2.com'},
                {'lesson_number': 3, 'lesson_title': 'MCP Client Integration', 'lesson_link': 'https://lesson3.com'},
                {'lesson_number': 4, 'lesson_title': 'Advanced MCP Patterns', 'lesson_link': 'https://lesson4.com'}
            ]
        }
        mock_vector_store_instance.get_course_outline.return_value = outline_data
        mock_vector_store.return_value = mock_vector_store_instance
        
        # Create RAG system
        rag_system = RAGSystem(sample_config)
        
        # Execute query
        response, sources = rag_system.query("What lessons are in the MCP course?")
        
        # Verify end-to-end flow
        assert "The MCP course covers 4 lessons" in response
        
        # Verify outline tool was called correctly
        mock_vector_store_instance.get_course_outline.assert_called_once_with("MCP")

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.SessionManager')
    @patch('anthropic.Anthropic')
    def test_query_failure_propagation(self, mock_anthropic_class, mock_session_mgr, mock_vector_store, mock_doc_processor, sample_config):
        """Test that query failures are properly propagated"""
        # Setup API client to fail
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API key invalid")
        mock_anthropic_class.return_value = mock_client
        
        # Create RAG system
        rag_system = RAGSystem(sample_config)
        
        # Query should fail
        with pytest.raises(Exception, match="API key invalid"):
            rag_system.query("Test query")

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.SessionManager')
    @patch('anthropic.Anthropic')
    def test_search_tool_error_handling(self, mock_anthropic_class, mock_session_mgr, mock_vector_store, mock_doc_processor, sample_config):
        """Test handling of search tool errors"""
        # Setup Anthropic client mock for tool usage
        mock_client = Mock()
        
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_content_block = Mock()
        mock_content_block.type = "tool_use"
        mock_content_block.name = "search_course_content"
        mock_content_block.input = {"query": "test"}
        mock_content_block.id = "search_123"
        mock_tool_response.content = [mock_content_block]
        
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="I apologize, but I couldn't search the course content due to a database error.")]
        
        mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        mock_anthropic_class.return_value = mock_client
        
        # Setup vector store to return error
        mock_vector_store_instance = Mock()
        error_results = Mock()
        error_results.error = "Database connection failed"
        mock_vector_store_instance.search.return_value = error_results
        mock_vector_store.return_value = mock_vector_store_instance
        
        # Create RAG system
        rag_system = RAGSystem(sample_config)
        
        # Execute query
        response, sources = rag_system.query("Search for something")
        
        # Should handle error gracefully
        assert "database error" in response
        assert len(sources) == 0  # No sources when search fails