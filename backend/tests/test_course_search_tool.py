import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool"""

    def test_get_tool_definition(self, course_search_tool):
        """Test that tool definition is properly structured"""
        definition = course_search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "query" in definition["input_schema"]["required"]
        
        # Optional parameters should be present but not required
        assert "course_name" in definition["input_schema"]["properties"] 
        assert "lesson_number" in definition["input_schema"]["properties"]
        assert "course_name" not in definition["input_schema"]["required"]
        assert "lesson_number" not in definition["input_schema"]["required"]

    def test_execute_successful_search(self, mock_vector_store):
        """Test successful search execution"""
        # Setup mock to return successful results
        mock_results = SearchResults(
            documents=["Test content from course"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://test.link"
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Verify vector store was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result format
        assert "[Test Course - Lesson 1]" in result
        assert "Test content from course" in result
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["display"] == "Test Course - Lesson 1"

    def test_execute_with_course_filter(self, mock_vector_store):
        """Test search with course name filter"""
        mock_results = SearchResults(
            documents=["Filtered content"],
            metadata=[{"course_title": "Specific Course", "lesson_number": 2}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = None
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Specific Course")
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Specific Course", 
            lesson_number=None
        )
        
        assert "[Specific Course - Lesson 2]" in result
        assert "Filtered content" in result

    def test_execute_with_lesson_filter(self, mock_vector_store):
        """Test search with lesson number filter"""
        mock_results = SearchResults(
            documents=["Lesson-specific content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 3}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", lesson_number=3)
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=3
        )
        
        assert "[Test Course - Lesson 3]" in result
        assert "Lesson-specific content" in result

    def test_execute_with_both_filters(self, mock_vector_store):
        """Test search with both course and lesson filters"""
        mock_results = SearchResults(
            documents=["Highly specific content"],
            metadata=[{"course_title": "Specific Course", "lesson_number": 5}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Specific Course", lesson_number=5)
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Specific Course",
            lesson_number=5
        )
        
        assert "[Specific Course - Lesson 5]" in result

    def test_execute_empty_results(self, mock_vector_store):
        """Test handling of empty search results"""
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("nonexistent query")
        
        assert result == "No relevant content found."
        assert len(tool.last_sources) == 0

    def test_execute_empty_results_with_filters(self, mock_vector_store):
        """Test empty results with filter information"""
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Nonexistent Course", lesson_number=99)
        
        assert "No relevant content found in course 'Nonexistent Course' in lesson 99." in result

    def test_execute_search_error(self, mock_vector_store):
        """Test handling of search errors"""
        error_results = SearchResults.empty("Database connection failed")
        mock_vector_store.search.return_value = error_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert result == "Database connection failed"

    def test_format_results_multiple_documents(self, mock_vector_store):
        """Test formatting multiple search results"""
        mock_results = SearchResults(
            documents=[
                "First document content",
                "Second document content", 
                "Third document content"
            ],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
                {"course_title": "Course A", "lesson_number": 3}
            ],
            distances=[0.1, 0.2, 0.3]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = None
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Check that all documents are formatted correctly
        assert "[Course A - Lesson 1]" in result
        assert "[Course B - Lesson 2]" in result
        assert "[Course A - Lesson 3]" in result
        assert "First document content" in result
        assert "Second document content" in result
        assert "Third document content" in result
        
        # Check that sources are tracked
        assert len(tool.last_sources) == 3
        assert tool.last_sources[0]["display"] == "Course A - Lesson 1"
        assert tool.last_sources[1]["display"] == "Course B - Lesson 2"
        assert tool.last_sources[2]["display"] == "Course A - Lesson 3"

    def test_format_results_with_lesson_links(self, mock_vector_store):
        """Test that lesson links are properly included in sources"""
        mock_results = SearchResults(
            documents=["Content with link"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["display"] == "Test Course - Lesson 1"
        assert tool.last_sources[0]["link"] == "https://example.com/lesson1"

    def test_format_results_missing_metadata(self, mock_vector_store):
        """Test handling of missing metadata fields"""
        mock_results = SearchResults(
            documents=["Content with incomplete metadata"],
            metadata=[{}],  # Empty metadata
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Should handle missing course_title gracefully
        assert "[unknown]" in result
        assert "Content with incomplete metadata" in result

    def test_format_results_no_lesson_number(self, mock_vector_store):
        """Test formatting when lesson_number is None"""
        mock_results = SearchResults(
            documents=["Course-level content"],
            metadata=[{"course_title": "Test Course"}],  # No lesson_number
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Should not include lesson number in header
        assert "[Test Course]" in result
        assert "Course-level content" in result
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["display"] == "Test Course"

    def test_last_sources_reset(self, mock_vector_store):
        """Test that last_sources is properly managed across searches"""
        # First search
        mock_results1 = SearchResults(
            documents=["First content"],
            metadata=[{"course_title": "Course 1", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results1
        
        tool = CourseSearchTool(mock_vector_store)
        tool.execute("first query")
        assert len(tool.last_sources) == 1
        
        # Second search
        mock_results2 = SearchResults(
            documents=["Second content A", "Second content B"],
            metadata=[
                {"course_title": "Course 2", "lesson_number": 1},
                {"course_title": "Course 2", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        mock_vector_store.search.return_value = mock_results2
        
        tool.execute("second query")
        assert len(tool.last_sources) == 2  # Should replace, not append
        assert tool.last_sources[0]["display"] == "Course 2 - Lesson 1"

    def test_vector_store_exception_handling(self, mock_vector_store):
        """Test that exceptions from vector store are properly handled"""
        mock_vector_store.search.side_effect = Exception("Vector store crashed")
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Should not crash and should return error message
        # Note: The current implementation doesn't catch exceptions in execute(),
        # so this test documents the current behavior and can guide improvements
        with pytest.raises(Exception):
            tool.execute("test query")


class TestCourseSearchToolIntegration:
    """Integration tests for CourseSearchTool with ToolManager"""

    def test_tool_registration(self, course_search_tool):
        """Test that tool can be registered with ToolManager"""
        manager = ToolManager()
        manager.register_tool(course_search_tool)
        
        definitions = manager.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_tool_execution_through_manager(self, mock_vector_store):
        """Test executing tool through ToolManager"""
        mock_results = SearchResults(
            documents=["Manager execution test"],
            metadata=[{"course_title": "Manager Test", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        
        tool = CourseSearchTool(mock_vector_store)
        manager = ToolManager()
        manager.register_tool(tool)
        
        result = manager.execute_tool("search_course_content", query="test query")
        
        assert "[Manager Test - Lesson 1]" in result
        assert "Manager execution test" in result

    def test_sources_retrieval_through_manager(self, mock_vector_store):
        """Test retrieving sources through ToolManager"""
        mock_results = SearchResults(
            documents=["Source test content"],
            metadata=[{"course_title": "Source Test", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        
        tool = CourseSearchTool(mock_vector_store)
        manager = ToolManager()
        manager.register_tool(tool)
        
        # Execute search
        manager.execute_tool("search_course_content", query="test query")
        
        # Get sources
        sources = manager.get_last_sources()
        assert len(sources) == 1
        assert sources[0]["display"] == "Source Test - Lesson 1"
        
        # Reset sources
        manager.reset_sources()
        sources = manager.get_last_sources()
        assert len(sources) == 0