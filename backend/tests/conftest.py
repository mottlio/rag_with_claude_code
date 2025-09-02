import pytest
import os
import sys
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any, Optional

# Add the backend directory to Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk, Source
from vector_store import SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem

@pytest.fixture
def sample_course():
    """Sample course data for testing"""
    return Course(
        title="Building Towards Computer Use with Anthropic",
        course_link="https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/",
        instructor="Colt Steele",
        lessons=[
            Lesson(
                lesson_number=0,
                title="Introduction", 
                lesson_link="https://learn.deeplearning.ai/courses/building-toward-computer-use-with-anthropic/lesson/a6k0z/introduction"
            ),
            Lesson(
                lesson_number=1,
                title="Getting Started with Claude",
                lesson_link="https://learn.deeplearning.ai/courses/lesson1"
            )
        ]
    )

@pytest.fixture 
def sample_course_chunks():
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="Welcome to Building Toward Computer Use with Anthropic. Built in partnership with Anthropic...",
            course_title="Building Towards Computer Use with Anthropic",
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="In this lesson, you'll learn how to use Claude's API to make basic requests...",
            course_title="Building Towards Computer Use with Anthropic", 
            lesson_number=1,
            chunk_index=1
        )
    ]

@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return SearchResults(
        documents=[
            "Welcome to Building Toward Computer Use with Anthropic. This course teaches...",
            "Claude can analyze images and understand screenshots..."
        ],
        metadata=[
            {
                "course_title": "Building Towards Computer Use with Anthropic",
                "lesson_number": 0
            },
            {
                "course_title": "Building Towards Computer Use with Anthropic", 
                "lesson_number": 1
            }
        ],
        distances=[0.1, 0.2]
    )

@pytest.fixture
def empty_search_results():
    """Empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[], 
        distances=[]
    )

@pytest.fixture
def error_search_results():
    """Error search results for testing"""
    return SearchResults.empty("Vector store connection failed")

@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock_store = Mock()
    mock_store.search.return_value = SearchResults(
        documents=["Test content"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.1]
    )
    mock_store.get_lesson_link.return_value = "https://test.link"
    mock_store.get_course_outline.return_value = {
        "course_title": "Test Course",
        "course_link": "https://test.course",
        "instructor": "Test Instructor",
        "lessons": [
            {"lesson_number": 1, "lesson_title": "Test Lesson", "lesson_link": "https://test.lesson"}
        ]
    }
    return mock_store

@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text="Test response")]
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response
    return mock_client

@pytest.fixture
def mock_tool_use_response():
    """Mock tool use response from Claude"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"
    
    # Mock content block for tool use
    mock_content_block = Mock()
    mock_content_block.type = "tool_use"
    mock_content_block.name = "search_course_content"
    mock_content_block.input = {"query": "test query"}
    mock_content_block.id = "test_id"
    
    mock_response.content = [mock_content_block]
    return mock_response

@pytest.fixture
def course_search_tool(mock_vector_store):
    """CourseSearchTool instance with mock vector store"""
    return CourseSearchTool(mock_vector_store)

@pytest.fixture
def course_outline_tool(mock_vector_store):
    """CourseOutlineTool instance with mock vector store"""
    return CourseOutlineTool(mock_vector_store)

@pytest.fixture
def tool_manager(course_search_tool, course_outline_tool):
    """ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    manager.register_tool(course_outline_tool)
    return manager

@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    config = Mock()
    config.ANTHROPIC_API_KEY = "test_api_key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config

@pytest.fixture
def mock_ai_generator(mock_anthropic_client):
    """Mock AI generator for testing"""
    ai_gen = AIGenerator("test_key", "claude-sonnet-4-20250514")
    ai_gen.client = mock_anthropic_client
    return ai_gen

# Test utilities
class TestDataHelper:
    """Helper class for creating test data"""
    
    @staticmethod
    def create_course_outline_data(course_title: str = "Test Course") -> Dict[str, Any]:
        return {
            'course_title': course_title,
            'course_link': f'https://{course_title.lower().replace(" ", "-")}.com',
            'instructor': 'Test Instructor',
            'lessons': [
                {'lesson_number': 1, 'lesson_title': 'Introduction', 'lesson_link': 'https://test1.com'},
                {'lesson_number': 2, 'lesson_title': 'Advanced Topics', 'lesson_link': 'https://test2.com'}
            ]
        }
    
    @staticmethod
    def create_search_results(num_results: int = 2) -> SearchResults:
        documents = [f"Test document {i+1} content" for i in range(num_results)]
        metadata = [
            {
                "course_title": "Test Course",
                "lesson_number": i+1
            } 
            for i in range(num_results)
        ]
        distances = [0.1 * (i+1) for i in range(num_results)]
        
        return SearchResults(
            documents=documents,
            metadata=metadata,
            distances=distances
        )

@pytest.fixture
def test_data_helper():
    """Test data helper instance"""
    return TestDataHelper()