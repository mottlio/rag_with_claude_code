import pytest
import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from typing import List, Dict, Any, Optional
from fastapi.testclient import TestClient

# Add the backend directory to Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk, Source
from vector_store import SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from session_manager import SessionManager

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

# API Testing Fixtures

@pytest.fixture
def mock_rag_system():
    """Mock RAG system for API testing"""
    mock_rag = Mock(spec=RAGSystem)
    mock_rag.query.return_value = (
        "This is a test response",
        [Source(display="Test Course - Lesson 1", link="https://test.link")]
    )
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Test Course 1", "Test Course 2"]
    }
    mock_rag.session_manager = Mock(spec=SessionManager)
    mock_rag.session_manager.create_session.return_value = "test_session_id"
    mock_rag.session_manager.clear_session.return_value = None
    mock_rag.add_course_folder.return_value = (2, 50)
    return mock_rag

@pytest.fixture
def test_app_factory():
    """Factory to create test FastAPI apps"""
    def _create_test_app(mock_rag_system=None):
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        from typing import List, Optional
        
        # Create test app without static files
        app = FastAPI(title="Test Course Materials RAG System")
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Use provided mock or default
        rag_system = mock_rag_system or Mock()
        
        class QueryRequest(BaseModel):
            query: str
            session_id: Optional[str] = None

        class QueryResponse(BaseModel):
            answer: str
            sources: List[Source]
            session_id: str

        class CourseStats(BaseModel):
            total_courses: int
            course_titles: List[str]
        
        @app.post("/api/query", response_model=QueryResponse)
        async def query_documents(request: QueryRequest):
            try:
                session_id = request.session_id
                if not session_id:
                    session_id = rag_system.session_manager.create_session()
                
                answer, sources = rag_system.query(request.query, session_id)
                
                return QueryResponse(
                    answer=answer,
                    sources=sources,
                    session_id=session_id
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/api/courses", response_model=CourseStats)
        async def get_course_stats():
            try:
                analytics = rag_system.get_course_analytics()
                return CourseStats(
                    total_courses=analytics["total_courses"],
                    course_titles=analytics["course_titles"]
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.delete("/api/sessions/{session_id}/clear")
        async def clear_session(session_id: str):
            try:
                rag_system.session_manager.clear_session(session_id)
                return {"status": "success", "message": f"Session {session_id} cleared"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/")
        async def root():
            return {"message": "Course Materials RAG System API"}
        
        return app
    
    return _create_test_app

@pytest.fixture
def test_client(test_app_factory, mock_rag_system):
    """Test client for API testing"""
    app = test_app_factory(mock_rag_system)
    return TestClient(app)

@pytest.fixture
def api_test_data():
    """Test data for API testing"""
    return {
        "valid_query": {
            "query": "What is computer use?",
            "session_id": "test_session_123"
        },
        "query_without_session": {
            "query": "How does Claude work?"
        },
        "empty_query": {
            "query": ""
        },
        "long_query": {
            "query": "A" * 1000
        }
    }

@pytest.fixture
def temp_dir():
    """Temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir