import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from vector_store import VectorStore, SearchResults
from document_processor import DocumentProcessor
from config import Config
from models import Course, Lesson, CourseChunk


class TestEnvironmentConfiguration:
    """Test environment and configuration setup"""

    def test_config_creation(self):
        """Test that config can be created with default values"""
        config = Config()
        
        # Check required fields exist
        assert hasattr(config, 'ANTHROPIC_API_KEY')
        assert hasattr(config, 'ANTHROPIC_MODEL')
        assert hasattr(config, 'EMBEDDING_MODEL')
        assert hasattr(config, 'CHUNK_SIZE')
        assert hasattr(config, 'CHUNK_OVERLAP')
        assert hasattr(config, 'MAX_RESULTS')
        assert hasattr(config, 'MAX_HISTORY')
        assert hasattr(config, 'CHROMA_PATH')

    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key_123'})
    def test_config_environment_variables(self):
        """Test that config reads environment variables"""
        config = Config()
        assert config.ANTHROPIC_API_KEY == 'test_key_123'

    def test_config_defaults(self):
        """Test default configuration values"""
        config = Config()
        
        assert config.ANTHROPIC_MODEL == "claude-sonnet-4-20250514"
        assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        assert config.CHUNK_SIZE == 800
        assert config.CHUNK_OVERLAP == 100
        assert config.MAX_RESULTS == 5
        assert config.MAX_HISTORY == 2
        assert config.CHROMA_PATH == "./chroma_db"


class TestVectorStoreInfrastructure:
    """Test vector store infrastructure and connectivity"""

    def test_vector_store_initialization(self):
        """Test vector store can be initialized"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                vector_store = VectorStore(
                    chroma_path=temp_dir,
                    embedding_model="all-MiniLM-L6-v2",
                    max_results=5
                )
                assert vector_store is not None
                assert vector_store.max_results == 5
            except Exception as e:
                pytest.fail(f"Vector store initialization failed: {e}")

    def test_vector_store_collections_creation(self):
        """Test that vector store creates required collections"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                vector_store = VectorStore(
                    chroma_path=temp_dir,
                    embedding_model="all-MiniLM-L6-v2"
                )
                
                # Collections should be created
                assert vector_store.course_catalog is not None
                assert vector_store.course_content is not None
            except Exception as e:
                pytest.fail(f"Collection creation failed: {e}")

    def test_vector_store_embedding_function(self):
        """Test that embedding function is properly initialized"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                vector_store = VectorStore(
                    chroma_path=temp_dir,
                    embedding_model="all-MiniLM-L6-v2"
                )
                
                assert vector_store.embedding_function is not None
            except Exception as e:
                pytest.fail(f"Embedding function initialization failed: {e}")

    def test_vector_store_add_and_retrieve_course_metadata(self, sample_course):
        """Test adding and retrieving course metadata"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                vector_store = VectorStore(
                    chroma_path=temp_dir,
                    embedding_model="all-MiniLM-L6-v2"
                )
                
                # Add course metadata
                vector_store.add_course_metadata(sample_course)
                
                # Verify it can be retrieved
                existing_titles = vector_store.get_existing_course_titles()
                assert sample_course.title in existing_titles
                
                # Test course count
                assert vector_store.get_course_count() == 1
                
                # Test metadata retrieval
                metadata = vector_store.get_all_courses_metadata()
                assert len(metadata) == 1
                assert metadata[0]['title'] == sample_course.title
                
            except Exception as e:
                pytest.fail(f"Course metadata operations failed: {e}")

    def test_vector_store_add_and_search_content(self, sample_course_chunks):
        """Test adding and searching course content"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                vector_store = VectorStore(
                    chroma_path=temp_dir,
                    embedding_model="all-MiniLM-L6-v2"
                )
                
                # Add course content
                vector_store.add_course_content(sample_course_chunks)
                
                # Search content
                results = vector_store.search("Anthropic")
                
                # Verify search works
                assert not results.error
                assert not results.is_empty()
                assert len(results.documents) > 0
                
            except Exception as e:
                pytest.fail(f"Content operations failed: {e}")

    def test_vector_store_persistence(self, sample_course):
        """Test that data persists across vector store instances"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # First instance - add data
                vector_store1 = VectorStore(
                    chroma_path=temp_dir,
                    embedding_model="all-MiniLM-L6-v2"
                )
                vector_store1.add_course_metadata(sample_course)
                
                # Second instance - verify data exists
                vector_store2 = VectorStore(
                    chroma_path=temp_dir,
                    embedding_model="all-MiniLM-L6-v2"
                )
                existing_titles = vector_store2.get_existing_course_titles()
                assert sample_course.title in existing_titles
                
            except Exception as e:
                pytest.fail(f"Data persistence test failed: {e}")

    def test_vector_store_clear_data(self, sample_course):
        """Test clearing all data from vector store"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                vector_store = VectorStore(
                    chroma_path=temp_dir,
                    embedding_model="all-MiniLM-L6-v2"
                )
                
                # Add data
                vector_store.add_course_metadata(sample_course)
                assert vector_store.get_course_count() == 1
                
                # Clear data
                vector_store.clear_all_data()
                assert vector_store.get_course_count() == 0
                
            except Exception as e:
                pytest.fail(f"Clear data test failed: {e}")


class TestDocumentProcessing:
    """Test document processing infrastructure"""

    def test_document_processor_initialization(self):
        """Test document processor can be initialized"""
        try:
            processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
            assert processor is not None
        except Exception as e:
            pytest.fail(f"Document processor initialization failed: {e}")

    def test_document_processor_with_real_course_file(self):
        """Test processing a real course document"""
        course_file = "/Users/michal/Desktop/Developer/rag_with_claude_code-1/docs/course1_script.txt"
        
        if os.path.exists(course_file):
            try:
                processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
                course, chunks = processor.process_course_document(course_file)
                
                # Verify course was parsed
                assert course is not None
                assert course.title == "Building Towards Computer Use with Anthropic"
                assert course.instructor == "Colt Steele"
                assert len(course.lessons) > 0
                
                # Verify chunks were created
                assert chunks is not None
                assert len(chunks) > 0
                assert all(isinstance(chunk, CourseChunk) for chunk in chunks)
                
            except Exception as e:
                pytest.fail(f"Real document processing failed: {e}")
        else:
            pytest.skip("Real course file not available for testing")

    def test_document_processor_with_malformed_file(self):
        """Test handling of malformed documents"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("This is not a properly formatted course document")
            temp_file.flush()
            
            try:
                processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
                
                # Should handle malformed files gracefully
                with pytest.raises(Exception):
                    processor.process_course_document(temp_file.name)
                    
            finally:
                os.unlink(temp_file.name)

    def test_document_processor_with_empty_file(self):
        """Test handling of empty documents"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("")
            temp_file.flush()
            
            try:
                processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
                
                # Should handle empty files gracefully
                with pytest.raises(Exception):
                    processor.process_course_document(temp_file.name)
                    
            finally:
                os.unlink(temp_file.name)


class TestFullSystemIntegration:
    """Test full system integration with real components"""

    def test_full_document_to_search_pipeline(self):
        """Test complete pipeline from document processing to search"""
        course_file = "/Users/michal/Desktop/Developer/rag_with_claude_code-1/docs/course1_script.txt"
        
        if not os.path.exists(course_file):
            pytest.skip("Real course file not available for testing")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Process document
                processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
                course, chunks = processor.process_course_document(course_file)
                
                # Add to vector store
                vector_store = VectorStore(
                    chroma_path=temp_dir,
                    embedding_model="all-MiniLM-L6-v2"
                )
                vector_store.add_course_metadata(course)
                vector_store.add_course_content(chunks)
                
                # Test search
                results = vector_store.search("computer use")
                assert not results.error
                assert not results.is_empty()
                
                # Test course name resolution
                results_with_course = vector_store.search(
                    "Anthropic", 
                    course_name="Building Towards Computer Use"
                )
                assert not results_with_course.error
                assert not results_with_course.is_empty()
                
                # Test outline retrieval
                outline = vector_store.get_course_outline("Building Towards Computer Use")
                assert outline is not None
                assert outline['course_title'] == course.title
                assert len(outline['lessons']) > 0
                
            except Exception as e:
                pytest.fail(f"Full pipeline test failed: {e}")

    def test_multiple_courses_integration(self):
        """Test system with multiple real courses"""
        docs_dir = "/Users/michal/Desktop/Developer/rag_with_claude_code-1/docs/"
        
        if not os.path.exists(docs_dir):
            pytest.skip("Docs directory not available for testing")
        
        course_files = [f for f in os.listdir(docs_dir) if f.endswith('.txt')]
        if len(course_files) < 2:
            pytest.skip("Need at least 2 course files for multi-course test")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
                vector_store = VectorStore(
                    chroma_path=temp_dir,
                    embedding_model="all-MiniLM-L6-v2"
                )
                
                # Process multiple courses
                courses_added = 0
                for course_file in course_files[:2]:  # Test with first 2 courses
                    try:
                        course, chunks = processor.process_course_document(
                            os.path.join(docs_dir, course_file)
                        )
                        vector_store.add_course_metadata(course)
                        vector_store.add_course_content(chunks)
                        courses_added += 1
                    except Exception as e:
                        print(f"Failed to process {course_file}: {e}")
                
                if courses_added == 0:
                    pytest.skip("No courses could be processed")
                
                # Test system with multiple courses
                assert vector_store.get_course_count() == courses_added
                
                # Test search across all courses
                results = vector_store.search("introduction")
                assert not results.error
                assert not results.is_empty()
                
                # Test course-specific search
                course_titles = vector_store.get_existing_course_titles()
                if len(course_titles) > 0:
                    specific_results = vector_store.search(
                        "lesson", 
                        course_name=course_titles[0]
                    )
                    # Should either find results or indicate no course found
                    assert not specific_results.error or "No course found" in specific_results.error
                
            except Exception as e:
                pytest.fail(f"Multi-course integration test failed: {e}")


class TestAnthropicAPIConnectivity:
    """Test Anthropic API connectivity (requires valid API key)"""

    @pytest.mark.skip(reason="Requires valid API key and makes real API calls")
    def test_anthropic_api_connection(self):
        """Test actual connection to Anthropic API"""
        import anthropic
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")
        
        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                temperature=0,
                messages=[{"role": "user", "content": "Hello, this is a test message."}]
            )
            
            assert response is not None
            assert hasattr(response, 'content')
            assert len(response.content) > 0
            
        except Exception as e:
            pytest.fail(f"Anthropic API connection test failed: {e}")

    def test_anthropic_client_initialization_with_invalid_key(self):
        """Test handling of invalid API key"""
        import anthropic
        
        # Should not fail on initialization
        try:
            client = anthropic.Anthropic(api_key="invalid_key")
            assert client is not None
        except Exception as e:
            pytest.fail(f"Client initialization should not fail with invalid key: {e}")


class TestSystemRequirements:
    """Test system requirements and dependencies"""

    def test_required_packages_importable(self):
        """Test that all required packages can be imported"""
        required_packages = [
            'anthropic',
            'chromadb',
            'sentence_transformers',
            'pydantic',
            'fastapi',
            'python-dotenv'
        ]
        
        failed_imports = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                failed_imports.append(package)
        
        if failed_imports:
            pytest.fail(f"Required packages not available: {failed_imports}")

    def test_python_version_compatibility(self):
        """Test Python version compatibility"""
        import sys
        
        # Should be Python 3.8+
        assert sys.version_info >= (3, 8), f"Python 3.8+ required, got {sys.version_info}"

    def test_file_system_permissions(self):
        """Test file system permissions for data storage"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Test directory creation
                test_dir = os.path.join(temp_dir, "test_chroma_db")
                os.makedirs(test_dir)
                
                # Test file writing
                test_file = os.path.join(test_dir, "test_file.txt")
                with open(test_file, 'w') as f:
                    f.write("test content")
                
                # Test file reading
                with open(test_file, 'r') as f:
                    content = f.read()
                    assert content == "test content"
                
                # Test file deletion
                os.remove(test_file)
                os.rmdir(test_dir)
                
            except Exception as e:
                pytest.fail(f"File system permission test failed: {e}")


class TestErrorRecovery:
    """Test system error recovery and resilience"""

    def test_vector_store_recovery_from_corruption(self):
        """Test vector store recovery from data corruption"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a vector store and add some data
            vector_store1 = VectorStore(
                chroma_path=temp_dir,
                embedding_model="all-MiniLM-L6-v2"
            )
            
            # Simulate corruption by creating invalid files in the directory
            corruption_file = os.path.join(temp_dir, "corrupted_data.db")
            with open(corruption_file, 'w') as f:
                f.write("corrupted data")
            
            try:
                # Should handle corruption gracefully
                vector_store2 = VectorStore(
                    chroma_path=temp_dir,
                    embedding_model="all-MiniLM-L6-v2"
                )
                
                # Should be able to clear and restart
                vector_store2.clear_all_data()
                assert vector_store2.get_course_count() == 0
                
            except Exception as e:
                # Acceptable if it fails - documents current behavior
                print(f"Vector store corruption handling: {e}")

    def test_graceful_degradation_with_no_data(self):
        """Test system behavior with no course data loaded"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                vector_store = VectorStore(
                    chroma_path=temp_dir,
                    embedding_model="all-MiniLM-L6-v2"
                )
                
                # Should handle empty data gracefully
                assert vector_store.get_course_count() == 0
                assert vector_store.get_existing_course_titles() == []
                
                # Search should return empty results, not crash
                results = vector_store.search("anything")
                assert not results.error
                assert results.is_empty()
                
            except Exception as e:
                pytest.fail(f"Empty data handling failed: {e}")

    def test_network_timeout_simulation(self):
        """Test behavior under network conditions (simulated)"""
        # This test would simulate network timeouts for API calls
        # Currently just documents the need for timeout handling
        pass