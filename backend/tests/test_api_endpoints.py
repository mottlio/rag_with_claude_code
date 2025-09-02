import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from models import Source


class TestQueryEndpoint:
    """Test /api/query endpoint functionality"""

    @pytest.mark.api
    def test_query_with_session_id(self, test_client, api_test_data):
        """Test query endpoint with provided session ID"""
        response = test_client.post("/api/query", json=api_test_data["valid_query"])
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test_session_123"
        assert data["answer"] == "This is a test response"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["display"] == "Test Course - Lesson 1"

    @pytest.mark.api
    def test_query_without_session_id(self, test_client, api_test_data):
        """Test query endpoint without session ID (should create one)"""
        response = test_client.post("/api/query", json=api_test_data["query_without_session"])
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test_session_id"  # From mock

    @pytest.mark.api
    def test_query_with_empty_query(self, test_client, api_test_data):
        """Test query endpoint with empty query"""
        response = test_client.post("/api/query", json=api_test_data["empty_query"])
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

    @pytest.mark.api
    def test_query_with_long_query(self, test_client, api_test_data):
        """Test query endpoint with very long query"""
        response = test_client.post("/api/query", json=api_test_data["long_query"])
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

    @pytest.mark.api
    def test_query_invalid_json(self, test_client):
        """Test query endpoint with invalid JSON"""
        response = test_client.post("/api/query", json={"invalid": "field"})
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.api
    def test_query_missing_required_field(self, test_client):
        """Test query endpoint with missing required query field"""
        response = test_client.post("/api/query", json={"session_id": "test"})
        
        assert response.status_code == 422  # Validation error

    @pytest.mark.api
    def test_query_endpoint_exception_handling(self, test_app_factory):
        """Test query endpoint exception handling"""
        mock_rag = Mock()
        mock_rag.session_manager.create_session.return_value = "test_session"
        mock_rag.query.side_effect = Exception("Test error")
        
        app = test_app_factory(mock_rag)
        client = TestClient(app)
        
        response = client.post("/api/query", json={"query": "test"})
        
        assert response.status_code == 500
        assert "Test error" in response.json()["detail"]

    @pytest.mark.api
    def test_query_content_type_validation(self, test_client):
        """Test query endpoint with wrong content type"""
        response = test_client.post("/api/query", data="not json")
        
        assert response.status_code == 422


class TestCoursesEndpoint:
    """Test /api/courses endpoint functionality"""

    @pytest.mark.api
    def test_get_courses_success(self, test_client):
        """Test successful course statistics retrieval"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 2
        assert data["course_titles"] == ["Test Course 1", "Test Course 2"]

    @pytest.mark.api
    def test_get_courses_exception_handling(self, test_app_factory):
        """Test course endpoint exception handling"""
        mock_rag = Mock()
        mock_rag.get_course_analytics.side_effect = Exception("Analytics error")
        
        app = test_app_factory(mock_rag)
        client = TestClient(app)
        
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Analytics error" in response.json()["detail"]

    @pytest.mark.api
    def test_get_courses_empty_data(self, test_app_factory):
        """Test course endpoint with no courses"""
        mock_rag = Mock()
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        app = test_app_factory(mock_rag)
        client = TestClient(app)
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []


class TestSessionEndpoint:
    """Test session management endpoints"""

    @pytest.mark.api
    def test_clear_session_success(self, test_client):
        """Test successful session clearing"""
        response = test_client.delete("/api/sessions/test_session_123/clear")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "test_session_123" in data["message"]

    @pytest.mark.api
    def test_clear_session_exception_handling(self, test_app_factory):
        """Test session clearing exception handling"""
        mock_rag = Mock()
        mock_rag.session_manager.clear_session.side_effect = Exception("Session error")
        
        app = test_app_factory(mock_rag)
        client = TestClient(app)
        
        response = client.delete("/api/sessions/test_session/clear")
        
        assert response.status_code == 500
        assert "Session error" in response.json()["detail"]

    @pytest.mark.api
    def test_clear_session_with_special_characters(self, test_client):
        """Test session clearing with special characters in session ID"""
        import urllib.parse
        session_id = "test-session_123!@#"
        encoded_session_id = urllib.parse.quote(session_id, safe='')
        response = test_client.delete(f"/api/sessions/{encoded_session_id}/clear")
        
        assert response.status_code == 200


class TestRootEndpoint:
    """Test root endpoint"""

    @pytest.mark.api
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns expected message"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert data["message"] == "Course Materials RAG System API"


class TestCORSHandling:
    """Test CORS middleware functionality"""

    @pytest.mark.api
    def test_cors_preflight_request(self, test_client):
        """Test CORS preflight handling"""
        response = test_client.options("/api/query", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        })
        
        # Should not return 405 Method Not Allowed
        assert response.status_code != 405

    @pytest.mark.api
    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are present in responses"""
        response = test_client.get("/api/courses", headers={
            "Origin": "http://localhost:3000"
        })
        
        assert response.status_code == 200
        # CORS middleware should add headers
        assert "access-control-allow-origin" in [h.lower() for h in response.headers.keys()]


class TestRequestValidation:
    """Test request validation and error handling"""

    @pytest.mark.api
    def test_invalid_http_methods(self, test_client):
        """Test invalid HTTP methods on endpoints"""
        # GET on POST endpoint
        response = test_client.get("/api/query")
        assert response.status_code == 405
        
        # POST on GET endpoint  
        response = test_client.post("/api/courses")
        assert response.status_code == 405

    @pytest.mark.api
    def test_nonexistent_endpoints(self, test_client):
        """Test requests to nonexistent endpoints"""
        response = test_client.get("/api/nonexistent")
        assert response.status_code == 404
        
        response = test_client.post("/api/invalid")
        assert response.status_code == 404

    @pytest.mark.api
    def test_malformed_json_payload(self, test_client):
        """Test malformed JSON in request body"""
        response = test_client.post("/api/query", 
                                   data="{'invalid': json}",
                                   headers={"Content-Type": "application/json"})
        
        assert response.status_code == 422

    @pytest.mark.api
    def test_large_payload_handling(self, test_client):
        """Test handling of very large payloads"""
        large_query = "A" * 100000  # 100KB query
        response = test_client.post("/api/query", json={"query": large_query})
        
        # Should handle large payloads gracefully
        assert response.status_code in [200, 413, 422]


class TestEndpointIntegration:
    """Test integration between different endpoints"""

    @pytest.mark.api
    def test_session_workflow(self, test_client):
        """Test complete session workflow"""
        # Create session via query
        query_response = test_client.post("/api/query", json={
            "query": "Test query"
        })
        
        assert query_response.status_code == 200
        session_id = query_response.json()["session_id"]
        
        # Use session in another query
        query_response2 = test_client.post("/api/query", json={
            "query": "Follow-up query",
            "session_id": session_id
        })
        
        assert query_response2.status_code == 200
        assert query_response2.json()["session_id"] == session_id
        
        # Clear the session
        clear_response = test_client.delete(f"/api/sessions/{session_id}/clear")
        
        assert clear_response.status_code == 200

    @pytest.mark.api 
    def test_concurrent_requests(self, test_client):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = test_client.post("/api/query", json={"query": "concurrent test"})
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5


class TestResponseFormat:
    """Test response format consistency"""

    @pytest.mark.api
    def test_query_response_format(self, test_client):
        """Test that query responses have consistent format"""
        response = test_client.post("/api/query", json={"query": "test"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Required fields
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data
        
        # Correct types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Source structure
        if data["sources"]:
            source = data["sources"][0]
            assert "display" in source
            assert "link" in source

    @pytest.mark.api
    def test_courses_response_format(self, test_client):
        """Test that courses responses have consistent format"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Required fields
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data
        
        # Correct types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] >= 0

    @pytest.mark.api
    def test_error_response_format(self, test_app_factory):
        """Test that error responses have consistent format"""
        mock_rag = Mock()
        mock_rag.query.side_effect = Exception("Test error")
        mock_rag.session_manager.create_session.return_value = "test_session"
        
        app = test_app_factory(mock_rag)
        client = TestClient(app)
        
        response = client.post("/api/query", json={"query": "test"})
        
        assert response.status_code == 500
        data = response.json()
        
        # Error format
        assert "detail" in data
        assert isinstance(data["detail"], str)