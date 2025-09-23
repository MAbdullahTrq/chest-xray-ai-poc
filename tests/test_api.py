"""
Test suite for Chest X-ray AI API
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from PIL import Image
import io
import numpy as np

# Import the FastAPI app
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.main import app

client = TestClient(app)

class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "message" in data
    
    def test_health_endpoint(self):
        """Test detailed health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "model_loaded" in data
        assert "gpu_available" in data

class TestAnalysisEndpoint:
    """Test X-ray analysis functionality"""
    
    def create_test_image(self, format="JPEG", size=(224, 224)):
        """Create a test image for testing"""
        # Create a simple test image (simulating chest X-ray)
        image_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        # Make it look more like an X-ray (darker with bright spots)
        image_array = image_array * 0.3 + np.random.randint(0, 100, (*size, 3))
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        
        image = Image.fromarray(image_array)
        img_buffer = io.BytesIO()
        image.save(img_buffer, format=format)
        img_buffer.seek(0)
        return img_buffer
    
    def test_analyze_valid_image(self):
        """Test analysis with valid JPEG image"""
        test_image = self.create_test_image("JPEG")
        
        response = client.post(
            "/analyze",
            files={"file": ("test_xray.jpg", test_image, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert data["status"] == "success"
        assert "processing_time" in data
        assert "findings" in data
        assert "top_prediction" in data
        assert "confidence" in data
        assert "recommendations" in data
        
        # Check findings structure
        findings = data["findings"]
        assert isinstance(findings, dict)
        assert len(findings) > 0
        
        # Check confidence is between 0 and 1
        assert 0 <= data["confidence"] <= 1
    
    def test_analyze_png_image(self):
        """Test analysis with PNG image"""
        test_image = self.create_test_image("PNG")
        
        response = client.post(
            "/analyze",
            files={"file": ("test_xray.png", test_image, "image/png")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    def test_analyze_invalid_file_type(self):
        """Test analysis with invalid file type"""
        # Create a text file
        text_content = io.BytesIO(b"This is not an image")
        
        response = client.post(
            "/analyze",
            files={"file": ("test.txt", text_content, "text/plain")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Unsupported file type" in data["detail"]
    
    def test_analyze_large_file(self):
        """Test analysis with file too large"""
        # Create a large test image (simulate >10MB)
        large_image = self.create_test_image("JPEG", size=(4000, 4000))
        
        response = client.post(
            "/analyze",
            files={"file": ("large_xray.jpg", large_image, "image/jpeg")}
        )
        
        # This might pass or fail depending on actual file size after compression
        # The test is mainly to ensure the endpoint handles large files gracefully
        assert response.status_code in [200, 413]  # Success or Payload Too Large
    
    def test_analyze_no_file(self):
        """Test analysis without file"""
        response = client.post("/analyze")
        
        assert response.status_code == 422  # Validation error
    
    def test_analyze_corrupted_image(self):
        """Test analysis with corrupted image data"""
        corrupted_data = io.BytesIO(b"fake image data that is not valid")
        
        response = client.post(
            "/analyze",
            files={"file": ("corrupted.jpg", corrupted_data, "image/jpeg")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid image format" in data["detail"]

class TestPathologiesEndpoint:
    """Test pathologies endpoint"""
    
    def test_get_pathologies(self):
        """Test getting supported pathologies"""
        response = client.get("/pathologies")
        
        if response.status_code == 503:
            # Model not loaded, skip test
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "pathologies" in data
        assert "total_count" in data
        assert isinstance(data["pathologies"], list)
        assert len(data["pathologies"]) > 0
        assert data["total_count"] == len(data["pathologies"])

class TestBatchAnalysis:
    """Test batch analysis functionality"""
    
    def create_test_image(self, format="JPEG", size=(224, 224)):
        """Create a test image for testing"""
        image_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        img_buffer = io.BytesIO()
        image.save(img_buffer, format=format)
        img_buffer.seek(0)
        return img_buffer
    
    def test_batch_analyze_multiple_images(self):
        """Test batch analysis with multiple images"""
        image1 = self.create_test_image("JPEG")
        image2 = self.create_test_image("PNG")
        
        files = [
            ("files", ("test1.jpg", image1, "image/jpeg")),
            ("files", ("test2.png", image2, "image/png"))
        ]
        
        response = client.post("/batch-analyze", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "batch_results" in data
        assert "processed_count" in data
        assert data["processed_count"] == 2
        assert len(data["batch_results"]) == 2
    
    def test_batch_analyze_too_many_files(self):
        """Test batch analysis with too many files (>5)"""
        files = []
        for i in range(6):  # Try to upload 6 files (limit is 5)
            image = self.create_test_image("JPEG")
            files.append(("files", (f"test{i}.jpg", image, "image/jpeg")))
        
        response = client.post("/batch-analyze", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "Maximum 5 files allowed" in data["detail"]

class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_nonexistent_endpoint(self):
        """Test accessing non-existent endpoint"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_wrong_method(self):
        """Test using wrong HTTP method"""
        response = client.get("/analyze")  # Should be POST
        assert response.status_code == 405  # Method Not Allowed

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment"""
    # This runs once before all tests
    print("Setting up test environment...")
    
    # You can add any setup code here
    # For example, ensuring the model is loaded, setting up test data, etc.
    
    yield
    
    # Cleanup after all tests
    print("Cleaning up test environment...")

# Performance tests
class TestPerformance:
    """Test API performance"""
    
    def create_test_image(self):
        """Create a test image"""
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG")
        img_buffer.seek(0)
        return img_buffer
    
    def test_analysis_performance(self):
        """Test that analysis completes within reasonable time"""
        test_image = self.create_test_image()
        
        import time
        start_time = time.time()
        
        response = client.post(
            "/analyze",
            files={"file": ("test_xray.jpg", test_image, "image/jpeg")}
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert response.status_code == 200
        # Analysis should complete within 10 seconds (generous limit for CI/CD)
        assert processing_time < 10.0
        
        # Check that reported processing time is reasonable
        data = response.json()
        reported_time = data.get("processing_time", 0)
        assert reported_time > 0
        assert reported_time < 10.0

# Integration tests
class TestIntegration:
    """Integration tests"""
    
    def test_full_workflow(self):
        """Test complete workflow from upload to results"""
        # 1. Check health
        health_response = client.get("/health")
        if health_response.status_code != 200:
            pytest.skip("API not healthy")
        
        # 2. Get pathologies
        pathologies_response = client.get("/pathologies")
        if pathologies_response.status_code == 503:
            pytest.skip("Model not loaded")
        
        # 3. Analyze image
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG")
        img_buffer.seek(0)
        
        analyze_response = client.post(
            "/analyze",
            files={"file": ("test_xray.jpg", img_buffer, "image/jpeg")}
        )
        
        assert analyze_response.status_code == 200
        
        # 4. Validate complete response
        data = analyze_response.json()
        pathologies_data = pathologies_response.json()
        
        # Check that findings contain known pathologies
        supported_pathologies = set(pathologies_data["pathologies"])
        found_pathologies = set(data["findings"].keys())
        
        # At least some overlap expected
        assert len(found_pathologies.intersection(supported_pathologies)) > 0

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
