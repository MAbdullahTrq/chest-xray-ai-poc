"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, Any
from datetime import datetime

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="System status")
    message: str = Field(..., description="Status message")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    gpu_available: Optional[bool] = Field(None, description="GPU availability")
    gpu_count: Optional[int] = Field(None, description="Number of available GPUs")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class AnalysisResponse(BaseModel):
    """X-ray analysis response model"""
    status: str = Field(..., description="Analysis status")
    processing_time: float = Field(..., description="Processing time in seconds")
    findings: Dict[str, float] = Field(..., description="Pathology predictions with confidence scores")
    top_prediction: str = Field(..., description="Most likely pathology")
    confidence: float = Field(..., description="Confidence score for top prediction")
    recommendations: str = Field(..., description="Clinical recommendations")
    image_info: Optional[Dict[str, Any]] = Field(None, description="Information about the processed image")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")

class ErrorResponse(BaseModel):
    """Error response model"""
    status: str = Field(default="error", description="Error status")
    message: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Specific error code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

class BatchAnalysisRequest(BaseModel):
    """Batch analysis request model"""
    files: list = Field(..., description="List of image files to analyze")
    options: Optional[Dict[str, Any]] = Field(None, description="Analysis options")

class BatchAnalysisResponse(BaseModel):
    """Batch analysis response model"""
    batch_results: list = Field(..., description="List of analysis results")
    processed_count: int = Field(..., description="Number of successfully processed images")
    failed_count: int = Field(default=0, description="Number of failed analyses")
    total_processing_time: Optional[float] = Field(None, description="Total processing time")
    timestamp: datetime = Field(default_factory=datetime.now, description="Batch analysis timestamp")

class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str = Field(..., description="Name of the loaded model")
    pathologies: list[str] = Field(..., description="List of detectable pathologies")
    total_pathologies: int = Field(..., description="Total number of pathologies")
    device: str = Field(..., description="Device used for inference")
    model_parameters: Optional[int] = Field(None, description="Number of model parameters")

class AnalysisOptions(BaseModel):
    """Analysis options model"""
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    enhance_contrast: bool = Field(default=True, description="Whether to apply contrast enhancement")
    target_size: tuple[int, int] = Field(default=(224, 224), description="Target image size for processing")
    return_raw_outputs: bool = Field(default=False, description="Whether to include raw model outputs")

# Example usage in API documentation
class ExampleResponses:
    """Example responses for API documentation"""
    
    ANALYSIS_SUCCESS = {
        "example": {
            "status": "success",
            "processing_time": 2.3,
            "findings": {
                "Pneumonia": 0.85,
                "Pneumothorax": 0.12,
                "Pleural Effusion": 0.08,
                "Cardiomegaly": 0.05,
                "No Finding": 0.15
            },
            "top_prediction": "Pneumonia",
            "confidence": 0.85,
            "recommendations": "High probability of pneumonia detected. Recommend immediate clinical correlation and consider antibiotic therapy if clinically appropriate.",
            "image_info": {
                "filename": "chest_xray.jpg",
                "size_kb": 245.6,
                "dimensions": "1024x1024"
            },
            "timestamp": "2024-01-15T10:30:00Z"
        }
    }
    
    HEALTH_CHECK = {
        "example": {
            "status": "healthy",
            "message": "Chest X-ray AI Diagnostic API is running",
            "model_loaded": True,
            "gpu_available": True,
            "gpu_count": 1,
            "timestamp": "2024-01-15T10:30:00Z"
        }
    }
    
    ERROR_RESPONSE = {
        "example": {
            "status": "error",
            "message": "Unsupported file type: image/gif. Supported types: JPEG, PNG, DICOM",
            "error_code": "INVALID_FILE_TYPE",
            "timestamp": "2024-01-15T10:30:00Z"
        }
    }
