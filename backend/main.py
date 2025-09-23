"""
FastAPI backend for Chest X-ray AI Diagnostic POC
"""

import os
import time
import logging
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from PIL import Image
import io
import numpy as np

from models.chest_xray import ChestXRayModel
from utils.image_processing import preprocess_image
from schemas.api_models import AnalysisResponse, HealthResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Chest X-ray AI Diagnostic API",
    description="AI-powered chest X-ray pathology detection using TorchXRayVision",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global model
    try:
        logger.info("Loading chest X-ray model...")
        model = ChestXRayModel()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Chest X-ray AI Diagnostic API is running",
        model_loaded=model is not None
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        message="System status check",
        model_loaded=model is not None,
        gpu_available=gpu_available,
        gpu_count=gpu_count
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_xray(file: UploadFile = File(...)):
    """
    Analyze chest X-ray image for pathology detection
    
    Args:
        file: Uploaded image file (JPEG, PNG, or DICOM)
        
    Returns:
        Analysis results with pathology predictions and confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    allowed_types = {'image/jpeg', 'image/png', 'image/jpg', 'application/dicom'}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file.content_type}. Supported types: JPEG, PNG, DICOM"
        )
    
    # Validate file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    file_size = 0
    
    try:
        start_time = time.time()
        
        # Read and validate image
        contents = await file.read()
        file_size = len(contents)
        
        if file_size > max_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large: {file_size/1024/1024:.1f}MB. Maximum allowed: 10MB"
            )
        
        logger.info(f"Processing file: {file.filename}, size: {file_size/1024:.1f}KB")
        
        # Process image
        try:
            image = Image.open(io.BytesIO(contents))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Run inference
        predictions = model.predict(processed_image)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Get top prediction
        top_pathology = max(predictions['pathologies'], key=predictions['pathologies'].get)
        top_confidence = predictions['pathologies'][top_pathology]
        
        # Generate recommendations
        recommendations = generate_recommendations(predictions['pathologies'])
        
        logger.info(f"Analysis completed in {processing_time:.2f}s. Top prediction: {top_pathology} ({top_confidence:.2f})")
        
        return AnalysisResponse(
            status="success",
            processing_time=round(processing_time, 2),
            findings=predictions['pathologies'],
            top_prediction=top_pathology,
            confidence=round(top_confidence, 3),
            recommendations=recommendations,
            image_info={
                "filename": file.filename,
                "size_kb": round(file_size / 1024, 1),
                "dimensions": f"{image.width}x{image.height}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def generate_recommendations(pathologies: Dict[str, float]) -> str:
    """Generate clinical recommendations based on predictions"""
    
    # Sort pathologies by confidence
    sorted_pathologies = sorted(pathologies.items(), key=lambda x: x[1], reverse=True)
    top_pathology, top_confidence = sorted_pathologies[0]
    
    if top_pathology.lower() == 'no finding' or top_confidence < 0.3:
        return "No significant pathology detected. Routine follow-up as clinically indicated."
    
    elif top_confidence > 0.7:
        if top_pathology.lower() in ['pneumonia', 'consolidation']:
            return f"High probability of {top_pathology.lower()} detected. Recommend immediate clinical correlation and consider antibiotic therapy if clinically appropriate."
        elif top_pathology.lower() == 'pneumothorax':
            return f"Possible {top_pathology.lower()} detected. Urgent clinical evaluation recommended."
        elif top_pathology.lower() in ['pleural effusion', 'cardiomegaly']:
            return f"{top_pathology.title()} detected. Recommend clinical correlation and consider further cardiac evaluation if indicated."
        else:
            return f"{top_pathology.title()} detected with high confidence. Recommend clinical correlation and appropriate follow-up."
    
    elif top_confidence > 0.5:
        return f"Moderate probability of {top_pathology.lower()} detected. Recommend clinical correlation and consider additional imaging if clinically indicated."
    
    else:
        return f"Low-moderate probability findings detected. Clinical correlation recommended. Consider repeat imaging if symptoms persist."

@app.get("/pathologies")
async def get_supported_pathologies():
    """Get list of supported pathologies"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "pathologies": model.get_pathology_labels(),
        "total_count": len(model.get_pathology_labels())
    }

@app.post("/batch-analyze")
async def batch_analyze(files: list[UploadFile] = File(...)):
    """Analyze multiple X-ray images (max 5 at once)"""
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 files allowed per batch")
    
    results = []
    for file in files:
        try:
            result = await analyze_xray(file)
            results.append({"filename": file.filename, "result": result})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    
    return {"batch_results": results, "processed_count": len(results)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
