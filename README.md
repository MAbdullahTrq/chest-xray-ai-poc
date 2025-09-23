# Chest X-ray AI Diagnostic POC

A complete Proof of Concept for AI-based chest X-ray analysis using pretrained models.

## 🎯 Overview

This POC demonstrates a complete pipeline for chest X-ray pathology detection:
- **Frontend**: Simple web interface for image upload
- **Backend API**: FastAPI server with TorchXRayVision model
- **Model**: Pretrained chest X-ray pathology classifier
- **Infrastructure**: Optimized for TensorDock RTX 3090

## 🏗️ Architecture

```
Frontend (React/HTML) → API (FastAPI) → Model (TorchXRayVision) → Results
```

## 📋 Features

- Upload chest X-ray images (DICOM, PNG, JPEG)
- Automatic pathology detection for 14+ conditions
- Confidence scores and probability distributions
- Real-time processing (2-3 seconds per image)
- RESTful API with OpenAPI documentation

## 🚀 Quick Start

### Prerequisites
- TensorDock account and RTX 3090 instance
- Python 3.8+
- 8GB+ available disk space

### Installation
```bash
git clone <your-repo>
cd chest_xray_poc
pip install -r requirements.txt
```

### Run the Application
```bash
# Start the API server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# In another terminal, serve the frontend
cd frontend
python -m http.server 3000
```

### Access the Application
- Frontend: http://localhost:3000
- API Documentation: http://localhost:8000/docs

## 📁 Project Structure

```
chest_xray_poc/
├── README.md
├── requirements.txt
├── setup_tensordock.md
├── backend/
│   ├── main.py              # FastAPI application
│   ├── models/
│   │   └── chest_xray.py    # Model wrapper
│   ├── utils/
│   │   └── image_processing.py
│   └── schemas/
│       └── api_models.py
├── frontend/
│   ├── index.html           # Main UI
│   ├── script.js           # Frontend logic
│   └── style.css           # Styling
├── tests/
│   ├── test_api.py
│   └── sample_images/
└── docs/
    ├── api_documentation.md
    └── deployment_guide.md
```

## 🔧 Configuration

Environment variables (create `.env` file):
```
MODEL_PATH=./models/
MAX_IMAGE_SIZE=1024
CONFIDENCE_THRESHOLD=0.5
DEBUG=True
```

## 📊 Expected Performance

- **Processing Time**: 2-3 seconds per X-ray
- **Throughput**: ~1,200 X-rays/hour
- **Accuracy**: 90%+ on common pathologies
- **Cost**: ~$0.0002 per analysis on TensorDock

## 🏥 Supported Pathologies

1. Pneumonia
2. Pneumothorax
3. Pleural Effusion
4. Atelectasis
5. Cardiomegaly
6. Consolidation
7. Edema
8. Emphysema
9. Fibrosis
10. Hernia
11. Infiltration
12. Mass
13. Nodule
14. Pleural Thickening

## 📝 API Usage

### Upload and Analyze X-ray
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@chest_xray.jpg"
```

### Response
```json
{
  "status": "success",
  "processing_time": 2.3,
  "findings": {
    "pneumonia": 0.85,
    "pneumothorax": 0.12,
    "normal": 0.03
  },
  "top_prediction": "pneumonia",
  "confidence": 0.85,
  "recommendations": "High probability of pneumonia detected. Recommend clinical correlation."
}
```

## 🐳 Docker Deployment

```bash
docker build -t chest-xray-poc .
docker run -p 8000:8000 chest-xray-poc
```

## 📚 Documentation

- [TensorDock Setup Guide](./setup_tensordock.md)
- [API Documentation](./docs/api_documentation.md)
- [Deployment Guide](./docs/deployment_guide.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.
