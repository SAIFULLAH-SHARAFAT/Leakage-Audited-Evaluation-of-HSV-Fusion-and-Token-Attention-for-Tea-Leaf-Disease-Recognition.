# Tea Leaf Disease Classification

## Overview

### Backend (FastAPI)

FastAPI server with Swin Transformer model for tea leaf disease classification. Includes explainability methods (Grad-CAM, Grad-CAM++, LayerCAM, SHAP) to visualize model predictions. Quality checks detect poor images and an agreement score measures consistency across explanation methods.

### Frontend (React Native/Expo)

Mobile app for capturing leaf images and displaying disease predictions with visual explanations.

---

## Getting Started

### 1. Backend Setup (One-time)

```bash
cd backend
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

> Weight file: place `ema_model_weights_only.pth` in `backend/` before running.

### 2. Backend Run (Start in one terminal)

```bash
cd backend
source env/bin/activate
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
Loading model...
Model loaded successfully with strict matching.
Target layer set: LayerNorm((768,), eps=1e-05, elementwise_affine=True)

Uvicorn running on http://0.0.0.0:8000
Press CTRL+C to quit
```

Health check: http://localhost:8000/health

---

## Frontend Setup

### Running on Simulator

No configuration needed. The app will connect to `localhost:8000`.

### Running on Physical Device

Update the API URL to use your machine's IP address instead of localhost.

Find your IP:
```bash
ifconfig | grep inet
```

Update frontend config:

Edit [frontend/constants/api.ts](frontend/constants/api.ts):
```typescript
export const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || 'http://192.168.X.X:8000';
```

Replace `192.168.X.X` with your actual IP.

Run the app:
```bash
cd frontend
npm start
```

---

## API Reference

### POST `/predict`

**Request:**
```
Content-Type: multipart/form-data
Body: { file: <image_file> }
```

**Response:**
```json
{
  "prediction": {
    "label": "Brown Blight",
    "confidence": 0.92
  },
  "agreement_score": 0.41,
  "quality_flags": {
    "background": false,
    "spread": false,
    "border": false
  },
  "explanations": {
    "gradcam": "data:image/png;base64,...",
    "gradcampp": "data:image/png;base64,...",
    "layercam": "data:image/png;base64,...",
    "shap": "data:image/png;base64,..."
  },
  "final_status": "accepted"
}
```

Status values:
- `accepted`: Prediction is reliable
- `retake_required`: Image quality insufficient

### GET `/health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes": 7
}
```

---

## Disease Classes

The model classifies into 7 categories:
1. Brown Blight
2. Gray Blight
3. Green mirid bug
4. Healthy leaf
5. Helopeltis
6. Red spider
7. Tea algal leaf spot

---

## Quality Checks

| Check | Threshold | Description |
|-------|-----------|-------------|
| **Background Focus** | < 55% center energy | Model focused on leaf, not soil/hand |
| **Spread Detection** | > 8 components OR largest < 40% | Attention is not scattered |
| **Border Concentration** | > 35% border energy | Spurious edge effects ruled out |
| **Agreement Score** | ≥ 0.35 | XAI methods agree on same region |

A quality check fails when 2 or more methods detect an issue.

---

## Project Structure

```
├── backend/
│   ├── main.py              # FastAPI server + /predict endpoint
│   ├── logic.py             # XAI generation, agreement, quality checks
│   ├── model.py             # SwinRGBHSV model wrapper
│   ├── ema_model_weights_only.pth
│   └── requirements.txt
├── frontend/
│   ├── hooks/usePredict.ts
│   ├── constants/api.ts
│   └── ...
└── README.md
```

---

## Features

- Disease classification with confidence scores
- Visual explanations using 4 different XAI methods
- Agreement scoring across explanation methods
- Automatic image quality validation
- Retake prompts for unclear images  

---

## How It Works

1. User captures leaf image on mobile device
2. Image sent to backend via POST request
3. Backend preprocesses and runs inference with Swin Transformer
4. Four XAI methods generate explanation heatmaps
5. Agreement score computed from heatmap overlap
6. Quality checks validate image suitability
7. Results returned with visual overlays
8. Frontend displays prediction or retake prompt

---

## Troubleshooting

If issues occur, check:
- Backend is running on port 8000
- Health endpoint responds: `curl http://localhost:8000/health`
- Model weights file exists in backend folder
- Frontend API URL matches backend address

**Status**: Ready for testing with real backend
