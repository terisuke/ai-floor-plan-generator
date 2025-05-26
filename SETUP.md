# AI Floor Plan Generator - Setup Instructions

## Prerequisites

- Node.js 18+ and npm
- Python 3.9+
- M2/M4 Max MacBook Pro (for MPS acceleration) or CUDA-capable GPU

## Setup Steps

### 1. Install Frontend Dependencies

```bash
npm install
```

### 2. Install Backend Dependencies

```bash
cd backend
npm install
cd ..
```

### 3. Install Python Dependencies

```bash
cd inference
pip3 install -r requirements.txt
cd ..
```

Note: On first run, the Stable Diffusion model will be downloaded (~5GB). This is a one-time download.

### 4. Start the Backend Server

```bash
cd backend
npm run dev
```

The backend will run on http://localhost:3001

### 5. Start the Frontend Development Server

In a new terminal:

```bash
npm run dev
```

The frontend will run on http://localhost:5173

## Usage

1. Open http://localhost:5173 in your browser
2. Enter grid dimensions (1-50 for both width and height)
3. Click "Generate Floor Plan"
4. Wait for the AI to generate the image (first run will be slower due to model loading)

## Architecture Overview

```
Frontend (React) → Backend API (Express) → Python Script (Stable Diffusion) → Generated Image
```

- Frontend sends grid dimensions to backend API
- Backend spawns Python process with parameters
- Python script uses Stable Diffusion to generate floor plan
- Generated image is saved and URL returned to frontend
- Frontend displays the generated image

## Troubleshooting

- If you get Python import errors, ensure you've installed all requirements
- If generation is slow, check that MPS (Mac) or CUDA (NVIDIA) is being used
- Generated images are saved in the `generated/` directory
- Check backend console for detailed logs

## Next Steps

- Fine-tune with LoRA using the dataset in `/dataset`
- Implement prompt engineering for better floor plan generation
- Add more parameters (room types, style preferences, etc.)