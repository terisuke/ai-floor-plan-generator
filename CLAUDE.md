# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Frontend
```bash
npm run dev      # Start frontend dev server (runs on http://localhost:5173)
npm run build    # Build frontend for production
npm run preview  # Preview production build
```

### Backend
```bash
cd backend
npm run dev      # Start backend dev server (runs on http://localhost:3001)
npm run build    # Build backend TypeScript
npm run start    # Run built backend
```

### Python Inference
```bash
cd inference
pip3 install -r requirements.txt  # Install Python dependencies
python3 generate_floor_plan.py --width 10 --height 10 --output test.png  # Test script
```

## Architecture Overview

Full-stack AI floor plan generation application with three main components:

### Frontend (React + TypeScript + Vite)
1. **App.tsx** - Main orchestrator managing global state (dimensions, imageUrl, loading, error)
2. **GridInputForm** - Collects width/height input (1-50 range) and triggers generation
3. **floorPlanService** - Calls backend API at http://localhost:3001
4. **ImageDisplay** - Renders generated images with loading/error states

### Backend (Node.js + Express + TypeScript)
- **POST /api/generate-floor-plan** - Receives dimensions, spawns Python process, returns image URL
- Serves generated images from `/generated` directory
- Handles process management and error handling

### AI Inference (Python + Stable Diffusion)
- Uses Hugging Face Diffusers library
- Supports MPS (Mac M2/M4) and CUDA acceleration
- Generates floor plans based on architectural prompts
- Saves images to backend's generated directory

### Key Implementation Details
- Grid dimensions are scaled by 40px per unit (e.g., 10x10 grid = 400x400px image)
- Maximum image size: 1200x1000px
- Dataset folder contains ~100 floor plan JSON/PNG pairs for future LoRA training
- First run downloads Stable Diffusion model (~5GB)

### Environment Configuration
- Frontend uses Vite with TypeScript strict mode and "@/" path alias
- Backend uses Express with CORS enabled
- Python script uses MPS on Mac for GPU acceleration
- Tailwind CSS for frontend styling

### Current Status
MVP with functional end-to-end pipeline. Ready for LoRA fine-tuning using the dataset to improve floor plan generation quality.