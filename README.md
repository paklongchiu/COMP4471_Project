# Object Detection & Background Removal

A web application for object detection and background removal using Faster R-CNN and Mask R-CNN models.

## Setup Instructions

### Backend Setup

1. Install Python dependencies:
```bash
pip install fastapi uvicorn torch torchvision pillow numpy python-multipart
```

2. Start the backend server:
```bash
python backend.py
```

The backend will run on `http://localhost:3000`

### Frontend Setup

1. Install Node.js dependencies:
```bash
npm install
```

2. Start the frontend development server:
```bash
npm run dev
```

The frontend will run on `http://localhost:8000` (or the port shown in the terminal)

## Usage

1. Make sure both backend and frontend servers are running
2. Open your browser and navigate to the frontend URL (usually `http://localhost:8000`)
3. Upload an image
4. The app will automatically detect objects in the image
5. Select the objects you want to keep
6. Click "Remove Background" to segment and remove the background
7. Download the processed image

## Features

- **Object Detection**: Uses Faster R-CNN to detect objects in images
- **Background Removal**: Uses Mask R-CNN to segment selected objects and remove background
- **Interactive UI**: Select/deselect objects, view bounding boxes, and download results
- **Simulation Mode**: Test the UI without backend (toggle available in the app)

## Project Structure

```
├── backend.py          # FastAPI backend server
├── frontend.js         # Original React component (now in src/App.jsx)
├── src/
│   ├── App.jsx         # Main React component
│   ├── main.jsx        # React entry point
│   └── index.css       # Global styles
├── index.html          # HTML template
├── package.json        # Node.js dependencies
└── vite.config.js      # Vite configuration

