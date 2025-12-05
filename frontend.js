import React, { useState, useRef, useEffect } from 'react';
import { Upload, Download, Loader, X, CheckSquare, Square } from 'lucide-react';

const API_BASE_URL = 'http://localhost:3000';

export default function ObjectSegmentationUI() {
  const [image, setImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [detectedObjects, setDetectedObjects] = useState([]);
  const [selectedObjects, setSelectedObjects] = useState(new Set());
  const [processedImage, setProcessedImage] = useState(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [error, setError] = useState(null);
  const [useSimulation, setUseSimulation] = useState(true); // Toggle for testing
  
  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  // Simulate Faster R-CNN detection (for testing without backend)
  const simulateDetection = (img) => {
    return new Promise((resolve) => {
      setTimeout(() => {
        const objects = [
          { id: 1, label: 'person', confidence: 0.95, bbox: [0.2, 0.15, 0.5, 0.8] },
          { id: 2, label: 'car', confidence: 0.89, bbox: [0.55, 0.4, 0.9, 0.75] },
          { id: 3, label: 'dog', confidence: 0.92, bbox: [0.1, 0.6, 0.35, 0.9] },
          { id: 4, label: 'bicycle', confidence: 0.87, bbox: [0.65, 0.2, 0.85, 0.5] },
        ];
        resolve(objects);
      }, 1500);
    });
  };

  // Real API call for object detection
  const detectObjectsAPI = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE_URL}/api/detect`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`Detection failed: ${response.statusText}`);
    }
    
    const data = await response.json();
    return data.objects;
  };

  // Simulate Mask R-CNN segmentation (for testing without backend)
  const simulateSegmentation = (img, selectedIds) => {
    return new Promise((resolve) => {
      setTimeout(() => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = img.width;
        canvas.height = img.height;
        
        ctx.drawImage(img, 0, 0);
        
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        
        for (let i = 0; i < data.length; i += 4) {
          const x = (i / 4) % canvas.width;
          const y = Math.floor((i / 4) / canvas.width);
          
          let inSelectedObject = false;
          detectedObjects.forEach(obj => {
            if (selectedIds.has(obj.id)) {
              const [x1, y1, x2, y2] = obj.bbox.map((val, idx) => 
                idx % 2 === 0 ? val * canvas.width : val * canvas.height
              );
              if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
                inSelectedObject = true;
              }
            }
          });
          
          if (!inSelectedObject) {
            data[i + 3] = 0;
          }
        }
        
        ctx.putImageData(imageData, 0, 0);
        resolve(canvas.toDataURL('image/png'));
      }, 2000);
    });
  };

  // Real API call for segmentation
  const segmentObjectsAPI = async (file, selectedIds) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('selected_objects', JSON.stringify(Array.from(selectedIds)));
    
    const response = await fetch(`${API_BASE_URL}/api/segment`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`Segmentation failed: ${response.statusText}`);
    }
    
    const blob = await response.blob();
    return URL.createObjectURL(blob);
  };

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setError(null);
    setImageFile(file);

    const reader = new FileReader();
    reader.onload = async (event) => {
      const img = new Image();
      img.onload = async () => {
        setImage(event.target.result);
        setDetectedObjects([]);
        setSelectedObjects(new Set());
        setProcessedImage(null);
        
        // Auto-detect objects
        setIsDetecting(true);
        try {
          let objects;
          if (useSimulation) {
            objects = await simulateDetection(img);
          } else {
            objects = await detectObjectsAPI(file);
          }
          setDetectedObjects(objects);
        } catch (err) {
          setError(`Detection failed: ${err.message}`);
          console.error(err);
        } finally {
          setIsDetecting(false);
        }
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  };

  const toggleObjectSelection = (objectId) => {
    const newSelected = new Set(selectedObjects);
    if (newSelected.has(objectId)) {
      newSelected.delete(objectId);
    } else {
      newSelected.add(objectId);
    }
    setSelectedObjects(newSelected);
  };

  const selectAll = () => {
    setSelectedObjects(new Set(detectedObjects.map(obj => obj.id)));
  };

  const deselectAll = () => {
    setSelectedObjects(new Set());
  };

  const handleSegmentation = async () => {
    if (selectedObjects.size === 0) {
      alert('Please select at least one object');
      return;
    }

    setError(null);
    setIsSegmenting(true);
    
    try {
      let result;
      if (useSimulation) {
        const img = imageRef.current;
        result = await simulateSegmentation(img, selectedObjects);
      } else {
        result = await segmentObjectsAPI(imageFile, selectedObjects);
      }
      setProcessedImage(result);
    } catch (err) {
      setError(`Segmentation failed: ${err.message}`);
      console.error(err);
    } finally {
      setIsSegmenting(false);
    }
  };

  const downloadImage = () => {
    const link = document.createElement('a');
    link.download = 'background-removed.png';
    link.href = processedImage;
    link.click();
  };

  const reset = () => {
    setImage(null);
    setImageFile(null);
    setDetectedObjects([]);
    setSelectedObjects(new Set());
    setProcessedImage(null);
    setError(null);
  };

  // Draw bounding boxes
  useEffect(() => {
    if (!image || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = imageRef.current;

    if (!img) return;

    canvas.width = img.width;
    canvas.height = img.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detectedObjects.forEach((obj) => {
      const [x1, y1, x2, y2] = obj.bbox.map((val, idx) => 
        idx % 2 === 0 ? val * canvas.width : val * canvas.height
      );
      
      const isSelected = selectedObjects.has(obj.id);
      ctx.strokeStyle = isSelected ? '#10b981' : '#3b82f6';
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      
      ctx.fillStyle = isSelected ? '#10b981' : '#3b82f6';
      const label = `${obj.label} ${(obj.confidence * 100).toFixed(0)}%`;
      ctx.font = '16px sans-serif';
      const textWidth = ctx.measureText(label).width;
      ctx.fillRect(x1, y1 - 25, textWidth + 10, 25);
      
      ctx.fillStyle = 'white';
      ctx.fillText(label, x1 + 5, y1 - 7);
    });
  }, [image, detectedObjects, selectedObjects]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold mb-2 text-center bg-gradient-to-r from-blue-400 to-green-400 bg-clip-text text-transparent">
          Object Detection & Background Removal
        </h1>
        <p className="text-gray-400 text-center mb-4">
          Upload image → Select objects → Remove background
        </p>

        {/* Mode Toggle */}
        <div className="flex justify-center mb-6">
          <button
            onClick={() => setUseSimulation(!useSimulation)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              useSimulation
                ? 'bg-yellow-600 hover:bg-yellow-700'
                : 'bg-green-600 hover:bg-green-700'
            }`}
          >
            {useSimulation ? '🎭 Using Simulation Mode' : '🚀 Using Real Backend'}
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/50 border border-red-500 rounded-lg">
            <p className="text-red-200">⚠️ {error}</p>
            <p className="text-sm text-red-300 mt-2">
              {useSimulation 
                ? 'Something went wrong with the simulation.'
                : 'Make sure your backend is running at ' + API_BASE_URL}
            </p>
          </div>
        )}

        {!image ? (
          <div className="bg-gray-800 rounded-lg p-12 border-2 border-dashed border-gray-600 hover:border-blue-500 transition-colors">
            <label className="flex flex-col items-center cursor-pointer">
              <Upload className="w-16 h-16 mb-4 text-gray-400" />
              <span className="text-xl mb-2">Upload Image</span>
              <span className="text-sm text-gray-400">Click or drag and drop</span>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
            </label>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="flex gap-4 flex-wrap">
              <button
                onClick={reset}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg flex items-center gap-2 transition-colors"
              >
                <X className="w-4 h-4" />
                Reset
              </button>
              {detectedObjects.length > 0 && (
                <>
                  <button
                    onClick={selectAll}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center gap-2 transition-colors"
                  >
                    <CheckSquare className="w-4 h-4" />
                    Select All
                  </button>
                  <button
                    onClick={deselectAll}
                    className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg flex items-center gap-2 transition-colors"
                  >
                    <Square className="w-4 h-4" />
                    Deselect All
                  </button>
                  <button
                    onClick={handleSegmentation}
                    disabled={selectedObjects.size === 0 || isSegmenting}
                    className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg flex items-center gap-2 transition-colors"
                  >
                    {isSegmenting ? (
                      <>
                        <Loader className="w-4 h-4 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      'Remove Background'
                    )}
                  </button>
                </>
              )}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-gray-800 rounded-lg p-4">
                <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                  {isDetecting ? (
                    <>
                      <Loader className="w-5 h-5 animate-spin" />
                      Detecting Objects...
                    </>
                  ) : (
                    `Detected Objects (${detectedObjects.length})`
                  )}
                </h2>
                <div className="relative bg-gray-900 rounded-lg overflow-hidden">
                  <img
                    ref={imageRef}
                    src={image}
                    alt="Original"
                    className="w-full h-auto"
                    crossOrigin="anonymous"
                  />
                  <canvas
                    ref={canvasRef}
                    className="absolute top-0 left-0 w-full h-full pointer-events-none"
                  />
                </div>
              </div>

              <div className="bg-gray-800 rounded-lg p-4">
                {!processedImage ? (
                  <>
                    <h2 className="text-xl font-semibold mb-4">
                      Select Objects ({selectedObjects.size} selected)
                    </h2>
                    <div className="space-y-2">
                      {detectedObjects.map((obj) => (
                        <button
                          key={obj.id}
                          onClick={() => toggleObjectSelection(obj.id)}
                          className={`w-full p-4 rounded-lg border-2 transition-all text-left ${
                            selectedObjects.has(obj.id)
                              ? 'border-green-500 bg-green-900/30'
                              : 'border-gray-600 bg-gray-700 hover:border-blue-500'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <div>
                              <div className="font-semibold text-lg capitalize">{obj.label}</div>
                              <div className="text-sm text-gray-400">
                                Confidence: {(obj.confidence * 100).toFixed(1)}%
                              </div>
                            </div>
                            {selectedObjects.has(obj.id) ? (
                              <CheckSquare className="w-6 h-6 text-green-500" />
                            ) : (
                              <Square className="w-6 h-6 text-gray-400" />
                            )}
                          </div>
                        </button>
                      ))}
                    </div>
                  </>
                ) : (
                  <>
                    <h2 className="text-xl font-semibold mb-4 flex items-center justify-between">
                      Processed Image
                      <button
                        onClick={downloadImage}
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center gap-2 transition-colors text-sm"
                      >
                        <Download className="w-4 h-4" />
                        Download
                      </button>
                    </h2>
                    <div className="bg-gray-900 rounded-lg overflow-hidden">
                      <div className="relative">
                        <div className="absolute inset-0 bg-[linear-gradient(45deg,#808080_25%,transparent_25%,transparent_75%,#808080_75%,#808080),linear-gradient(45deg,#808080_25%,transparent_25%,transparent_75%,#808080_75%,#808080)] bg-[length:20px_20px] bg-[position:0_0,10px_10px]" />
                        <img
                          src={processedImage}
                          alt="Processed"
                          className="w-full h-auto relative z-10"
                        />
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        <div className="mt-8 p-4 bg-gray-800 rounded-lg border border-gray-700">
          <h3 className="font-semibold mb-2">🔧 Setup Instructions:</h3>
          <ul className="text-sm text-gray-400 space-y-1">
            <li>• Toggle between Simulation Mode (no backend needed) and Real Backend</li>
            <li>• Backend URL configured at: {API_BASE_URL}</li>
            <li>• To use real models: Start backend with "python backend.py"</li>
            <li>• Install dependencies: pip install fastapi uvicorn torch torchvision pillow python-multipart</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
