from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import io
import json
import numpy as np

app = FastAPI()

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000', 'http://localhost:8000', 'http://127.0.0.1:3000', 'http://127.0.0.1:8000', 'http://localhost:5173'],  # Allow multiple ports for development (Vite default is 5173)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SELECTED_CLASSES = [
    '__background__',
    'person', 'car', 'dog', 'cat', 'bus', 'truck', 'bicycle', 'motorcycle', 'bench',
    'bird', 'horse', 'sheep', 'cow', 'elephant', 'traffic light', 'stop sign',
    'fire hydrant', 'boat', 'train', 'airplane'
]

NUM_CLASSES = len(SELECTED_CLASSES)

# Load models (do this once at startup)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# For Faster R-CNN
faster_rcnn = fasterrcnn_resnet50_fpn(weights='DEFAULT')
in_features = faster_rcnn.roi_heads.box_predictor.cls_score.in_features
faster_rcnn.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
faster_rcnn.load_state_dict(torch.load('models/faster_rcnn_weights.pth', map_location=device))
faster_rcnn.to(device)
faster_rcnn.eval()

# For Mask R-CNN
mask_rcnn = maskrcnn_resnet50_fpn(weights='DEFAULT')
in_features = mask_rcnn.roi_heads.box_predictor.cls_score.in_features
mask_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
in_features_mask = mask_rcnn.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
mask_rcnn.roi_heads.mask_predictor = MaskRCNNPredictor(
    in_features_mask, hidden_layer, NUM_CLASSES
)
mask_rcnn.load_state_dict(torch.load('models/mask_rcnn_weights.pth', map_location=device))
mask_rcnn.to(device)
mask_rcnn.eval()

# COCO class names
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

@app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Detect objects using Faster R-CNN
    Returns: List of detected objects with bounding boxes and labels
    """
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Convert to tensor
    transform = torchvision.transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run detection
    with torch.no_grad():
        predictions = faster_rcnn(image_tensor)[0]
    
    # Filter predictions by confidence threshold
    threshold = 0.1
    boxes = predictions['boxes'][predictions['scores'] > threshold].cpu().numpy()
    labels = predictions['labels'][predictions['scores'] > threshold].cpu().numpy()
    scores = predictions['scores'][predictions['scores'] > threshold].cpu().numpy()
    
    # Get image dimensions
    width, height = image.size
    
    # Format results
    objects = []
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = box
        class_name = SELECTED_CLASSES[label] if label < len(SELECTED_CLASSES) else 'unknown'
        
        objects.append({
            'id': i + 1,
            'label': class_name,
            'confidence': float(score),
            'bbox': [
                float(x1 / width),   # normalized x1
                float(y1 / height),  # normalized y1
                float(x2 / width),   # normalized x2
                float(y2 / height)   # normalized y2
            ]
        })
    
    return {'objects': objects}


@app.post("/api/segment")
async def segment_objects(
    file: UploadFile = File(...),
    selected_objects: str = Form(...)
):
    """
    Segment selected objects and remove background using Mask R-CNN
    Returns: PNG image with transparent background
    """
    # Parse selected object IDs
    selected_ids = json.loads(selected_objects)
    
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Convert to tensor
    transform = torchvision.transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run Mask R-CNN
    with torch.no_grad():
        predictions = mask_rcnn(image_tensor)[0]
    
    # Filter by confidence threshold
    threshold = 0.5
    masks = predictions['masks'][predictions['scores'] > threshold].cpu().numpy()
    scores = predictions['scores'][predictions['scores'] > threshold].cpu().numpy()
    
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Create empty mask for selected objects
    height, width = img_array.shape[:2]
    combined_mask = np.zeros((height, width), dtype=bool)
    
    # Combine masks for selected objects (1-indexed to 0-indexed)
    for obj_id in selected_ids:
        if obj_id - 1 < len(masks):
            mask = masks[obj_id - 1][0] > 0.5  # Threshold mask
            combined_mask = np.logical_or(combined_mask, mask)
    
    # Create RGBA image
    result = np.zeros((height, width, 4), dtype=np.uint8)
    result[:, :, :3] = img_array  # RGB channels
    result[:, :, 3] = (combined_mask * 255).astype(np.uint8)  # Alpha channel
    
    # Convert to PIL Image
    result_image = Image.fromarray(result, 'RGBA')
    
    # Save to bytes
    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return StreamingResponse(img_byte_arr, media_type="image/png")


@app.get("/")
async def root():
    return {"message": "Object Detection API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
