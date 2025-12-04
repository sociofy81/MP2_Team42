import os
from pydantic import BaseModel
import io
import uuid
import shutil
from typing import List
import traceback
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import numpy as np
from PIL import Image
import cv2
import torch
from mobile_sam import sam_model_registry, SamPredictor
from simple_lama_inpainting import SimpleLama
import uvicorn
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

class SegmentRequest(BaseModel):
    job_id: str
    x: int
    y: int

# -----------------------
# Configuration
# -----------------------
MOBILE_SAM_CHECKPOINT = "./mobile_sam.pt"
WORK_DIR = Path("./jobs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_IMAGE_SIZE = 1024
# -----------------------

WORK_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Segmentation & Inpainting API - MobileSAM + LaMa")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global job status map
jobs = {}

# Global model instances
mobile_sam_predictor = None
lama_model = None

# -----------------------
# Model Loading
# -----------------------
def load_models():
    global mobile_sam_predictor, lama_model
    
    print("="*60)
    print("🎨 Initializing AI Models")
    print("="*60)
    
    print("🚀 Loading MobileSAM...")
    model_type = "vit_t"
    
    if not os.path.exists(MOBILE_SAM_CHECKPOINT):
        raise FileNotFoundError(f"MobileSAM checkpoint not found at {MOBILE_SAM_CHECKPOINT}")
    
    mobile_sam = sam_model_registry[model_type](checkpoint=MOBILE_SAM_CHECKPOINT)
    mobile_sam.to(device=DEVICE)
    mobile_sam.eval()
    mobile_sam_predictor = SamPredictor(mobile_sam)
    print(f"✓ MobileSAM loaded! (Device: {DEVICE})")
    
    print("🦙 Loading LaMa Inpainting Model...")
    lama_model = SimpleLama()
    print("✓ LaMa loaded!")
    
    print("="*60)

load_models()

# -----------------------
# Utility Functions
# -----------------------
def read_image_bytes(img_bytes: bytes):
    """Read image from bytes and return PIL Image."""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")

def resize_image(img: Image.Image, max_size=MAX_IMAGE_SIZE):
    """Resize image if necessary."""
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img

def segment_object(img_np: np.ndarray, coordinates: tuple):
    """
    Segment a single object using MobileSAM with point selection only.
    Returns boolean mask (H, W).
    """
    mobile_sam_predictor.set_image(img_np)
    
    x, y = coordinates
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    masks, scores, _ = mobile_sam_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    best_idx = int(np.argmax(scores))
    mask = masks[best_idx]
    mask_bool = mask > 0.5
    return mask_bool

def create_enhanced_mask(mask: np.ndarray) -> np.ndarray:
    """Create enhanced mask from boolean mask (balanced method)."""
    mask_uint8 = (mask.astype(np.uint8) * 255).astype(np.uint8)
    
    # Use balanced method (5x5 dilation + Gaussian blur)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    enhanced_mask = cv2.GaussianBlur(dilated, (5, 5), 0)
    
    return enhanced_mask > 127

def inpaint_object(img_np: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """Inpaint a single object using LaMa."""
    try:
        mask_bool = (mask > 0.5)
        enhanced_mask = create_enhanced_mask(mask_bool)
        
        if img_np.dtype != np.uint8:
            img_uint8 = (img_np * 255).astype(np.uint8)
        else:
            img_uint8 = img_np.copy()
        
        img_pil = Image.fromarray(img_uint8)
        mask_uint8 = (enhanced_mask.astype(np.uint8) * 255)
        mask_pil = Image.fromarray(mask_uint8).convert('L')
        
        inpainted_pil = lama_model(img_pil, mask_pil)
        inpainted_np = np.array(inpainted_pil)
        
        if inpainted_np.shape[:2] != img_uint8.shape[:2]:
            inpainted_np = cv2.resize(
                inpainted_np,
                (img_uint8.shape[1], img_uint8.shape[0]),
                interpolation=cv2.INTER_LANCZOS4
            )
        
        return inpainted_np
    except Exception as e:
        print(f"Inpainting error: {e}")
        return None

def get_bbox_from_mask(mask: np.ndarray, padding: int = 10) -> Optional[tuple]:
    """Calculate bounding box from boolean mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return None
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    h, w = mask.shape
    
    y_min = max(0, y_min - padding)
    y_max = min(h, y_max + padding + 1)
    x_min = max(0, x_min - padding)
    x_max = min(w, x_max + padding + 1)
    
    return x_min, y_min, x_max, y_max

def crop_with_transparency(img_np: np.ndarray, mask: np.ndarray, padding: int = 10) -> Optional[np.ndarray]:
    """Crop object with alpha channel (RGBA)."""
    if mask is None or np.sum(mask) == 0:
        return None
    
    bbox = get_bbox_from_mask(mask, padding)
    if bbox is None:
        return None
    
    x1, y1, x2, y2 = bbox
    cropped_img = img_np[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    
    h, w = cropped_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = cropped_img
    rgba[:, :, 3] = (cropped_mask.astype(np.uint8) * 255).astype(np.uint8)
    
    return rgba

# -----------------------
# Background Job Processing
# -----------------------
def process_segmentation_job(
    job_id: str,
    img_bytes: bytes,
    coordinates_list: List[tuple]
):
    """Process segmentation, cropping, and inpainting (point selection only)."""
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    jobs[job_id] = {"status": "processing", "progress": 0}
    
    try:
        # Load and prepare image
        img_pil = read_image_bytes(img_bytes)
        img_pil = resize_image(img_pil, MAX_IMAGE_SIZE)
        img_np = np.array(img_pil)
        
        # Save original
        orig_path = job_dir / "original.png"
        img_pil.save(str(orig_path))
        jobs[job_id]["original_size"] = img_pil.size
        
        # Segmentation (point selection only)
        jobs[job_id]["progress"] = 20
        masks_list = []
        for i, coord in enumerate(coordinates_list):
            try:
                mask = segment_object(img_np, coord)
                mask_uint8 = (mask.astype(np.uint8) * 255)
                mask_path = job_dir / f"mask_{i}.png"
                cv2.imwrite(str(mask_path), mask_uint8)
                masks_list.append(mask)
            except Exception as e:
                print(f"Segmentation failed for object {i}: {e}")
                masks_list.append(None)
        
        # Cropping
        jobs[job_id]["progress"] = 40
        cropped_list = []
        for i, mask in enumerate(masks_list):
            if mask is not None and np.sum(mask) > 0:
                cropped = crop_with_transparency(img_np, mask, padding=10)
                if cropped is not None:
                    cropped_path = job_dir / f"cropped_{i}.png"
                    Image.fromarray(cropped, mode="RGBA").save(str(cropped_path))
                    cropped_list.append(cropped)
                else:
                    cropped_list.append(None)
            else:
                cropped_list.append(None)
        
        # Inpainting
        jobs[job_id]["progress"] = 60
        inpainted_list = []
        for i, mask in enumerate(masks_list):
            if mask is not None and np.sum(mask) > 0:
                inpainted = inpaint_object(img_np, mask)
                if inpainted is not None:
                    inpainted_path = job_dir / f"inpainted_{i}.png"
                    Image.fromarray(inpainted.astype(np.uint8)).save(str(inpainted_path))
                    inpainted_list.append(inpainted)
                else:
                    inpainted_list.append(None)
            else:
                inpainted_list.append(None)
        
        # Full-image inpainting after all removals
        jobs[job_id]["progress"] = 80
        combined_mask = None
        for mask in masks_list:
            if mask is not None:
                combined_mask = mask if combined_mask is None else (combined_mask | mask)
        
        if combined_mask is not None and np.sum(combined_mask) > 0:
            full_inpaint = inpaint_object(img_np, combined_mask)
            if full_inpaint is not None:
                full_inpaint_path = job_dir / "full_inpainted.png"
                Image.fromarray(full_inpaint.astype(np.uint8)).save(str(full_inpaint_path))
        
        jobs[job_id]["status"] = "ready"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["job_dir"] = str(job_dir.resolve())
        jobs[job_id]["num_objects"] = len(masks_list)
        jobs[job_id]["successful_segmentations"] = sum(1 for m in masks_list if m is not None)
        
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e) + "\n" + traceback.format_exc()
        print(f"Error processing job {job_id}: {e}")

# -----------------------
# API Endpoints
# -----------------------

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...)
):
    """
    Upload an image for processing.
    Returns job_id for tracking.
    """
    try:
        contents = await file.read()
        job_id = uuid.uuid4().hex
        jobs[job_id] = {"status": "queued"}
        
        # Quick save
        job_dir = WORK_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "orig_uploaded.png").write_bytes(contents)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Image uploaded. Send segmentation request to start processing."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: {e}")

@app.post("/segment")
async def segment_image(
    payload: SegmentRequest,
    background_tasks: BackgroundTasks = None
):
    """
    Start segmentation, cropping, and inpainting process with point selection.
    
    Args:
        job_id: Job ID from upload endpoint
        coordinates_list: List of (x, y) point coordinates
    """
    job_id = payload.job_id
    coordinates_list = [(payload.x, payload.y)]
    print(coordinates_list)


    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_dir = WORK_DIR / job_id
    orig_path = job_dir / "orig_uploaded.png"
    
    if not orig_path.exists():
        raise HTTPException(status_code=400, detail="Original image not found")
    
    try:
        img_bytes = orig_path.read_bytes()
        
        if background_tasks:
            # Run in background if background_tasks is provided
            background_tasks.add_task(
                process_segmentation_job,
                job_id,
                img_bytes,
                coordinates_list
            )
        else:
            # Run synchronously (for testing)
            process_segmentation_job(job_id, img_bytes, coordinates_list)
        
        return {"job_id": job_id, "status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation start failed: {e}")

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get job processing status and progress."""
    info = jobs.get(job_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return info

@app.get("/download/{job_id}/{filename}")
async def download_artifact(job_id: str, filename: str):
    """
    Download processed artifacts.
    
    Filenames:
        - original.png
        - cropped_0.png, cropped_1.png, ...
        - inpainted_0.png, inpainted_1.png, ...
        - full_inpainted.png
        - mask_0.png, mask_1.png, ...
    """
    info = jobs.get(job_id)
    if info is None or info.get("status") != "ready":
        raise HTTPException(status_code=404, detail="Job not found or not ready")
    
    job_dir = Path(info["job_dir"])
    path = job_dir / filename
    
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return StreamingResponse(
        path.open("rb"),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.delete("/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """Delete job and all associated files."""
    info = jobs.get(job_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_dir = Path(info.get("job_dir", ""))
    if job_dir.exists():
        shutil.rmtree(job_dir)
    
    jobs.pop(job_id, None)
    return {"status": "deleted", "job_id": job_id}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "models_loaded": mobile_sam_predictor is not None and lama_model is not None
    }

@app.get("/")
async def root():
    """API documentation."""
    return {
        "name": "Segmentation & Inpainting API",
        "version": "1.0",
        "description": "Point selection only with inpainting",
        "endpoints": {
            "POST /upload": "Upload image",
            "POST /segment": "Run segmentation and inpainting (send point coordinates)",
            "GET /status/{job_id}": "Check job status",
            "GET /download/{job_id}/{filename}": "Download results",
            "DELETE /cleanup/{job_id}": "Clean up job",
            "GET /health": "Health check"
        }
    }

# -----------------------
# Run Server
# -----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
