# Segmentation & Inpainting AI Engine

## Executive Vision

The future of content creation is conversational, lightweight, and mobile-first. Professional photo and video editors today require desktop-class hardware and complex interfaces. By 2030, creators expect AI-assisted workflows that operate on limited compute—where a simple tap or voice command like "remove this object" or "extract this person" instantly delivers production-ready results.

This AI engine enables that vision. It brings desktop-quality segmentation and object removal to any device, preserving human control while automating tedious selection and inpainting tasks. Photographers, designers, marketers, and creators can now focus on creative intent rather than tool mastery.

## Core Purpose

Real-time, point-and-click object segmentation and removal powered by lightweight AI models optimized for mobile and edge devices. No bounding boxes. No complex masks. Just tap an object and watch it disappear—or extract it with perfect transparency.

## Key Capabilities

- **Point-Based Segmentation**: Tap any object once to segment it instantly. No bounding boxes, no manual tracing, no training required.

- **Intelligent Object Removal**: Automatically fill removed areas with contextually aware inpainting that matches surrounding textures and colors.

- **Batch Processing**: Segment and remove multiple objects from a single image in parallel, perfect for complex scenes.

- **Transparent Extraction**: Export selected objects as PNG with full alpha channel for compositing in other applications.

- **Async Background Processing**: Large images process without blocking the UI. Check progress in real-time, download when ready.

- **RESTful Integration**: Standard HTTP endpoints designed for seamless integration with web, mobile, and desktop frontends.

## Architecture Overview

The pipeline combines two lightweight AI models in a proven workflow:

```
Client Upload
    ↓
FastAPI Server (async job queue)
    ↓
Image Preprocessing (resize, normalize)
    ↓
MobileSAM Segmentation (point → mask)
    ↓
Parallel Processing:
├─ Object Extraction (crop + alpha channel)
├─ Mask Refinement (dilation + blur)
└─ LaMa Inpainting (fill removed regions)
    ↓
Output Assembly (results + metadata)
    ↓
Client Download (PNG, JPEG, JSON)
```

### Why This Architecture?

- **Lightweight Models**: MobileSAM (10M parameters) runs on phones and laptops, not just GPUs.
- **Parallel Execution**: Segmentation, cropping, and inpainting happen concurrently for speed.
- **Background Jobs**: AsyncIO prevents timeouts on slow networks or large images.
- **Modular Design**: Each component (segmentation, inpainting, extraction) can be replaced independently.
- **Zero Client State**: All computation happens server-side; clients remain stateless and responsive.

## Performance Profile
<img width="2250" height="1650" alt="deepseek_mermaid_20251204_940160" src="https://github.com/user-attachments/assets/68da1e1b-8772-4b00-8206-841c7a3d8c71" />

| Component | Model Size | GPU Runtime | CPU Runtime | Memory Peak | Notes |
|-----------|------------|------------|------------|------------|-------|
| Image Loading | - | 50ms | 50ms | 50-200MB | Includes resizing and normalization |
| MobileSAM Segmentation | 40MB | 100ms | 500ms | 500MB | Single point inference |
| Mask Refinement | - | 30ms | 30ms | 100MB | Dilation + Gaussian blur |
| LaMa Inpainting | 100MB | 200ms | 2000ms | 1000MB | Handles large masked regions |
| Object Cropping | - | 50ms | 50ms | 100MB | RGBA export with padding |
| **Total Pipeline** | **140MB** | **350ms** | **2550ms** | **2GB Peak** | Multi-object processing |

Optimization notes:
- GPU acceleration (CUDA) reduces latency by 7-8x for segmentation and inpainting.
- CPU path remains viable for mobile inference and edge devices (< 3 seconds for typical images).
- Memory peaks during inpainting; smaller images (<512px) fit comfortably on 2GB devices.

## System Design Decisions

### 1. Threading Over Background Tasks
**Decision**: Use Python `threading.Thread` instead of FastAPI BackgroundTasks.

**Rationale**: Direct thread control ensures status updates are synchronous and reliable across all clients. BackgroundTasks introduces timing ambiguity when coordinating job state changes with HTTP response cycles, especially problematic for polling-based status checks.

### 2. Point-Based Input Over Bounding Boxes
**Decision**: Accept single (x, y) click coordinates rather than rectangular regions.

**Rationale**: Reduces cognitive load for users. A single tap is faster than dragging a box; users can remain in their photo-viewing context without mode-switching. MobileSAM's efficiency makes per-point segmentation economical.

### 3. Async Job Processing Over Synchronous Returns
**Decision**: All processing happens asynchronously; clients poll status endpoints.

**Rationale**: Prevents server timeouts on large images. Clients remain responsive even while waiting. Progress tracking provides visual feedback, improving perceived performance.

### 4. File-System Storage Over In-Memory Caching
**Decision**: Persist intermediate results (masks, crops, inpainted versions) to disk.

**Rationale**: Allows clients to resume downloads if interrupted. Enables debugging and result review without re-processing. Scales to multiple concurrent users without RAM pressure.

### 5. Transparent PNG Export Over JPG
**Decision**: Cropped objects export as RGBA PNG by default.

**Rationale**: Preserves alpha channel for compositing. Creators expect to drop extracted objects into other designs; JPG compression with opaque backgrounds breaks this workflow.

## API Endpoints
<img width="2895" height="3191" alt="deepseek_mermaid_20251204_37dcc3" src="https://github.com/user-attachments/assets/f3f2338c-669a-490c-b025-437ceac1aff1" />


### Workflow Endpoints

| Endpoint | Method | Purpose | Request | Response |
|----------|--------|---------|---------|----------|
| `/upload` | POST | Register image for processing | `multipart/form-data (file)` | `{job_id, status: "uploaded"}` |
| `/segment` | POST | Start segmentation at point(s) | `?job_id=X&coordinates_list=[[x,y],...]` | `{job_id, status: "processing"}` |
| `/status/{job_id}` | GET | Poll job progress | - | `{status, progress, num_objects, ...}` |
| `/download/{job_id}/{filename}` | GET | Retrieve result file | - | Binary file stream |
| `/cleanup/{job_id}` | DELETE | Purge job data | - | `{status: "deleted"}` |

### Utility Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | API documentation (JSON) |
| `/health` | GET | System health check (device, models loaded) |

### Downloadable Artifacts

For each job, the following files are available:

- `original.png` - Uploaded image (resized)
- `mask_0.png`, `mask_1.png`, ... - Binary segmentation masks
- `cropped_0.png`, `cropped_1.png`, ... - Extracted objects (RGBA)
- `inpainted_0.png`, `inpainted_1.png`, ... - Per-object removal results
- `full_inpainted.png` - Final image with all objects removed and background filled

## Installation

### Prerequisites

- Python 3.8 or later
- CUDA 11.8+ (optional, for GPU acceleration)
- 4GB+ RAM (8GB+ recommended)
- 2GB free disk space for model weights and temporary files

### Setup Steps

1. **Clone and Navigate**
```bash
git clone <repository-url>
cd segmentation-api
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Download Model Weights**
```bash
# MobileSAM checkpoint (40MB)
wget https://github.com/ChaoningZhang/MobileSAM/releases/download/v1.0/mobile_sam.pt
```

4. **Install Dependencies**
```bash
# PyTorch with CUDA support (adjust index-url for your setup)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core dependencies
pip install -r requirements.txt
```

5. **Verify Installation**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "from mobile_sam import sam_model_registry; print('MobileSAM available')"
```

## Workflow

### Step 1: Start the Server

```bash
python main.py
# or with custom settings:
python main.py --host 0.0.0.0 --port 8000 --reload
```

Server logs will show:
```
============================================================
🎨 Initializing AI Models
============================================================
🚀 Loading MobileSAM...
✓ MobileSAM loaded! (Device: cuda)
🦙 Loading LaMa Inpainting Model...
✓ LaMa loaded!
============================================================
INFO: Uvicorn running on http://0.0.0.0:8000
```

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

### Step 2: Upload an Image

Client sends image file:

```bash
curl -X POST -F "file=@photo.jpg" http://localhost:8000/upload
```

Server responds with:
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "uploaded",
  "message": "Image uploaded. Send coordinates to segment."
}
```

### Step 3: Segment Objects

Client specifies points (x, y) to segment:

```bash
curl -X POST "http://localhost:8000/segment?job_id=a1b2c3d4e5f6" \
  -H "Content-Type: application/json" \
  -d '{"coordinates_list": [[250, 300], [400, 150]]}'
```

Backend immediately:
- Sets job status to "processing"
- Starts background thread
- Returns 200 OK

```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "processing",
  "message": "Segmenting 2 objects..."
}
```

Processing pipeline executes:
1. Load image, normalize
2. Run MobileSAM for each point → masks
3. Extract objects with transparency (RGBA)
4. Refine masks (dilation + blur)
5. Inpaint using LaMa
6. Assemble final results

### Step 4: Poll Status and Download

Client polls every 1 second:

```bash
curl http://localhost:8000/status/a1b2c3d4e5f6
```

Response during processing:
```json
{
  "status": "processing",
  "progress": 45,
  "job_dir": "/path/to/jobs/a1b2c3d4e5f6"
}
```

Once complete (`status: "ready"`), download artifacts:

```bash
# Download full inpainted image
curl http://localhost:8000/download/a1b2c3d4e5f6/full_inpainted.png -o result.png

# Download extracted object 0 (transparent)
curl http://localhost:8000/download/a1b2c3d4e5f6/cropped_0.png -o object.png

# Download segmentation mask for object 1
curl http://localhost:8000/download/a1b2c3d4e5f6/mask_1.png -o mask.png
```

### Step 5: Cleanup

Remove job data when finished:

```bash
curl -X DELETE http://localhost:8000/cleanup/a1b2c3d4e5f6
```

All temporary files and job metadata are purged:
```json
{
  "status": "deleted",
  "job_id": "a1b2c3d4e5f6"
}
```

## Error Handling

### Common Scenarios

**Job Not Found**
- Cause: Job ID expired (cleaned up), mistyped, or never uploaded
- HTTP 404
- Mitigation: Verify job_id immediately after upload; cleanup only when finished

```json
{
  "detail": "Job not found"
}
```

**Image Upload Fails**
- Cause: Invalid file format, corrupted file, too large
- HTTP 400
- Mitigation: Validate image format (JPEG, PNG) and size (<50MB) client-side

```json
{
  "detail": "Upload failed: Invalid image data: unidentified image file"
}
```

**Segmentation Error (no mask found)**
- Cause: Point clicked on background or edge case where MobileSAM returns empty mask
- Status: "error"
- Mitigation: Try clicking closer to object center; check image contrast

```json
{
  "status": "error",
  "error": "Segmentation failed for object 0: mask too small"
}
```

**Out of Memory (GPU)**
- Cause: Image too large for available VRAM
- Mitigation: Reduce `MAX_IMAGE_SIZE` in config, or reduce batch size (process fewer objects per job)

```python
# main.py
MAX_IMAGE_SIZE = 512  # reduce from 1024
```

**Timeout (large image on CPU)**
- Cause: Processing takes > client timeout (typically 30s)
- Mitigation: Client should poll status endpoint in loop, not wait for single response

## Configuration

### Environment Variables (`.env` file)

```env
PORT=8000
HOST=0.0.0.0
WORKERS=1
LOG_LEVEL=info

MAX_IMAGE_SIZE=1024
DEVICE=cuda
MOBILE_SAM_CHECKPOINT=./mobile_sam.pt
WORK_DIR=./jobs
```

### Runtime Parameters

Edit `main.py` to customize:

```python
MOBILE_SAM_CHECKPOINT = "./mobile_sam.pt"  # Path to model weights
WORK_DIR = Path("./jobs")                  # Temporary storage
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_IMAGE_SIZE = 1024                      # Max image dimension (pixels)
```

## Model Details

### MobileSAM (Segmentation)

- **Architecture**: Vision Transformer Tiny (ViT-T)
- **Parameters**: 10 million (vs. 630M in original SAM)
- **Input**: Image + single (x, y) click point
- **Output**: Binary mask (0 = background, 1 = object)
- **Latency**: ~100ms GPU, ~500ms CPU
- **Accuracy**: ~95% IoU on common objects; comparable to full SAM for typical use cases
- **Why it's efficient**: Lightweight attention mechanisms; distilled from larger SAM via knowledge transfer

### LaMa (Inpainting)

- **Architecture**: Fast Fourier Convolutions (FFCs) with learnable spectral bias
- **Purpose**: Fill masked regions with plausible content
- **Strengths**: Handles large missing areas (> 50% of image), preserves fine details
- **Weaknesses**: Can blur textures if mask edges are imprecise
- **Speed**: ~200ms GPU, ~2s CPU per image

### Combined Strengths

The pairing is synergistic: MobileSAM's precise boundary detection feeds into LaMa, which has proven robust to boundary imperfections. This avoids error cascade common in older inpainting pipelines.

## Development Notes

### Code Structure

- **`main.py`**: FastAPI app, endpoints, job orchestration (~500 lines)
- **Model Loading** (`load_models()`): Initializes MobileSAM and LaMa on startup
- **Job Processing** (`process_segmentation_job()`): Background thread function that coordinates pipeline
- **Utility Functions**: Image I/O, mask refinement, cropping, inpainting wrapping

### Testing Locally

```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Full workflow (Python)
python tests/integration_test.py

# 3. Swagger UI
# Visit: http://localhost:8000/docs
# Try endpoints interactively
```

### Extending the Pipeline

To add a new processing step (e.g., edge sharpening):

1. Write function: `def sharpen_image(img_np) -> np.ndarray`
2. Insert into `process_segmentation_job()` after inpainting
3. Save output: `Image.fromarray(...).save(job_dir / "sharpened.png")`
4. Update `/download` endpoint filename whitelist if needed

## Caution and Limitations

**GPU Memory**: On NVIDIA GPUs with < 2GB VRAM, reduce `MAX_IMAGE_SIZE` to 512 or process one object at a time. Monitor `nvidia-smi` during heavy loads.

**Segmentation Accuracy**: MobileSAM occasionally fails on:
- Thin or complex boundaries (hair, foliage)
- Objects with low contrast to background
- Very small objects (< 50 pixels)

Workaround: Tap multiple nearby points for a single object; the model's multi-mask output will refine the result.

**Inpainting Artifacts**: LaMa can introduce watercolor-like blurs if:
- Mask edges are jagged (use mask refinement)
- Background is highly textured (complex patterns)
- Lighting is inconsistent (shadows cast across removal region)

For production use, consider post-processing inpainted regions with Gaussian blur or color correction.

**Cleanup Policy**: Jobs are NOT automatically cleaned up. Call `/cleanup/{job_id}` manually or implement a scheduled task (cron) to purge old jobs:

```bash
# Delete jobs older than 24 hours
find ./jobs -type d -mtime +1 -exec rm -r {} \;
```

**Concurrent Users**: Each job spawns a background thread. With 10+ simultaneous jobs on limited hardware, queueing and memory pressure increase. For production, consider:
- Job queue (Redis + Celery)
- Load balancer (nginx)
- Auto-scaling infrastructure (Kubernetes)

---

## Citation and Attribution

Built on:
- **MobileSAM**: Chaoning Zhang et al., "Faster Segment Anything: Towards Lightweight SAM for Mobile Applications"
- **LaMa**: Roman Suvorov et al., "Resolution-robust Large Mask Inpainting with Fourier Convolutions"

For questions, issues, or feature requests, open an issue on the repository.

Made for creators who demand speed, control, and intelligence in equal measure.
