# server.py
import os
import io
import uuid
import shutil
import traceback
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from diffusers import MarigoldNormalsPipeline
import sys
import uvicorn
from pathlib import Path
from starlette.concurrency import run_in_threadpool

# -----------------------
# Configuration: Edit as needed
# EDIT AS PER WRITTEN IN README_AI_RELIGHT_ENGINE.md
# -----------------------
REPO_PATH = "../Depth-Anything-V2"   # UPDATE IF DIFFERENT TO Depth-Anything-V2 repo directory
DEPTH_CHECKPOINT = "../Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth" # UPDATE IF DIFFERENT TO THE depth_anything_v2_vitl.pth file
MARIGOLD_ID = "prs-eth/marigold-normals-v1-1"
WORK_DIR = Path("./jobs")
DEVICE = torch.device("cpu")
NUM_MARIGOLD_STEPS = 10   # lower -> faster preview, raise for quality
# -----------------------

# Make workspace
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Add repo path so DepthAnything import works
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

# Import DepthAnything model (assumes repository structure matches)
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except Exception as e:
    print("Could not import DepthAnythingV2. Make sure REPO_PATH is correct and repo present.")
    raise

app = FastAPI(title="Relighting API - DepthAnything + Marigold (local)")

# Allow CORS for testing from phone (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory job status map (small)
jobs = {}

# -----------------------
# Load models (at startup)
# -----------------------
print("Device:", DEVICE)
print("Loading DepthAnything V2 checkpoint... (this may take a while)")

depth_model = None
marigold_pipe = None

def load_models():
    global depth_model, marigold_pipe
    # DepthAnything
    depth_model = DepthAnythingV2(encoder='vitl')  # as in your code
    checkpoint = torch.load(DEPTH_CHECKPOINT, map_location="cpu")
    depth_model.load_state_dict(checkpoint)
    depth_model.eval()
    depth_model.to(DEVICE)
    print("DepthAnything loaded.")

    # MarigoldNormals pipeline (diffusion)
    try:
        marigold_pipe = MarigoldNormalsPipeline.from_pretrained(MARIGOLD_ID)
        marigold_pipe.to(DEVICE)
        print("MarigoldNormals pipeline loaded.")
    except Exception as e:
        print("Warning: failed to load MarigoldNormalsPipeline:", e)
        marigold_pipe = None

# load synchronously on start
load_models()

# -----------------------
# Utility helpers
# -----------------------
def read_image_bytes_to_bgr(np_bytes: bytes):
    arr = np.frombuffer(np_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")
    return img

def save_pil_as_jpeg_bytes(pil_img: Image.Image, quality=95) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def relight_from_artifacts(job_dir: Path, light_x: float, light_y: float, light_z: float,
                           color_hex: str = "#ffffff",
                           intensity: float = 2.2,
                           ambient: float = 0.18,
                           specular_weight: float = 0.6,
                           shininess: float = 64.0,
                           boost: float = 600.0,
                           shadow_strength: float = 6.0,
                           shadow_bias: float = 0.02):
    """
    Load original image, depth and normals from job_dir and compute relit image.
    Returns bytes (JPEG).
    """
    # Paths
    orig_path = job_dir / "orig.png"
    depth_path = job_dir / "depth.png"
    normal_path = job_dir / "normals.png"

    if not (orig_path.exists() and depth_path.exists() and normal_path.exists()):
        raise FileNotFoundError("Missing artifacts for job")

    # 1. load images
    orig_bgr = cv2.imread(str(orig_path))
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    H, W = orig_rgb.shape[:2]

    # load normals (ensure correct shape)
    norm_img = Image.open(str(normal_path)).convert("RGB")
    norm_img = norm_img.resize((W, H), Image.BILINEAR)
    norm_np = np.array(norm_img).astype(np.float32) / 255.0   # 0..1

    # convert normals to [-1,1] and normalize
    normals_np = norm_np * 2.0 - 1.0
    norm_mag = np.linalg.norm(normals_np, axis=-1, keepdims=True) + 1e-6
    normals_np = normals_np / norm_mag

    # depth: grayscale path (we saved earlier). load and normalize 0..1
    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
    depth_np = depth_img.astype(np.float32) / 255.0
    depth_np = cv2.resize(depth_np, (W, H), interpolation=cv2.INTER_LINEAR)
    dmin, dmax = depth_np.min(), depth_np.max()
    if dmax - dmin > 1e-6:
        depth_np = (depth_np - dmin) / (dmax - dmin)
    depth_np = depth_np.astype(np.float32)

    # convert to torch
    device = DEVICE
    img_t = torch.from_numpy(orig_rgb).float().to(device)        # H W 3
    norm_t = torch.from_numpy(normals_np).float().to(device)     # H W 3
    depth_t = torch.from_numpy(depth_np).float().to(device)      # H W

    img_bhwc = img_t
    norm_bhwc = norm_t
    depth_bhw = depth_t

    # parse color_hex into rgb
    c = color_hex.lstrip("#")
    light_color = torch.tensor([
        int(c[0:2], 16) / 255.0,
        int(c[2:4], 16) / 255.0,
        int(c[4:6], 16) / 255.0
    ], device=device).view(1,1,3)

    # meshgrid
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    xx = xx.float()
    yy = yy.float()

    depth_world = depth_bhw * boost

    lx = max(0, min(W-1, int(round(light_x))))
    ly = max(0, min(H-1, int(round(light_y))))
    # clamp z as float
    lz = float(light_z)

    # ensure indices in range
    depth_at_light = depth_world[ly, lx]
    dx = lx - xx
    dy = ly - yy
    dz = (depth_at_light - depth_world) + lz
    light_vec = torch.stack([dx, dy, dz], dim=-1)
    dist = torch.norm(light_vec, dim=-1, keepdim=True).clamp(min=1e-6)
    light_dir = light_vec / dist

    # normalize normals
    norm_mag = torch.norm(norm_bhwc, dim=-1, keepdim=True).clamp(min=1e-6)
    normals = norm_bhwc / norm_mag

    # Lambert (ndotl)
    ndotl = (normals * light_dir).sum(dim=-1, keepdim=True).clamp(min=0.0)

    # Blinn-Phong spec
    view_dir = torch.tensor([0.0,0.0,1.0], device=device).view(1,1,3)
    half_vec = F.normalize(light_dir + view_dir, dim=-1)
    spec_angle = (normals * half_vec).sum(dim=-1, keepdim=True).clamp(min=0.0)
    spec = spec_angle ** shininess

    # Smooth falloff
    r = (dist.mean().item() + 1.0) * 0.7
    falloff = 1.0 / (1.0 + (dist / r)**2)
    falloff = falloff.clamp(0.0, 1.0)

    # Shadows (simple occlusion)
    occlusion = torch.sigmoid((depth_world - depth_at_light) * shadow_strength + shadow_bias)
    shadow_mask = (1 - occlusion).unsqueeze(-1)

    # Lighting combination
    diffuse = ndotl
    specular = spec
    lighting = (diffuse + specular_weight * specular) * falloff
    lighting = lighting * intensity
    lighting = lighting * (1 - shadow_mask * 0.85)

    final_light = ambient + lighting
    final_light_rgb = final_light * light_color

    relit = img_bhwc * final_light_rgb
    relit = relit.clamp(0, 1)

    relit_np = (relit.cpu().numpy() * 255).astype(np.uint8)
    relit_bgr = cv2.cvtColor(relit_np, cv2.COLOR_RGB2BGR)
    # encode to jpeg bytes
    ret, enc = cv2.imencode('.jpg', relit_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ret:
        raise RuntimeError("Failed to encode result")
    return enc.tobytes()


# -----------------------
# Long-running tasks: process upload into artifacts (depth + normals)
# -----------------------
def process_image_job(job_id: str, img_bytes: bytes):
    """
    Runs depth model and marigold normals on the image, writes artifacts to disk.
    """
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    jobs[job_id] = {"status": "processing"}
    try:
        # Save original
        orig_path = job_dir / "orig.png"
        with open(orig_path, "wb") as f:
            f.write(img_bytes)

        # 1) DepthAnything -> produce depth.npy and depth.png (grayscale)
        # Convert to RGB and transform
        bgr = read_image_bytes_to_bgr(img_bytes)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((518, 518)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # inference -- run on device
        with torch.no_grad():
            input_tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
            depth_pred = depth_model(input_tensor)
            depth_map = depth_pred.squeeze().cpu().numpy()
        # Save depth map as normalized grayscale image
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_path = job_dir / "depth.png"
        cv2.imwrite(str(depth_path), depth_normalized)

        # 2) Marigold normals (diffusion)
        normals_path = job_dir / "normals.png"
        if marigold_pipe is not None:
            # Use PIL for marigold input
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            # run pipeline (this may be slow on CPU)
            pipeline_output = marigold_pipe(pil_img, num_inference_steps=NUM_MARIGOLD_STEPS)
            normal_map = pipeline_output.prediction
            normal_map = np.squeeze(normal_map)
            # possibility shapes: (3,H,W) or (H,W,3)
            if normal_map.ndim == 3 and normal_map.shape[0] == 3:
                normal_map = np.moveaxis(normal_map, 0, 2)
            # convert [-1,1] to [0,255]
            normal_map_uint8 = ((normal_map * 0.5 + 0.5) * 255).clip(0,255).astype(np.uint8)
            Image.fromarray(normal_map_uint8).save(str(normals_path))
        else:
            # fallback: compute normals from depth (cheap approx)
            # load depth and compute gradient to get normals (low quality)
            depth_img = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            dzdx = cv2.Sobel(depth_img, cv2.CV_32F, 1, 0, ksize=5)
            dzdy = cv2.Sobel(depth_img, cv2.CV_32F, 0, 1, ksize=5)
            # approximate normals: (-dzdx, -dzdy, 1)
            normal = np.stack([-dzdx, -dzdy, np.ones_like(depth_img)], axis=-1)
            n = np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-9
            normal = normal / n
            normal_uint8 = ((normal * 0.5 + 0.5) * 255).clip(0,255).astype(np.uint8)
            Image.fromarray(normal_uint8).save(str(normals_path))

        jobs[job_id]["status"] = "ready"
        jobs[job_id]["job_dir"] = str(job_dir.resolve())
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e) + "\n" + traceback.format_exc()
        print("Error processing job:", e)
    return


# -----------------------
# Endpoints
# -----------------------

@app.post("/upload")
async def upload_image(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload an image. Server will process depth + normals in background.
    Returns job_id.
    """
    contents = await file.read()
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued"}
    # save raw quickly
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "orig_uploaded.png").write_bytes(contents)
    # run heavy processing in background
    if background_tasks is not None:
        background_tasks.add_task(run_in_threadpool, process_image_job, job_id, contents)
    else:
        # synchronous fallback (not recommended)
        await run_in_threadpool(process_image_job, job_id, contents)

    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
def job_status(job_id: str):
    info = jobs.get(job_id)
    if info is None:
        raise HTTPException(status_code=404, detail="job_id not found")
    return info

@app.post("/relight")
async def relight(job_id: str, x: float, y: float, z: float,
                  color: Optional[str] = "#ffffff",
                  intensity: Optional[float] = 2.2):
    """
    Recompute relit image for a previously processed job.
    Returns JPEG image bytes.
    """
    info = jobs.get(job_id)
    if info is None or info.get("status") != "ready":
        raise HTTPException(status_code=400, detail="job not ready or not found")
    job_dir = Path(info["job_dir"])

    # Run relight (fast) in threadpool to avoid blocking
    try:
        img_bytes = await run_in_threadpool(relight_from_artifacts, job_dir, x, y, z, color, intensity)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"relight failed: {e}")

    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")

@app.get("/download/{job_id}/{filename}")
def download_artifact(job_id: str, filename: str):
    info = jobs.get(job_id)
    if info is None or info.get("status") not in ("ready",):
        raise HTTPException(status_code=404, detail="job not found or not ready")
    job_dir = Path(info["job_dir"])
    path = job_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return StreamingResponse(path.open("rb"), media_type="application/octet-stream")

@app.delete("/cleanup/{job_id}")
def cleanup_job(job_id: str):
    info = jobs.get(job_id)
    if info is None:
        raise HTTPException(status_code=404, detail="job not found")
    job_dir = Path(info.get("job_dir", ""))
    if job_dir.exists():
        shutil.rmtree(job_dir)
    jobs.pop(job_id, None)
    return {"status": "deleted"}

# -----------------------
# Run server if executed directly
# -----------------------
if __name__ == "__main__":
    # For local dev use: uvicorn server:app --host 0.0.0.0 --port 8000
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

