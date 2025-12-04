# 🔦 AI Image Relighting Engine

This project enables **real-time relighting of single 2D images** using AI-generated geometry cues: **depth 
maps and surface normals**. The system preprocesses an image once using ML models, then allows users to freely modify lighting parameters (position, color, intensity, specularity) with millisecond response.

---

## 🚀 Features

- 📥 Upload any RGB image  
- 🧠 Automatic depth estimation using **Depth Anything V2**  
- 🎨 Surface normal estimation using **Marigold Normals (Diffusion)**  
- 💡 Real-time relighting with:
  - Point light position (x, y, z)
  - Color & intensity
  - Ambient + specular tuning
  - Soft shadows & falloff
- 🗂 Download processing artifacts
- 🧹 Automatic job cleanup

---

## ⚙️ Pipeline Breakdown

| Stage | Component | Model / Method | Frequency | Output |
|-------|-----------|----------------|-----------|--------|
| Upload Image | I/O | File ingestion | Once per job | `orig.png` |
| Depth Prediction | Depth Anything V2 (ViT-L) | Transformer inference | Once | `depth.png` |
| Normal Prediction | Marigold Normals (Diffusion) | UNet + DDIM | Once | `normals.png` |
| Relighting | Physically-based shading | No ML | Multiple | JPEG preview |

### Why a Two-Stage Pipeline?

Depth and normal estimation are computationally expensive, while relighting is fast.  
So we compute geometry **once**, then allow infinite interactive relighting cycles.

---

## 🧮 Compute Profile

| Component | Model Size | CPU Runtime | Memory Usage | Notes |
|-----------|------------|-------------|--------------|-------|
| Depth Anything V2 | ~350M params | **2–5s** | >2GB | Reasonable trade-off between accuracy and speed |
| Marigold Normals | ~1.1B params | **5-10s** | >4GB | Diffusion-based — high quality, slow |
| Relighting Engine | No ML | **30–90ms** | Low | Runs in real time |

> On GPU: Marigold inference drops from **30s → <1s**, enabling near real-time workflow.

---

## 🧠 Runtime Design Decisions

| System Choice | Decision | Reason |
|---------------|----------|--------|
| Model Runtime | CPU-first | Works on commodity machines |
| Async Processing | Background threadpool | Prevents blocking upload calls |
| Geometry Persistence | Job-based filesystem storage | Avoids recomputing expensive inference |
| Normal Fallback | Sobel normal estimation | Ensures robustness if model unavailable |
| Lighting Model | Lambertian + Blinn-Phong + shadow mask | Good balance of realism & speed |
| File Format Strategy | Lossless PNG for artifacts, JPEG for output | Quality + performance |

---

## 🧰 API Overview

| Endpoint | Purpose |
|----------|---------|
| `POST /upload` | Upload image & start processing |
| `GET /status/{job_id}` | Poll inference progress |
| `POST /relight` | Generate new relighted preview |
| `GET /download/{job_id}/{filename}` | Retrieve depth/normal artifacts |
| `DELETE /cleanup/{job_id}` | Delete stored data |

---
## 🚂 Setup
- Place the extracted folder for relighting `Relighting_Backend` in a separate folder 
- Clone Depth-Anything-V2 (`https://github.com/DepthAnything/Depth-Anything-V2`) into a folder titled `Depth-Anything-V2` 
- Create a folder titled `checkpoints` and place the following .vitl model file downloaded from this link: `https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true`
- Specify the directory addresses in the code file where commented in the `server.py`  file (in `CONFIGURATION` comment block at the top)
## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🔨 End-to-End Workflow

### Step 1: Run backend

```sh
uvicorn server:app
```

### Step 2: Open frontend and upload an image  
Backend performs:
- Save orig.png
- Compute depth.png
- Compute normals.png
- Report status via `/status/{job_id}`

### Step 3: Relight interactively  
Frontend calls `/relight` repeatedly; previews are generated in 30–90ms.

### Step 4: Download artifacts

```sh
GET /download/{job_id}/depth.png  
GET /download/{job_id}/normals.png
```

### Step 5: Cleanup
```sh
DELETE /cleanup/{job_id}
```

------------------------------------------------------------

## ☢️ Error Handling & Common Issues

### 422 Unprocessable Entity on /relight  
- You missed required JSON fields:
```sh
{
  "job_id": "...",
  "x": 0.0,
  "y": 0.0,
  "z": 50.0,
}
```
### 404 on /download  
- Job folder doesn't exist or was already cleaned.

### Marigold fails to load  
- Memory insufficient. System auto-falls back to Sobel normals.

------------------------------------------------------------

## 🧑‍💻 Development Notes

- Heavy inference runs in background threads
- Geometry is cached per job
- Relighting uses analytic shading instead of ML
- Frontend polls /status before enabling relight UI
- Cleanup is manual, but can be automated

------------------------------------------------------------

## ⚠️ Caution
- It is advised to change the IP address on the frontend once the server's IP address has changed.