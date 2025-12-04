# **Relighting Server API**

This project provides a FastAPI-based backend for image relighting using PyTorch, OpenCV, and custom depth/normal/lighting models.  

It handles image uploads, runs relighting jobs, stores intermediate results, and serves the processed outputs back to the client.

## **Features**

-   Upload an image and create a new relighting job
    
-   Estimate depth and normals using ML models
    
-   Run relighting with user-controlled light direction
    
-   Download outputs (original, depth, normals, relit)
    
-   Delete job directories to free storage

-   CORS enabled for frontend integration    

-   Background task support for cleanup and long-running jobs

## **Endpoints**

### **1. POST 	`/upload`**

Create a new relighting job.

**Request:**  
`multipart/form-data`

-   `file`: image file
    

**Response:**
```sh
{
  "job_id": "<uuid>",
  "job_dir": "<path>"
}
```

### **2. POST `/relight`**

Run relighting on an existing job.

**Body:**
```sh
{
  "job_id": "<uuid>",
  "x": float,
  "y": float,
  "z": float
}
```
Light direction `(x, y, z)` controls how the image is relit.

**Response:**
```sh
{
  "status": "ok",
  "relit_path": "<path>"
}
```

### **3. GET `/download/{job_id}/{filename}`**

Download any output file from the job directory.  
Valid filenames include:

-   `input.png`
    
-   `depth.png`
    
-   `normals.png`
    
-   `relit.png`

### **4. DELETE `/job/{job_id}`**

Delete an entire job directory.

**Response:**
```sh
{
  "status": "deleted"
}
```
### Project Structure
```sh
main.py
server.py
requirements.txt
README.md
jobs/
    <generated job folders>
```
## **Running Locally**

### **Install dependencies**
```sh
pip install -r requirements.txt
```
Make sure PyTorch is installed with the backend you need (CPU/MPS/CUDA).

### **Run**
```sh
uvicorn server:app --host 0.0.0.0 --port 8000
```
## **Environment Notes**

-   Requires Python 3.9+
    
-   Uses FastAPI, PyTorch, OpenCV, Pillow
    
-   Automatically manages job storage under `jobs/`
    
-   Uses UUIDs for job isolation
    
## **Error Handling**

The API returns consistent JSON errors for:

-   Missing job ID
    
-   Missing or corrupt files
    
-   Model inference errors
    
-   File not found during download


