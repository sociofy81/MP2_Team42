#!/usr/bin/env python3
import uvicorn
import os
import sys

if __name__ == "__main__":
    # Check if mobile_sam.pt exists
    if not os.path.exists("mobile_sam.pt"):
        print("❌ Error: mobile_sam.pt not found!")
        print("Please download MobileSAM checkpoint and place it in the current directory.")
        print("Download from: https://github.com/ChaoningZhang/MobileSAM")
        sys.exit(1)
    
    # Create jobs directory
    os.makedirs("jobs", exist_ok=True)
    
    print("🚀 Starting Segmentation & Inpainting API...")
    print("📁 Jobs directory: ./jobs")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("\nTo stop the server, press Ctrl+C")
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",      # Allows external connections
        port=8000,           # Standard port
        reload=False,        # Set to True for development
        workers=1,           # Increase for production
        log_level="info"
    )