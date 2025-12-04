from fastapi import FastAPI
import torch

app = FastAPI()

@app.get("/")
def home():
    mes1 = torch.backends.mps.is_available()
    mes2 = torch.backends.mps.is_built()
    return {"message1": mes1, "message2": mes2}

