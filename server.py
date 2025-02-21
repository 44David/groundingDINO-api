from fastapi import FastAPI, Request
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, OwlViTProcessor, OwlViTForObjectDetection
from fastapi.middleware.cors import CORSMiddleware

from grounding_dino import grounding_dino_predict
from owl_vit import owl_vit_predict

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
	CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/prediction/groundingdino")
async def predict(req: Request):
    
	req_json = await req.json()

	req_url = req_json["s3Url"]
	req_text = req_json["prompt"]
	
	grounding_dino_predict(req_url, req_text)


@app.post("/prediction/owl-vit")
async def predict(req: Request):

	req_json = await req.json()
	req_url = req_json["s3Url"]
	req_text = req_json["prompt"]

	owl_vit_predict(req_url, req_text)