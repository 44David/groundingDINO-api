from fastapi import FastAPI, Request
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, OwlViTProcessor, OwlViTForObjectDetection
from fastapi.middleware.cors import CORSMiddleware

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

	model_id = "IDEA-Research/grounding-dino-base"
	device = "cuda" if torch.cuda.is_available() else "cpu"

	processor = AutoProcessor.from_pretrained(model_id)
	model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

	image = Image.open(requests.get(req_url, stream=True).raw)

	inputs = processor(images=image, text=req_text, return_tensors="pt").to(device)
	with torch.no_grad():
		outputs = model(**inputs)

	results = processor.post_process_grounded_object_detection(
		outputs,
		inputs.input_ids,
		box_threshold=0.4,
		text_threshold=0.3,
		target_sizes=[image.size[::-1]]
	)

	return {"output": results}

@app.post("/prediction/owl-vit")
async def predict(req: Request):

	req_json = await req.json()
	req_url = req_json["s3Url"]
	req_text = req_json["prompt"]
		
	processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
	model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

	image = Image.open(requests.get(req_url, stream=True).raw)
	text_labels = [[ req_text ]]

	inputs = processor(text=text_labels, images=image, return_tensors="pt")
	outputs = model(**inputs)

	# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
	target_sizes = torch.tensor([(image.height, image.width)])

	# Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
	results = processor.post_process_grounded_object_detection(
		outputs=outputs, target_sizes=target_sizes, threshold=0.1, text_labels=text_labels
	)

	# Retrieve predictions for the first image for the corresponding text queries
	result = results[0]

	boxes, scores, text_labels = result["boxes"], result["scores"], result["text_labels"]

	for box, score, text_label in zip(boxes, scores, text_labels):
		box = [round(i, 2) for i in box.tolist()]
		print(f"Detected {text_label} with confidence {round(score.item(), 3)} at location {box}")