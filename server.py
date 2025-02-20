from fastapi import FastAPI
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import requests
from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

app = FastAPI()

@app.get("/prediction/groundingdino")
def predict():
		model_id = "IDEA-Research/grounding-dino-base"
		device = "cuda" if torch.cuda.is_available() else "cpu"

		processor = AutoProcessor.from_pretrained(model_id)
		model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

		image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(image_url, stream=True).raw)
		# Check for cats and remote controls
		# VERY important: text queries need to be lowercased + end with a dot
		text = "a cat. a remote control."

		inputs = processor(images=image, text=text, return_tensors="pt").to(device)
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

@app.get("/prediction/owl-vit")
def predict():
	model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

	url = "http://images.cocodataset.org/val2017/000000039769.jpg"

	image = Image.open(requests.get(url, stream=True).raw)

	text_labels = [["a photo of a cat", "a photo of a dog"]]

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