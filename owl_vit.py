import requests
import torch
import io
from PIL import Image, ImageDraw
from s3 import Upload
import mimetypes
from urllib.parse import urlparse
import os
from transformers import OwlViTProcessor, OwlViTForObjectDetection

def owl_vit_predict(req_url, req_text):
	processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
	model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
	image = Image.open(requests.get(req_url, stream=True).raw)
	print(req_text)
	text_labels = [ req_text ]
	print('text_labels', text_labels)
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
     
		draw = ImageDraw.Draw(image)
  
		xmin = box[0] 
		ymin = box[1]
		xmax = box[2]
		ymax = box[3]
		
		draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=5)
		draw.text((xmin, ymin), f"{text_label}: {round(score.item(), 2)}", fill="white")

	in_mem_file = io.BytesIO()
	image.save(in_mem_file, format=image.format)
	in_mem_file.seek(0)		

	upload = Upload()
	parsed_url = urlparse(req_url)
	filename = os.path.basename(parsed_url.path)
	mime_type, _ = mimetypes.guess_type(filename)

	upload.s3_upload(in_mem_file, filename, mime_type)

	result_s3_url = upload.create_presigned_url(filename)
	
	return result_s3_url