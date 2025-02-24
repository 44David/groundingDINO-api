import requests

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from s3 import Upload
import mimetypes
from urllib.parse import urlparse
import os
import io
def grounding_dino_predict(req_url, req_text):

    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    image = Image.open(requests.get(req_url, stream=True).raw)
    text_labels = [req_text] 

    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=0.4,
        text_threshold=0.3,
        target_sizes=[(image.height, image.width)]
    )
    
    result = results[0]
    for box, score, text_label in zip(result["boxes"], result["scores"], result["text_labels"]):
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

