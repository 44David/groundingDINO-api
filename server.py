from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from grounding_dino import grounding_dino_predict
from owl_vit import owl_vit_predict

app = FastAPI()


app.add_middleware(
	CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/prediction/groundingdino")
async def predict_groundingdino(req: Request):
    
	req_json = await req.json()

	req_url = req_json["s3Url"]
	req_text = req_json["prompt"]
	
	result_url = grounding_dino_predict(req_url, req_text)
 
	return {"url": result_url}


@app.post("/prediction/owl-vit")
async def predict_owl_vit(req: Request):

	req_json = await req.json()
	req_url = req_json["s3Url"]
	req_text = req_json["prompt"]

	result_url = owl_vit_predict(req_url, req_text)
 
	return {"url": result_url}
	