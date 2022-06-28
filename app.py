from fastapi import FastAPI, File, UploadFile
from starlette.responses import Response
import uvicorn
import time
import torch
from inference_torch import prediction
from preprocess.converter import TokenLabelConverter
from models.model import Model, ViTSTR

model = torch.load('./models/last_model.pth')
converter = TokenLabelConverter()

# Create Fast API
app = FastAPI()

@app.get("/")
async def index():
    return {"messages": "Open the documentations /docs or /redoc"}

@app.post("/text_recogz")
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()
        start_time = time.time()
        pred_str = prediction(model, image, converter, 'api')
        end_time = time.time()

        return {
            "filename": str(file.filename),
            "contentype": str(file.content_type),
            "output text": str(pred_str),
            "inference time": str(end_time - start_time)
        }
    except:
        return Response("Internal server error", status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
