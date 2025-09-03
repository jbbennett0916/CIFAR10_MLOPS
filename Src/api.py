from fastapi import FastAPI, UploadFile, File
from src.inference import predict_image

app = FastAPI(title="CIFAR-10 TinyVGG Classifier")

@app.get("/")
def root():
    return {"status":"ok", "message":"Welcome to the CIFAR-10 TinyVGG Classifier API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    label, confidence = predict_image(img_bytes)
    return {"label":label, "confidence":confidence}







