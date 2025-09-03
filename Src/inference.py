import io, json, torch
from PIL import Image
from torchvision import transforms
from src.model import create_model

def load_model(model_path="models/tinyvgg_best.pth", meta_path="models/meta.json", device="cpu"):
    device = torch.device(device)
    # load the trained weights from the checkpoint file
    ckpt = torch.load(model_path, map_location=device)
    # reads the meta data stored in the json file
    with open(meta_path) as f: meta = json.load(f)
    # create the TinyVGG model
    model = create_model().to(device)
    # loads learned parameters into the model
    model.load_state_dict(ckpt["state_dict"])
    # set the model to inference mode
    model.eval()
    mean, std, classes = meta["mean"], meta["std"], meta["classes"]
    # transform pipeline for the input image
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # return everything needed for prediction
    return model, transform, classes, device


def predict_image(image_bytes):
    # loads the model and other preprocessing tools
    model, transform, classes, device = load_model()
    # reads a image from raw bytes and ensure it has 3 color channels (RGB)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # converts the image into a C,H,W Tensor and PyTorch expects a batch dimension in the form of N,C,H,W
    # unsqueeze adds a batch dimension to the front of the tensor
    x = transform(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        # runs the image through the model to get raw predictions (logits)
        logits = model(x)
        # converts the logits into probabilities. removes the batch dimension.
        # after softmax, probs still includes the batch dimension, so we use squeeze to remove it
        probs = torch.softmax(logits, dim=1).squeeze(0)
    # finds the index of the class with the highest probability
    top_idx = probs.argmax().item()
    # returns the predicted class and the associated probability
    return classes[top_idx], probs[top_idx].item()
