import io, json, torch
from PIL import Image
from torchvision import transforms
from Src.model import create_model

def load_model(model_path="models/tinyvgg_best.pth", meta_path="models/meta.json", device="cpu"):
    device = torch.device(device)
    ckpt = torch.load(model_path, map_location=device)
    with open(meta_path) as f: meta = json.load(f)
    model = create_model().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean, std, classes = meta["mean"], meta["std"], meta["classes"]
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return model, transform, classes, device

def predict_image(image_bytes):
    model, transform, classes, device = load_model()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    top_idx = probs.argmax().item()
    return classes[top_idx], probs[top_idx].item()
