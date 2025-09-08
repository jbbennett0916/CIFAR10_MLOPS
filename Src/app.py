import gradio as gr
from src.inference import predict_image
from PIL import Image
import io


def predict(image):
    if image is None:
        return "No image provided"
    
    # convert PIL image to bytes
    buf = io.BytesIO()
    # save the uploaded image into the buffer in PNG format
    image.save(buf, format="PNG")
    # extract the raw bytes from the buffer
    image_bytes = buf.getvalue()

    # use the predict_image function from inference
    pred_label, prob = predict_image(image_bytes)
    return f"Predicted Label: {pred_label} | Confidence: {prob:.2f}"

# gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload, drag and drop, or paste an image here."),
    outputs="text",
    title="CIFAR-10 TinyVGG Classifier",
    description="The model will predict the class of the uploaded image (one of 10 CIFAR-10 classes).",
)


if __name__ == "__main__":
    demo.launch()