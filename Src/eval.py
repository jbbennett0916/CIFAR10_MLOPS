import torch, json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from src.model import create_model
from src.data import get_dataloader


def load_best(model_path="models/tinyvgg_best.pth",meta_path="models/meta.json"):
    """
    Load the best model and its metadata from specified paths.

    Args:
        model_path (str, optional): Path to the saved model checkpoint. Defaults to "models/tinyvgg_best.pth".
        meta_path (str, optional): Path to the metadata JSON file. Defaults to "models/meta.json".

    Returns:
        tuple: A tuple containing the model state dictionary and metadata dictionary.
    """
    # load the model checkpoint (state_dict + metadata like classes)
    ckpt = torch.load(model_path,map_location="cpu")
    # load normalization stats and class labels from meta.json
    with open(meta_path, "r") as f:
        meta=json.load(f)
    # create the model and load the state_dict
    model = create_model()
    model.load_state_dict(ckpt["state_dict"])
    # set the model to evaluation mode
    model.eval()
    # prefer classes from checkpoint if available, else from meta.json
    classes = ckpt.get("classes", meta["classes"])
    return model, classes


def evaluate():
    """
    Evaluates the best saved model on the test dataset:
    - Runs inference
    - Collects predictions and ground truth
    - Prints classification report
    - Plots and saves confusion matrix
    """
    # get the test dataloader
    _, test_loader, classes = get_dataloader(batch_size=32)

    # load the best saved model
    model, _ = load_best()

    y_true, y_pred = [], []

    with torch.inference_mode():
        for X,y in test_loader:
            logits = model(X)
            y_true.extend(y.numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).numpy().tolist())

    # print precision, recall, f1-score per class
    print(classification_report(y_true, y_pred, target_names=classes))

    cm = confusion_matrix(y_true, y_pred)

    # plot and save the confusion matrix
    fig = plt.figure(figsize=(8,8))
    plt.imshow(cm,interpolation="nearest",cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    fig.savefig("models/confusion_matrix.png")
    print("Saved: models/confusion_matrix.png")



if __name__ == "__main__":
    # Entry point: run evaluation when script is executed
    evaluate()