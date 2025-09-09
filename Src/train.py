import torch, json
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
from src.model import create_model
from src.data import get_dataloader
import mlflow.pytorch


def accuracy(outputs, targets):
    """Calculate classification accuracy.

    Args:
        outputs (torch.Tensor): raw model outputs (logits) with shape [N, num_classes]
        targets (torch.Tensor): true class labels with shape [N]

    Returns:
        float: accuracy value between 0 and 1
    """
    return (outputs.argmax(dim=1) == targets).float().mean().item()



def train_model(epochs=20, batch_size=32, lr=.001, model_dir="models"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, classes = get_dataloader(batch_size=batch_size)
    # create the model
    model = create_model().to(device)
    # create the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=lr)
    # create the model directory if it doesn't exist
    Path(model_dir).mkdir(exist_ok=True)
    # set the best model accuracy to 0, will be updated during training with the best accuracy
    best_acc = 0

    # -- mlflow: start tracking this run --
    with mlflow.start_run():
        # log hyperparameters
        mlflow.log_param("epochs",epochs)
        mlflow.log_param("batch_size",batch_size)
        mlflow.log_param("learning_rate",lr)
        # training loop
        for epoch in range(epochs):
            model.train()
            # set the training loss, accuracy and steps to 0 at the start of each epoch
            tr_loss, tr_acc, steps = 0, 0, 0
            # iterate over the training batches
            for X,y in tqdm(train_loader):
                # move the data to the device, #
                X,y = X.to(device), y.to(device)
                # Forward Pass: compute the model predictions
                logits = model(X)
                # calculate the loss
                loss = loss_fn(logits, y)
                optimizer.zero_grad()
                loss.backward()
                # update the model weights
                optimizer.step()
                # update the training loss and accuracy for eac
                tr_loss += loss.item()
                tr_acc += accuracy(logits, y)
                steps += 1
            # average training loss and accuracy over training batches
            tr_loss /= steps
            tr_acc /= steps

            # Eval loop
            model.eval()
            te_loss, te_acc, steps = 0,0,0
            with torch.inference_mode():
                for X,y in test_loader:
                    X,y = X.to(device), y.to(device)
                    logits = model(X)
                    loss = loss_fn(logits, y)
                    te_loss += loss.item()
                    te_acc += accuracy(logits, y)
                    steps += 1
            # average testing loss and accuracy over testing batches
            te_loss /= steps
            te_acc /= steps
            print(f"[{epoch+1}/{epochs}] train_acc={tr_acc:.3f} test_acc={te_acc:.3f}")

            # -- mlflow: log metrics for this epoch --
            mlflow.log_metric("train_loss", tr_loss, step=epoch)
            mlflow.log_metric("train_acc", tr_acc, step=epoch)
            mlflow.log_metric("test_loss", te_loss, step=epoch)
            mlflow.log_metric("test_acc",te_acc, step=epoch)

            # saves the best model
            if te_acc > best_acc:
                best_acc = te_acc
                # saves model checkpoints (weights and classes)
                torch.save({"state_dict":model.state_dict(),"classes":classes},f"{model_dir}/tinyvgg_best.pth")

                # saves meta data for preprocessing and label mapping at inference time
                with open(f"{model_dir}/meta.json", "w") as f:
                    json.dump({"mean": [0.4914,0.4822,0.4465], "std": [0.2470,0.2435,0.2616], "classes": classes}, f)

                # --mlflow: log the best model --
                mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    train_model()
