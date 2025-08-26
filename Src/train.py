import torch, json
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
from Src.model import create_model
from Src.data import get_dataloader


def accuracy(outputs, targets):
    """google style accuracy calculation
     Args:
         outputs (torch.Tensor): model outputs
         targets (torch.Tensor): ground truth labels
     Returns:
         float: accuracy
     """
    return (outputs.argmax(dim=1) == targets).float().mean().item()



def train_model(epochs=20, batch_size=32, lr=.001, model_dir="models"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, classes = get_dataloader(batch_size=batch_size)
    # create the model
    model = create_model().to(device)
    # create the loss function and optimizer
    loss_fn = nn.CrossEntroyLoss()
    optimizer = Adam(params=model.parameters(), lr=lr)
    # create the model directory if it doesn't exist
    Path(model_dir).mkdir(exist_ok=True)
    # set the best model accuracy to 0, will be updated during training with the best accuracy
    best_acc = 0

    # training loop
    for epoch in range(epochs):
        model.train()
        tr_loss, tr_acc, steps = 0, 0, 0
        for X,y in tqdm(train_loader):
            X,y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            tr_acc += accuracy(logits, y)
            steps += 1
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

        te_loss /= steps
        te_acc /= steps
        print(f"[{epoch+1}/{epochs}] train_acc={tr_acc:.3f} test_acc={te_acc:.3f}")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({"state_dict":model.state_dict(),"classes":classes},f"{model_dir}/tinyvgg_best.pth")

            with open(f"{model_dir}/meta.json", "w") as f:
                json.dump({"mean": [0.4914,0.4822,0.4465], "std": [0.2470,0.2435,0.2616], "classes": classes}, f)

if __name__ == "__main__":
    train_model()
