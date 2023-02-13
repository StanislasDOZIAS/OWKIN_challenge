from pathlib import Path
import os
import copy

import wandb

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

from owkin.models.base_model import BaseModel


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RocLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_predict, y):
        y_predict_pos = y_predict[y == 1]
        y_predict_neg = y_predict[y == 0]

        return (
            (
                1 - y_predict_pos.unsqueeze(dim=0) + y_predict_neg.unsqueeze(dim=1)
            ).clamp_min(0)
        ).mean()


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model: BaseModel,
    learning_rate: float = 1e-6,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    nb_epochs: int = 5000,
    criterion: nn.Module = nn.BCELoss(),
    use_wandb: bool = False,
    exp_name: bool = None,
    device: torch.device = torch.device("cuda"),
):
    """
    Launch the training of the model with all theses parameters. The best epoch on the val set will be saved.
    A Wandb log can be launched and a run_name can be set to add information to the experiment.
    """
    model.to(device)
    train_set = CustomDataset(X_train, y_train)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, drop_last=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    run_name = f"{model.name}/{criterion._get_name()}/bs_{batch_size}/wd_{'{:.0e}'.format(weight_decay)}/lr_{'{:.0e}'.format(learning_rate)}"
    if exp_name is not None:
        run_name += f"/{exp_name}"

    try:
        if use_wandb:
            config = copy.deepcopy(model.config)
            config["criterion"] = criterion._get_name()
            config["learning_rate"] = learning_rate
            config["weight_decay"] = weight_decay
            config["batch_size"] = batch_size

            wandb.init(
                project="OWKIN",
                name=run_name,
                config=config,
            )

        else:
            list_train_loss = []
            list_train_roc_auc_score = []
            list_val_roc_auc_score = []

        best_val_roc_auc_score = 0
        best_epoch = 0
        best_state_dict = copy.deepcopy(model.state_dict())

        for epoch in range(1, nb_epochs + 1):
            model.train()
            train_losses = []
            for x, y in train_dataloader:
                optimizer.zero_grad()
                x = x.to(device)
                y_predict = model(x)
                loss = criterion(y_predict.cpu(), y.unsqueeze(dim=1).float())
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            train_loss = np.array(train_losses).mean()

            model.eval()
            train_roc_auc_score = roc_auc_score(
                y_train, model(torch.Tensor(X_train).to(device)).cpu().detach()
            )
            val_roc_auc_score = roc_auc_score(
                y_val, model(torch.Tensor(X_val).to(device)).cpu().detach()
            )

            if val_roc_auc_score > best_val_roc_auc_score:
                best_val_roc_auc_score = val_roc_auc_score
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())

            if use_wandb:
                to_log = {}
                to_log["main/train_loss"] = train_loss
                to_log["main/train_roc_auc_score"] = train_roc_auc_score
                to_log["main/val_roc_auc_score"] = val_roc_auc_score
                wandb.log(to_log)

            else:
                list_train_loss.append(train_loss)
                list_train_roc_auc_score.append(train_roc_auc_score)
                list_val_roc_auc_score.append(val_roc_auc_score)

                if (epoch == 1) or (epoch % (nb_epochs // 10) == 0):
                    print(
                        f"epoch {epoch}: loss={'{:.3f}'.format(train_loss)}, train_roc_auc_score={'{:.3f}'.format(train_roc_auc_score)}, val_auc_roc_auc_score={'{:.3f}'.format(val_roc_auc_score)}"
                    )

    except KeyboardInterrupt:
        pass

    PATH_DIR = Path(f"saved_models/{run_name}")

    PATH = (
        PATH_DIR
        / f"best_epoch_{best_epoch}_score_{'{:.3f}'.format(best_val_roc_auc_score)}.pt"
    )

    model.best_path = PATH

    if not PATH_DIR.is_dir():
        os.makedirs(PATH_DIR)

    torch.save(best_state_dict, PATH)

    if use_wandb:
        wandb.finish()
    else:
        return list_train_loss, list_train_roc_auc_score, list_val_roc_auc_score
