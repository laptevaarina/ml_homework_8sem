import time
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
tqdm.get_lock().locks = []

from tools import DEVICE
from model import SiameseNetwork, ArcFaceLoss
from hparams import learning_rate, weight_decay, epochs
from preprocess import make_loaders


def train(model, train_loader, validation_loader, loss_fn, optimizer, learning_rate):
    size = len(train_loader.dataset)
    train_num_batches = len(train_loader)
    val_num_batches = len(validation_loader)
    model.train()
    init_momentum = 0.5

    for epoch in range(epochs):
        total_train_loss, total_val_loss = 0, 0
        curr_momentum = init_momentum + (1 - init_momentum) * (epoch / epochs)

        model.train()
        with tqdm(total=train_num_batches) as progress_bar:
            for batch, (X_train_batch, Y_train_batch) in enumerate(train_loader):
                X_train_batch, Y_train_batch = X_train_batch.to(DEVICE), Y_train_batch.to(DEVICE).to(torch.float)
                image1_train_batch = X_train_batch[:, 0].unsqueeze(1)
                image2_train_batch = X_train_batch[:, 1].unsqueeze(1)

                train_predictions = model(image1_train_batch, image2_train_batch).squeeze()
                # embeddings = torch.cat((output1.unsqueeze(1), output2.unsqueeze(1)), dim=1).to(DEVICE)
                loss = loss_fn(train_predictions, Y_train_batch)  # Y_train_batch.long()
                total_train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.param_groups[0]['betas'] = (curr_momentum, 0.8)
                optimizer.param_groups[0]['lr'] = learning_rate
                optimizer.step()

                progress_bar.update()
                progress_bar.set_description(
                    'Epoch {:d} -- Loss: {:.4f}'.format(epoch + 1, loss.item()))

            avg_train_loss = total_train_loss / train_num_batches
            learning_rate *= 0.99

            model.eval()
            with torch.no_grad():
                for batch, (X_val_batch, Y_val_batch) in enumerate(validation_loader):
                    X_val_batch, Y_val_batch = X_val_batch.to(DEVICE), Y_val_batch.to(DEVICE).to(torch.float)
                    image1_val_batch = X_val_batch[:, 0].unsqueeze(1)
                    image2_val_batch = X_val_batch[:, 1].unsqueeze(1)

                    val_predictions = model(image1_val_batch, image2_val_batch).squeeze()
                    # embeddings_val = torch.cat((output1.unsqueeze(1), output2.unsqueeze(1)), dim=1).to(DEVICE)
                    total_val_loss += loss_fn(val_predictions, Y_val_batch).item()  # Y_val_batch.long()

            curr_avg_val_loss = total_val_loss / val_num_batches

            progress_bar.set_description('Epoch {:d} Train loss = {:.5f}, Val loss = {:.2f}'.format(
                epoch+1, avg_train_loss, curr_avg_val_loss)
            )
            progress_bar.refresh()

            metrics = {'training_loss': avg_train_loss, 'validation_loss': curr_avg_val_loss}
            wandb.log(metrics, step=epoch+1)

    torch.save(model.state_dict(), f"model_siamese_BCEloss.pt")


if __name__ == "__main__":
    model = SiameseNetwork().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                            betas=(0.5, 0.8))
    # loss_fn = ArcFaceLoss(num_classes=2, embedding_size=4096, margin=0.3, scale=1.0).to(DEVICE)
    loss_fn = nn.BCELoss().to(DEVICE)

    train_lfw_dataloader, val_lfw2_dataloader, test_lfw_dataloader = make_loaders()

    wandb.init(project=f"hw5", name="siamese_network_BCE")
    train(model, train_lfw_dataloader, val_lfw2_dataloader, loss_fn, optimizer, learning_rate)
