import json
import torch
import torch.nn as nn

from tools import DEVICE
from model import SiameseNetwork, ArcFaceLoss
from preprocess import make_loaders

fc2 = nn.Sequential(
                nn.Linear(4096, 1),
                nn.Sigmoid()
            ).to(DEVICE)


def predict(model, test_loader, loss_fn):
    test_predictions_list = []
    Y_test_list = []
    correct_predictions, total_test_loss = 0, 0
    size = len(test_loader.dataset)
    num_batches = len(test_loader)

    model.eval()  # Put the model into evaluation mode
    with torch.no_grad():  # Operations inside this block will not be tracked for gradient computation
        for batch, (X_test_batch, Y_test_batch) in enumerate(test_loader):
            print(Y_test_batch)
            X_test_batch, Y_test_batch = X_test_batch.to(DEVICE), Y_test_batch.to(DEVICE).to(torch.float)
            print(X_test_batch.shape, X_test_batch)
            image1_test_batch = X_test_batch[:, 0].unsqueeze(1)
            image2_test_batch = X_test_batch[:, 1].unsqueeze(1)

            test_probabilities = model(image1_test_batch, image2_test_batch).squeeze()
            # output1, output2 = output1.to(DEVICE), output2.to(DEVICE)
            # embeddings_test = torch.cat((output1.unsqueeze(1), output2.unsqueeze(1)), dim=1).to(DEVICE)
            total_test_loss += loss_fn(test_probabilities, Y_test_batch).item()  # Y_test_batch.long()

            # L1_dist = torch.abs(output1 - output2).to(DEVICE)
            # test_probabilities = fc2(L1_dist).squeeze()

            test_predictions = (test_probabilities >= 0.5).float()
            correct_predictions += (test_predictions == Y_test_batch).sum().item()
            test_predictions_list.extend(test_predictions.cpu().numpy())
            Y_test_list.extend(Y_test_batch.cpu().numpy())

    accuracy = correct_predictions / size
    curr_avg_test_loss = total_test_loss / num_batches

    with open(f"metrics_BCE.json", "w+") as f:
        json.dump({"accuracy": accuracy, "average loss on test": curr_avg_test_loss}, f)
        print("\n", file=f)

    return accuracy, curr_avg_test_loss


if __name__ == "__main__":
    train_lfw_dataloader, val_lfw2_dataloader, test_lfw_dataloader = make_loaders()

    model = SiameseNetwork()
    model.load_state_dict(torch.load('model_siamese_BCEloss.pt'))
    model.to(DEVICE)

    # loss_fn = ArcFaceLoss(num_classes=2, embedding_size=4096, margin=0.3, scale=1.0).to(DEVICE)
    loss_fn = nn.BCELoss().to(DEVICE)

    accuracy, curr_avg_test_loss = predict(model, test_lfw_dataloader, loss_fn)
    print(accuracy, curr_avg_test_loss)

