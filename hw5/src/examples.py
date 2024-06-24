import cv2
import json
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from model import SiameseDataset, SiameseNetwork
from tools import DEVICE


def display_results(img2, pred, person, i):
    fig = plt.figure(figsize=(5, 2.5))
    t = f'{person}' if pred else f'NOT {person}'
    plt.suptitle(t)
    plt.imshow(img2, cmap='gray')
    plt.savefig(f'../data/result/{person}/result_{person}_{i}')
    # plt.show()


def load_our_dataset(path_to_dataset, person):
    folders = {'Arina': [], 'Sasha': [], 'Nadya': []}
    pairs_dict = {'Image1': [], 'Image2': [], 'Label': []}

    for name in folders:
        for i in range(6):
            image_name = f"{name}_{int(i + 1):04d}.jpg"
            folders[name].append(image_name)

    img_name1 = folders[person][0]
    path_img1 = f'{path_to_dataset}{person}/{img_name1}'
    img1 = cv2.imread(path_img1, cv2.IMREAD_GRAYSCALE)
    img1 = img1.astype('float32') / 255.0
    for name in folders:
        for img_name in folders[name]:
            if img_name == img_name1:
                continue
            pairs_dict['Image1'].append(img1)

            path_img2 = f'{path_to_dataset}{name}/{img_name}'
            img2 = cv2.imread(path_img2, cv2.IMREAD_GRAYSCALE)
            img2 = img2.astype('float32') / 255.0
            pairs_dict['Image2'].append(img2)

            person2 = img_name.split('_')[0]
            label = 1 if person == person2 else 0
            pairs_dict['Label'].append(label)

    return pd.DataFrame(pairs_dict)


def predict_examples(model, test_loader, loss_fn, person):
    test_predictions_list = []
    Y_test_list = []
    correct_predictions, total_test_loss = 0, 0
    size = len(test_loader.dataset)
    num_batches = len(test_loader)

    model.eval()  # Put the model into evaluation mode
    with torch.no_grad():  # Operations inside this block will not be tracked for gradient computation
        for batch, (X_test_batch, Y_test_batch) in enumerate(test_loader):
            X_test_batch, Y_test_batch = X_test_batch.to(DEVICE), Y_test_batch.to(DEVICE).to(torch.float)
            image1_test_batch = X_test_batch[:, 0].unsqueeze(1)
            image2_test_batch = X_test_batch[:, 1].unsqueeze(1)

            test_probabilities = model(image1_test_batch, image2_test_batch).unsqueeze(0)
            # output1, output2 = output1.to(DEVICE), output2.to(DEVICE)
            # embeddings_test = torch.cat((output1.unsqueeze(1), output2.unsqueeze(1)), dim=1).to(DEVICE)
            total_test_loss += loss_fn(test_probabilities, Y_test_batch).item()  # Y_test_batch.long()

            # L1_dist = torch.abs(output1 - output2).to(DEVICE)
            # test_probabilities = fc2(L1_dist).squeeze()

            test_predictions = (test_probabilities >= 0.5).float()
            correct_predictions += (test_predictions == Y_test_batch).sum().item()

            img1 = (image1_test_batch.squeeze(1)[0]*255.0).cpu()
            img2 = (image2_test_batch.squeeze(1)[0]*255.0).cpu()
            display_results(img2, test_predictions, person, batch)

            test_predictions_list.extend(test_predictions.cpu().numpy())
            Y_test_list.extend(Y_test_batch.cpu().numpy())

    accuracy = correct_predictions / size
    curr_avg_test_loss = total_test_loss / num_batches

    with open(f"../data/result/{person}/metrics_{person}.json", "w+") as f:
        json.dump({"accuracy": accuracy, "average loss on test": curr_avg_test_loss}, f)
        print("\n", file=f)

    return accuracy, curr_avg_test_loss


if __name__ == "__main__":
    person = input("Who you want to check? (Arina, Nadya, Sasha): ")
    df_laskipro = load_our_dataset('../data/our_dataset/', person)
    test_laskipro_dataset = SiameseDataset(df_laskipro)
    test_laskipro_dataloader = DataLoader(test_laskipro_dataset, batch_size=1, shuffle=True)

    model = SiameseNetwork()
    model.load_state_dict(torch.load('model_siamese_BCEloss.pt'))
    model.to(DEVICE)

    # loss_fn = ArcFaceLoss(num_classes=2, embedding_size=4096, margin=0.3, scale=1.0).to(DEVICE)
    loss_fn = nn.BCELoss().to(DEVICE)

    accuracy, curr_avg_test_loss = predict_examples(model, test_laskipro_dataloader, loss_fn, person)
    print(accuracy, curr_avg_test_loss)
