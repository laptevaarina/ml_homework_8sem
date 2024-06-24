import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset

from model import SiameseDataset
from hparams import batch_size

DATA_PATH = '../data/'
LFW_PATH = '../data/lfw/'
TRAIN_PATH = '../data/pairsDevTrain.txt'
TEST_PATH = '../data/pairsDevTest.txt'


def read_txt_file_and_determine_size(file_path):
    with open(file_path, 'r') as data:
        data_lines = data.readlines()
        samples = data_lines[1:]
        num_of_identical = int(data_lines[0])
        num_of_different = len(samples) - num_of_identical
        return samples, num_of_identical, num_of_different


def convert_pairs_path_to_df(data, is_identical):
    pairs_dict = {'Image1': [], 'Image2': [], 'Label': []}
    if is_identical:
        for pair in data:
            img1 = cv2.imread(pair[0], cv2.IMREAD_GRAYSCALE)
            img1 = img1.astype('float32') / 255.0
            img2 = cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)
            img2 = img2.astype('float32') / 255.0

            pairs_dict['Image1'].append(img1)
            pairs_dict['Image2'].append(img2)
            pairs_dict['Label'].append(1)
    else:
        for pair in data:
            img1 = cv2.imread(pair[0], cv2.IMREAD_GRAYSCALE)
            img1 = img1.astype('float32') / 255.0
            img2 = cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)
            img2 = img2.astype('float32') / 255.0

            pairs_dict['Image1'].append(img1)
            pairs_dict['Image2'].append(img2)
            pairs_dict['Label'].append(0)

    return pd.DataFrame(pairs_dict)


def create_images_path(rows, is_identical):
    image_pairs_path = []
    for i in range(len(rows)):
        row_components = rows[i].split()
        folder_name = LFW_PATH + row_components[0] + '/'
        image_name = f"{row_components[0]}_{int(row_components[1]):04d}.jpg"  # Pad the image number with leading zeros
        image1_path = folder_name + image_name

        if is_identical:
            image_name = f"{row_components[0]}_{int(row_components[2]):04d}.jpg"
            image2_path = folder_name + image_name
        else:
            folder_name = LFW_PATH + row_components[2] + '/'
            image_name = f"{row_components[2]}_{int(row_components[3]):04d}.jpg"
            image2_path = folder_name + image_name
        image_pairs_path.append(np.array([image1_path, image2_path]))

    return image_pairs_path


def load_lfw_dataset():
    """
    Loads and preprocesses the lfw2 dataset.

    Returns:
        train_pairs_df (pd.DataFrame): Training DataFrame (including identical and different individuals).
        test_pairs_df (pd.DataFrame): Test DataFrame (including identical and different individuals).
    """

    X_training_data, num_of_train_identical, num_of_train_different = read_txt_file_and_determine_size(TRAIN_PATH)
    X_test_data, num_of_test_identical, num_of_test_different = read_txt_file_and_determine_size(TEST_PATH)

    X_train_identical_pairs = create_images_path(X_training_data[:num_of_train_identical], True)
    X_train_different_pairs = create_images_path(X_training_data[num_of_train_identical:], False)
    X_test_identical_pairs = create_images_path(X_test_data[:num_of_test_identical], True)
    X_test_different_pairs = create_images_path(X_test_data[num_of_test_identical:], False)

    identical_pairs_df = convert_pairs_path_to_df(X_train_identical_pairs, True)
    different_pairs_df = convert_pairs_path_to_df(X_train_different_pairs, False)
    train_pairs_df = pd.concat([identical_pairs_df, different_pairs_df], ignore_index=True)

    identical_pairs_df = convert_pairs_path_to_df(X_test_identical_pairs, True)
    different_pairs_df = convert_pairs_path_to_df(X_test_different_pairs, False)
    test_pairs_df = pd.concat([identical_pairs_df, different_pairs_df], ignore_index=True)

    return train_pairs_df, test_pairs_df


def make_loaders():
    train_pairs_df, test_pairs_df = load_lfw_dataset()

    # Original training set
    original_train_dataset = SiameseDataset(train_pairs_df)

    # Augmented training set
    augmented_train_dataset = SiameseDataset(train_pairs_df, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=30),
    ]))

    combined_train_lfw_dataset = ConcatDataset([original_train_dataset, augmented_train_dataset])

    val_size = int(0.1 * len(combined_train_lfw_dataset))
    indices = list(range(len(combined_train_lfw_dataset)))

    random_state = 42
    np.random.seed(random_state)
    np.random.shuffle(indices)

    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_dataset = Subset(combined_train_lfw_dataset, train_indices)  # Create Subset datasets using the indices
    train_lfw_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_lfw2_dataset = Subset(combined_train_lfw_dataset, val_indices)
    val_lfw2_dataloader = DataLoader(val_lfw2_dataset, batch_size=batch_size, shuffle=True)

    test_lfw_dataset = SiameseDataset(test_pairs_df)
    test_lfw_dataloader = DataLoader(test_lfw_dataset, batch_size=batch_size, shuffle=True)

    return train_lfw_dataloader, val_lfw2_dataloader, test_lfw_dataloader
