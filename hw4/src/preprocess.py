import os
import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


def read_alphabets(alphabet_directory_path):
    """
    Reads all the characters from a given alphabet_directory
    Args:
      alphabet_directory_path (str): path to diretory with files
    Returns:
      datax (np.array): array of path name of images
      datay (np.array): array of labels
    """
    datax = []  # all file names of images
    datay = []  # all class names

    for name in os.listdir(alphabet_directory_path):
        class_original, class_rot90, class_rot180, class_rot270 = [], [], [], []

        path = alphabet_directory_path.split("/")[-2] + "/" + name
        datay += [path]
        datay += [path + "_90"]
        datay += [path + "_180"]
        datay += [path + "_270"]

        for character in os.listdir(alphabet_directory_path + name):
            filename = alphabet_directory_path + name + "/" + character
            image = cv2.resize(cv2.imread(filename), (28, 28))
            image_rot90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            image_rot180 = cv2.rotate(image, cv2.ROTATE_180)
            image_rot270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            class_original.append(image)
            class_rot90.append(image_rot90)
            class_rot180.append(image_rot180)
            class_rot270.append(image_rot270)

        datax += [class_original]
        datax += [class_rot90]
        datax += [class_rot180]
        datax += [class_rot270]

    return np.array(datax), np.array(datay)


def read_images(base_directory):
    """
    Reads all the alphabets from the base_directory
    Uses multithreading to decrease the reading time drastically
    """
    datax = None
    datay = None

    results = [read_alphabets(base_directory + '/' + directory + '/') for directory in os.listdir(base_directory)]

    for result in results:
        if datax is None:
            datax = result[0]
            datay = result[1]
        else:
            datax = np.concatenate([datax, result[0]])
            datay = np.concatenate([datay, result[1]])
    return datax, datay


def read_data(path_to_train, path_to_test):
    trainx, trainy = read_images(path_to_train)
    testx, testy = read_images(path_to_test)
    return trainx, trainy, testx, testy


def extract_sample(n_way, n_support, n_query, datax, datay):
    """
    Picks random sample of size n_support + n_querry, for n_way classes
    Args:
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      datax (np.array): dataset of images
      datay (np.array): dataset of labels
    Returns:
      (dict) of:
        (torch.Tensor): sample of images. Size (n_way, n_support + n_query, (dim))
        (int): n_way
        (int): n_support
        (int): n_query
    """
    sample = []
    K = np.random.choice(np.unique(datay), n_way, replace=False)
    for cls in K:
        datax_cls = datax[datay == cls][0]
        perm = np.random.permutation(datax_cls)
        sample_cls = perm[:(n_support + n_query)]
        sample.append(sample_cls)

    sample = np.array(sample)
    sample = torch.from_numpy(sample).float()
    sample = sample.permute(0, 1, 4, 2, 3)
    return ({
        'images': sample,
        'n_way': n_way,
        'n_support': n_support,
        'n_query': n_query
    })


def display_sample(sample):
    """
    Displays sample in a grid
    Args:
      sample (torch.Tensor): sample of images to display
    """
    #need 4D tensor to create grid, currently 5D
    sample_4D = sample.view(sample.shape[0] * sample.shape[1], *sample.shape[2:])
    #make a grid
    out = torchvision.utils.make_grid(sample_4D, nrow=sample.shape[1])

    plt.figure(figsize=(16, 7))
    plt.imshow(out.permute(1, 2, 0))