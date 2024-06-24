import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.nn import functional as F

from tools import DEVICE


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_size, margin, scale):
        """
        ArcFace: Additive Angular Margin Loss for Deep Face Recognition
        (https://arxiv.org/pdf/1801.07698.pdf)
        Args:
            num_classes: The number of classes in your training dataset
            embedding_size: The size of the embeddings that you pass into
            margin: m in the paper, the angular margin penalty in radians
            scale: s in the paper, feature scale
        """
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale

        self.W = torch.nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_normal_(self.W)

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (None, embedding_size)
            labels: (None,)
        Returns:
            loss: scalar
        """
        cosine = self.get_cosine(embeddings)  # (None, n_classes)
        mask = self.get_target_mask(labels)  # (None, n_classes)
        cosine_of_target_classes = cosine[mask == 1]  # (None, )
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(
            cosine_of_target_classes
        )  # (None, )
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes)  # (None,1)
        logits = cosine_of_target_classes + (mask * diff)  # (None, n_classes)
        logits = self.scale_logits(logits)  # (None, n_classes)
        return nn.CrossEntropyLoss()(logits, labels)

    def get_cosine(self, embeddings):
        """
        Args:
            embeddings: (None, embedding_size)
        Returns:
            cosine: (None, n_classes)
        """
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.W))
        return cosine

    def get_target_mask(self, labels):
        """
        Args:
            labels: (None,)
        Returns:
            mask: (None, n_classes)
        """
        batch_size = labels.size(0)
        onehot = torch.zeros(batch_size, self.num_classes, device=labels.device)
        onehot.scatter_(1, labels.unsqueeze(-1), 1)
        return onehot

    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        """
        Args:
            cosine_of_target_classes: (None,)
        Returns:
            modified_cosine_of_target_classes: (None,)
        """
        eps = 1e-6
        # theta in the paper
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1 + eps, 1 - eps))
        return torch.cos(angles + self.margin)

    def scale_logits(self, logits):
        """
        Args:
            logits: (None, n_classes)
        Returns:
            scaled_logits: (None, n_classes)
        """
        return logits * self.scale


class SiameseDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        tensor_images = []
        for col in dataframe.columns[:-1]:
            tensor_images.append(torch.tensor(dataframe[col].tolist(), dtype=torch.float))
        self.x = torch.stack(tensor_images, dim=1)  # When dim=1 the tensors are transposed and stacked along the column
        self.labels = torch.tensor(dataframe.iloc[:, -1].tolist(), dtype=torch.float)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.labels[idx]
        if self.transform:
            augmented_image = self.transform(image)
            return augmented_image, label
        else:
            return image, label


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(10, 10), stride=1, padding=0),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(256, 256, kernel_size=7),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
                    nn.Linear(256 * 3 * 3, 4096),
                    nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
                    nn.Linear(4096, 1),
                    nn.Sigmoid()
        )

        self.W = torch.nn.Parameter(torch.Tensor(2, 4096)).to(DEVICE)
        nn.init.xavier_normal_(self.W)

    def get_cosine(self, embeddings):
        """
        Args:
            embeddings: (None, embedding_size)
        Returns:
            cosine: (None, n_classes)
        """
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.W))
        return cosine

    def forward_once(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        feature_vector = self.fc1(x)
        return feature_vector

    def forward(self, x1, x2):
        y1 = self.forward_once(x1)
        y2 = self.forward_once(x2)
        # return y1, y2
        L1_dist = torch.abs(y1 - y2)
        L1_weighted_dist = self.fc2(L1_dist)
        return L1_weighted_dist.squeeze()


