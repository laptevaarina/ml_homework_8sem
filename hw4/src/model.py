import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tools import euclidean_dist, DEVICE


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()

        self.encoder = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                               out_channels=out_channels,
                                               padding=1,
                                               kernel_size=3),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2),
                                     )

    def forward(self, x):
        x = self.encoder(x)
        return x


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes):
        super(EncoderCNN, self).__init__()

        blocks = [EncoderBlock(in_channels=in_channels,
                               out_channels=out_channels)]

        for i in range(2):
            blocks.append(EncoderBlock(in_channels=out_channels,
                                       out_channels=out_channels))

        blocks.append(EncoderBlock(in_channels=out_channels,
                                   out_channels=n_classes))

        self._blocks = nn.ModuleList(blocks)

        self.neck = nn.Flatten()

    def forward(self, x):
        for block in self._blocks:
            x = block(x)

        out = self.neck(x)
        return out


class ProtoNet(nn.Module):
    def __init__(self, encoder):
        """
        Args:
            encoder : CNN encoding the images in sample
            n_way (int): number of classes in a classification task
            n_support (int): number of labeled examples per class in the support set
            n_query (int): number of labeled examples per class in the query set
        """
        super(ProtoNet, self).__init__()
        self.encoder = encoder.to(DEVICE)

    #         self.criterion = nn.CrossEntropyLoss()

    def parse_feature(self, X, n_support):
        x_support = X[:, :n_support]
        x_query = X[:, n_support:]

        return x_support, x_query

    def set_forward(self, X, n_way, n_support, n_query):
        x_support, x_query = self.parse_feature(X, n_support)

        x_support = x_support.contiguous().view(n_way * n_support,
                                                *x_support.size()[2:])
        x_query = x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])

        x_all = torch.cat([x_support, x_query], 0).to(DEVICE)
        fwd = self.encoder.forward(x_all)

        fwd_dim = fwd.size(-1)
        fwd_proto = fwd[:n_way * n_support].view(n_way, n_support, fwd_dim).mean(1)
        fwd_query = fwd[n_way * n_support:]

        dists = euclidean_dist(fwd_query, fwd_proto)
        scores = -dists
        return scores

    def set_forward_loss(self, sample):
        """
        Computes loss, accuracy and output for classification task
        Args:
            sample (torch.Tensor): shape (n_way, n_support+n_query, (dim))
        Returns:
            torch.Tensor: shape(2), loss, accuracy and y_hat (predict)
        """
        sample_images = sample['images'].to(DEVICE)
        n_way = sample['n_way']
        n_support = sample['n_support']
        n_query = sample['n_query']

        y_query = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
        y_query = Variable(y_query, requires_grad=False).to(DEVICE)

        scores = self.set_forward(sample_images, n_way, n_support, n_query)
        #         loss_val = self.criterion(scores, y_query)

        #         cur_loss = loss_val.detach().cpu().numpy()
        logit = F.log_softmax(scores, dim=1).view(n_way, n_query, -1)
        loss_val = -logit.gather(2, y_query).squeeze().view(-1).mean()
        tmp, y_hat = logit.max(2)
        acc_val = torch.eq(y_hat, y_query.squeeze()).float().mean()
        #         acc_val = f1_score(y_query, pred, average='macro')

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item(),
            'y_hat': y_hat}
