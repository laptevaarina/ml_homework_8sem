from preprocess import extract_sample, read_data, display_sample
from train import load_protonet_conv
from tools import DEVICE
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.manifold import TSNE


def predict_visual(sample, model):
    sample_images = sample['images'].to(DEVICE)
    n_way = sample['n_way']
    n_support = sample['n_support']
    n_query = sample['n_query']

    x_support = sample_images[:, :n_support]
    x_query = sample_images[:, n_support:]

    x_support = x_support.contiguous().view(n_way * n_support,
                                            *x_support.size()[2:])
    x_query = x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])

    x_all = torch.cat([x_support, x_query], 0).to(DEVICE)
    fwd = model.encoder.forward(x_all)

    fwd_dim = fwd.size(-1)
    fwd_proto = fwd[:n_way * n_support].view(n_way, n_support, fwd_dim).mean(1)
    fwd_proto = fwd_proto.cpu().detach().numpy()
    fwd_query = fwd[n_way * n_support:].view(n_way, n_support, fwd_dim).cpu().detach().numpy()
    fwd_proto = np.expand_dims(fwd_proto, axis=1)

    feats = np.concatenate([fwd_proto, fwd_query], axis=1)
    feats = feats.reshape(-1, fwd.shape[-1])

    print('Train TSNE ...')
    tsne = TSNE(n_components=2, perplexity=5, n_jobs=4)
    x_feats = tsne.fit_transform(feats)

    print('Plot labels ...')

    plt.figure(figsize=(12, 10))

    for i in range(n_way):
        l, r = (n_query + 1) * i, (n_query + 1) * (i + 1)
        plt.scatter(x_feats[l:r, 0], x_feats[l:r, 1])

    plt.legend([str(i) for i in range(n_way)])
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _, _, testx, testy = read_data('../images_background/images_background',
                                   '../images_evaluation/images_evaluation')

    model = load_protonet_conv(x_dim=(3, 28, 28),
                               hid_dim=64,
                               z_dim=64)
    model.load_state_dict(torch.load("model_omniglot.pt"))
    model.to(DEVICE)

    n_way = 5
    n_support = 5
    n_query = 5

    my_sample = extract_sample(n_way, n_support, n_query, testx, testy)
    display_sample(my_sample['images'])
    my_loss, my_output = model.set_forward_loss(my_sample)
    print(my_output)

    predict_visual(my_sample, model)
