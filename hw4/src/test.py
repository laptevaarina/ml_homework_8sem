import json
import torch
from tqdm.notebook import tnrange

from train import load_protonet_conv
from preprocess import extract_sample, read_data
from tools import DEVICE
# from hparams import n_way, n_support, n_query


def test(model, test_x, test_y, n_way, n_support, n_query, test_episode):
    """
    Tests the protonet
    Args:
      model: trained model
      test_x (np.array): images of testing set
      test_y (np.array): labels of testing set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      test_episode (int): number of episodes to test on
    """
    running_loss = 0.0
    running_acc = 0.0
    for episode in tnrange(test_episode):
        sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
        loss, output = model.set_forward_loss(sample)
        running_loss += output['loss']
        running_acc += output['acc']

    avg_loss = running_loss / test_episode
    avg_acc = running_acc / test_episode

    with open(f"acc_loss_test.json", "w+") as f:
        json.dump({"loss": avg_loss, "acc": avg_acc}, f)
        print("\n", file=f)

    print('Test results -- Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))


if __name__ == "__main__":
    trainx, trainy, testx, testy = read_data('../images_background/images_background', '../images_evaluation/images_evaluation')

    model = load_protonet_conv(x_dim=(3, 28, 28),
        hid_dim=64,
        z_dim=64)
    model.load_state_dict(torch.load("model_omniglot.pt"))
    model.to(DEVICE)

    n_way = 5
    n_support = 5
    n_query = 5

    test_episode = 1000

    test(model, testx, testy, n_way, n_support, n_query, test_episode)
