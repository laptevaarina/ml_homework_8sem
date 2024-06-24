import wandb
import torch
import torch.optim as optim
from tqdm.notebook import tnrange
from tqdm import tqdm

from model import EncoderCNN, ProtoNet
from preprocess import extract_sample, read_data
from tools import DEVICE
from hparams import max_epoch, epoch_size, n_way, n_support, n_query, learning_rate

tqdm.get_lock().locks = []


def load_protonet_conv(**kwargs):
    """
    Loads the prototypical network model
    Arg:
      x_dim (tuple): dimension of input image
      hid_dim (int): dimension of hidden layers in conv blocks
      z_dim (int): dimension of embedded image
    Returns:
      Model (Class ProtoNet)
    """
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    encoder = EncoderCNN(x_dim[0], hid_dim, z_dim).to(DEVICE)

    return ProtoNet(encoder).to(DEVICE)


def train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size):
    """
    Trains the protonet
    Args:
      model
      optimizer
      train_x (np.array): images of training set
      train_y(np.array): labels of training set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      max_epoch (int): max epochs to train on
      epoch_size (int): episodes per epoch
    """
    # divide the learning rate by 2 at each epoch, as suggested in paper
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
    epoch = 0  # epochs done so far
    stop = False  # status to know when to stop

    while epoch < max_epoch and not stop:
        running_loss = 0.0
        running_acc = 0.0

        with tqdm(total=epoch_size) as progress_bar:
            for episode in range(epoch_size):
                sample = extract_sample(n_way, n_support, n_query, train_x, train_y)
                optimizer.zero_grad()
                # print(model.encoder.device, sample.device)
                loss, output = model.set_forward_loss(sample)
                running_loss += output['loss']
                running_acc += output['acc']
                loss.backward()
                optimizer.step()

                progress_bar.update()
                progress_bar.set_description('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'. format(epoch + 1, output['loss'], output['acc']))

        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size

        progress_bar.set_description('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, epoch_loss, epoch_acc))
        progress_bar.refresh()

        epoch += 1
        scheduler.step()

        metrics = {'learning_rate': scheduler.get_last_lr(), 'epoch_loss': epoch_loss, 'epoch_acc': epoch_acc}
        wandb.log(metrics, step=epoch)

    torch.save(model.state_dict(), f"model_omniglot.pt")


if __name__ == "__main__":
    trainx, trainy, testx, testy = read_data('../images_background/images_background', '../images_evaluation/images_evaluation')

    model = load_protonet_conv(
        x_dim=(3, 28, 28),
        hid_dim=64,
        z_dim=64,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_x = trainx
    train_y = trainy

    wandb.init(project=f"hw4", name="first_prototype")
    train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size)
