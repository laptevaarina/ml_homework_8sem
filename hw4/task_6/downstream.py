import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tools import DEVICE
import wandb
from tqdm import tqdm
from add_data.simCLR import ProjectionHead, BaseTrainProcess, PreModel
from add_data.load import load_datasets, set_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from cifar import load_cifar10
import numpy as np


class ClsfModel(nn.Module):
    def __init__(self, encoder, emb_size):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.projector = ProjectionHead(emb_size, 2048, 10)

    def forward(self, x):
        out = self.encoder(x)
        xp = self.projector(torch.squeeze(out))
        return xp


class ClsfTrainProcess:
    def __init__(self, hyp, model, train_dataset, valid_dataset, name):
        self.best_loss = 1e100
        self.best_acc = 0.0
        self.current_epoch = -1
        self.device = DEVICE
        self.name = name

        self.hyp = hyp
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.model = model.to(self.device)

        self.init_params()

    def _init_data(self):
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.hyp['batch_size'],
                                       shuffle=True,
                                       num_workers=self.hyp['n_workers'],
                                       pin_memory=True,
                                       drop_last=True
                                       )

        self.valid_loader = DataLoader(self.valid_dataset,
                                       batch_size=self.hyp['batch_size'],
                                       shuffle=True,
                                       num_workers=self.hyp['n_workers'],
                                       pin_memory=True,
                                       drop_last=True
                                       )

    def _init_model(self):
        model_params = [params for params in self.model.parameters() if params.requires_grad]
        self.optimizer = torch.optim.AdamW(model_params, lr=self.hyp['lr'], weight_decay=self.hyp['weight_decay'])

        # # "decay the learning rate with the cosine decay schedule without restarts"
        self.warmupscheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: (epoch + 1) / 10.0)
        self.mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            500,
            eta_min=0.05,
            last_epoch=-1,
        )

        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def init_params(self):
        self._init_data()
        self._init_model()

    def train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        self.model.zero_grad()

        cum_loss = 0.0
        proc_loss = 0.0
        cum_acc = 0.0
        proc_acc = 0.0

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc=f'Train {self.current_epoch}/{self.hyp["epochs"]}')
        for idx, (xi, xj, l, im) in pbar:
            xi, l = xi.to(self.device), l.to(self.device)

            with torch.set_grad_enabled(True):
                out = self.model(xi.float())
                loss = self.criterion(out, l)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.model.zero_grad()

            cur_loss = loss.detach().cpu().numpy()
            cum_loss += cur_loss
            proc_loss = (proc_loss * idx + cur_loss) / (idx + 1)

            _, pred = torch.softmax(out.detach(), dim=1).topk(k=1)
            acc = accuracy_score(l.detach().cpu(), pred.detach().cpu())
            cum_acc += acc
            proc_acc = (proc_acc * idx + acc) / (idx + 1)

            s = f'Train {self.current_epoch + 1}/{self.hyp["epochs"]}, Loss: {proc_loss:4.3f}, Acc: {proc_acc:4.3f}'
            pbar.set_description(s)

        cum_loss /= len(self.train_loader)
        cum_acc /= len(self.train_loader)

        metrics = {f'train_loss {self.name}': cum_loss, f'train_acc {self.name}': cum_acc}
        wandb.log(metrics, step=self.current_epoch)

        return [cum_loss, cum_acc]

    def valid_step(self):
        self.model.eval()

        cum_loss = 0.0
        proc_loss = 0.0
        cum_acc = 0.0
        proc_acc = 0.0

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader),
                    desc=f'Valid {self.current_epoch}/{self.hyp["epochs"] - 1}')
        for idx, (xi, xj, l, im) in pbar:
            xi, l = xi.to(self.device), l.to(self.device)

            with torch.set_grad_enabled(False):
                out = self.model(xi.float())
                loss = self.criterion(out, l)

            cur_loss = loss.detach().cpu().numpy()
            cum_loss += cur_loss
            proc_loss = (proc_loss * idx + cur_loss) / (idx + 1)

            _, pred = torch.softmax(out.detach(), dim=1).topk(k=1)
            acc = accuracy_score(l.detach().cpu(), pred.detach().cpu())
            cum_acc += acc
            proc_acc = (proc_acc * idx + acc) / (idx + 1)

            s = f'Valid {self.current_epoch + 1}/{self.hyp["epochs"]}, Loss: {proc_loss:4.3f}, Acc: {proc_acc:4.3f}'
            pbar.set_description(s)

        cum_loss /= len(self.train_loader)
        cum_acc /= len(self.train_loader)

        metrics = {f'val_loss {self.name}': cum_loss, f'val_acc {self.name}': cum_acc}
        wandb.log(metrics, step=self.current_epoch)

        return [cum_loss, cum_acc]

    def run(self):
        train_losses = []
        valid_losses = []

        for epoch in range(self.hyp['epochs']):
            self.current_epoch = epoch

            loss_train = self.train_step()
            train_losses.append(loss_train)

            if epoch < 10:
                self.warmupscheduler.step()
            else:
                self.mainscheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]

            loss_valid = self.valid_step()
            valid_losses.append(loss_valid)

        torch.cuda.empty_cache()

        return train_losses, valid_losses


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10("cifar_data", channels_last=True)
    train_dataset, valid_dataset = load_datasets(X_train, y_train, X_val, y_val, crop_coef=1.4)
    print('Train size:', len(train_dataset), 'Valid size:', len(valid_dataset))

    with open('hyp_params.yaml', 'r') as f:
        hyps = yaml.load(f, Loader=yaml.SafeLoader)
    set_seed(hyps['seed'])

    trainer = BaseTrainProcess(hyps, train_dataset, valid_dataset)
    train_losses, valid_losses = trainer.run()

    X_train, X_val, y_train, y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=42)
    train_dataset, valid_dataset = load_datasets(X_train, y_train, X_val, y_val, crop_coef=1.4)

    model = ClsfModel(trainer.model.encoder, trainer.model.emb_size)

    with open('hyp_params_clsf.yaml', 'r') as f:
        hyps = yaml.load(f, Loader=yaml.SafeLoader)
    set_seed(hyps['seed'])

    trainer_clsf = ClsfTrainProcess(hyps, model, train_dataset, valid_dataset, 'clsf')

    wandb.init(project=f"hw4", name="downstream")

    train_losses_clsf, valid_losses_clsf = trainer_clsf.run()

    np.savetxt("train_losses_clsf.csv", np.array(train_losses_clsf), delimiter=",")
    np.savetxt("valid_losses_clsf.csv", np.array(valid_losses_clsf), delimiter=",")

    premodel = PreModel()
    with open('hyp_params_clsf.yaml', 'r') as f:
        hyps = yaml.load(f, Loader=yaml.SafeLoader)
    set_seed(hyps['seed'])

    pre_trainer = ClsfTrainProcess(hyps, premodel, train_dataset, valid_dataset, 'model')
    pre_train_losses, pre_valid_losses = pre_trainer.run()

    np.savetxt("pre_valid_losses.csv", np.array(pre_valid_losses), delimiter=",")
    np.savetxt("pre_train_losses.csv", np.array(pre_train_losses), delimiter=",")
