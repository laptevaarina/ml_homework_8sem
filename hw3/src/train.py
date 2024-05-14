from tqdm import tqdm
import torch
import torch.nn as nn
import math
import wandb
import json
import evaluate as evaluate
from torchtext.data import BucketIterator

from model import EncoderDecoder, NoamOpt
from tools import DEVICE, convert_batch, tokens_to_words
from preprocess import data_preparing, load_dataset, load_word_field
from hparams import epochs, smoothing_coefficient

tqdm.get_lock().locks = []


def do_epoch(model, criterion, data_iter, optimizer=None, name=None):
    epoch_loss = 0

    is_train = not optimizer is None
    name = name or ''
    model.train(is_train)

    batches_count = len(data_iter)

    predictions = []
    references = []
    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batches_count) as progress_bar:
            for i, batch in enumerate(data_iter):
                source_inputs, target_inputs, source_mask, target_mask = convert_batch(batch)

                logits = model.forward(source_inputs, target_inputs[:, :-1], source_mask, target_mask[:, :-1, :-1])
                logits = logits.contiguous().view(-1, logits.shape[-1])

                for logit in logits:
                    # idx = torch.topk(logit.flatten(), target_inputs.size()[1]).indices
                    predictions.append(' '.join(word_field.vocab.itos[torch.argmax(logit)]))

                target = target_inputs[:, 1:].contiguous().view(-1)
                for tg in target:
                    references.append(' '.join(word_field.vocab.itos[tg]))

                loss = criterion(logits, target)

                epoch_loss += loss.item()

                if optimizer:
                    optimizer.optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                progress_bar.update()
                progress_bar.set_description('{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(name, loss.item(),
                                                                                         math.exp(loss.item())))

            progress_bar.set_description('{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(
                name, epoch_loss / batches_count, math.exp(epoch_loss / batches_count))
            )
            progress_bar.refresh()

    rouge = evaluate.load('rouge')
    rouge_metric = rouge.compute(predictions=predictions, references=references)

    return epoch_loss / batches_count, rouge_metric # ['rougeL']


def fit(model, criterion, optimizer, train_iter, epochs_count=1, val_iter=None):
    best_val_loss = None
    for epoch in range(epochs_count):
        name_prefix = '[{} / {}] '.format(epoch + 1, epochs_count)
        train_loss, rouge_train = do_epoch(model, criterion, train_iter, optimizer, name_prefix + 'Train:')

        val_loss = None
        if not val_iter is None:
            val_loss, rouge_val = do_epoch(model, criterion, val_iter, None, name_prefix + '  Val:')

        metrics = rouge_val
        metrics['val_loss'] = val_loss
        metrics['train_loss'] = train_loss
        wandb.log(metrics, step=epoch)

    torch.save(model.state_dict(), f"model_last.pt")

    with open("run_id.txt", "w+") as f:
        print(wandb.run.id, file=f)


if __name__ == "__main__":
    # train_dataset, val_dataset, test_dataset, word_field = data_preparing()
    train_dataset = load_dataset("../data/train")
    val_dataset = load_dataset("../data/val")
    word_field = load_word_field("../data")

    print(len(word_field.vocab))

    train_iter, val_iter = BucketIterator.splits(
        datasets=(train_dataset, val_dataset), batch_sizes=(16, 32), shuffle=True, device=DEVICE, sort=False
    )

    wandb.init(project=f"hw3", name="prototype_with_emb")

    model = EncoderDecoder(source_vocab_size=len(word_field.vocab), target_vocab_size=len(word_field.vocab)).to(DEVICE)

    pad_idx = word_field.vocab.stoi['<pad>']

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=smoothing_coefficient).to(DEVICE)

    # criterion = LabelSmoothing(len(word_field.vocab), pad_idx, 0.4).to(DEVICE)

    optimizer = NoamOpt(model)

    fit(model, criterion, optimizer, train_iter, epochs_count=epochs, val_iter=val_iter)
