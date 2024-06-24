import torch
import seaborn as sns
import matplotlib.pyplot as plt

BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'


def draw(data, x, y):
    plt.gcf().set_size_inches(30, 30)
    sns.heatmap(data, xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, cbar=False)  #, ax=ax)


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
    return device


DEVICE = get_device()


def tokens_to_words(word_field, sent):
    sentence = []
    for elem in sent:
        sentence.append(word_field.vocab.itos[elem])
    return sentence


def words_to_tokens(word_field, sent):
    sentence = [word_field.vocab.stoi[BOS_TOKEN]]
    for elem in sent:
        sentence.append(word_field.vocab.stoi[elem])
    sentence.append(word_field.vocab.stoi[EOS_TOKEN])
    return torch.tensor(sentence)[:, None]


def subsequent_mask(size):
    mask = torch.ones(size, size, device=DEVICE).triu_()
    return mask.unsqueeze(0) == 0


def make_mask(source_inputs, target_inputs, pad_idx):
    source_mask = (source_inputs != pad_idx).unsqueeze(-2)
    target_mask = (target_inputs != pad_idx).unsqueeze(-2)
    target_mask = target_mask & subsequent_mask(target_inputs.size(-1)).type_as(target_mask)
    return source_mask, target_mask


def convert_elem(elem, pad_idx=1):
    source_inputs, target_inputs = elem.source.transpose(0, 1), elem.target.transpose(0, 1)
    source_mask, target_mask = make_mask(source_inputs, target_inputs, pad_idx)
    return source_inputs, target_inputs, source_mask, target_mask


def convert_batch(batch, pad_idx=1):
    source_inputs, target_inputs = batch.source.transpose(0, 1), batch.target.transpose(0, 1)
    source_mask, target_mask = make_mask(source_inputs, target_inputs, pad_idx)

    return source_inputs, target_inputs, source_mask, target_mask

