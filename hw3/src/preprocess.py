import json

import dill
import torch
import wget
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torchtext.data import Field, Example, Dataset, BucketIterator

from navec import Navec

from tools import DEVICE

BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'

path = '../data/navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)


def save_dataset(dataset, path):
    with open(path+"/dataset.Field", "wb") as f:
        dill.dump(dataset.fields, f)

    with open(path+"/dataset.Example", "wb") as f:
        dill.dump(dataset.examples, f)


def load_dataset(path):
    with open(path + "/dataset.Field", "rb") as f:
        test_fields = dill.load(f)

    with open(path + "/dataset.Example", "rb") as f:
        test_examples = dill.load(f)

    return Dataset(test_examples, test_fields)


def save_word_field(field, path):
    with open(path + "/word_field.Field", "wb") as f:
        dill.dump(field, f)


def load_word_field(path):
    with open(path + "/word_field.Field", "rb") as f:
        word_field = dill.load(f)

    return word_field


def data_preparing():
    word_field = Field(tokenize='moses', init_token=BOS_TOKEN, eos_token=EOS_TOKEN, lower=True)
    fields = [('source', word_field), ('target', word_field)]

    data = pd.read_csv('../data/news.csv', delimiter=',')

    examples = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        source_text = word_field.preprocess(row.text)
        target_text = word_field.preprocess(row.title)
        examples.append(Example.fromlist([source_text, target_text], fields))

    dataset = Dataset(examples, fields)

    train_dataset, val_dataset = dataset.split(split_ratio=0.9)
    train_dataset, test_dataset = train_dataset.split(split_ratio=0.89)

    print('Train size =', len(train_dataset))
    print('Validation size =', len(val_dataset))
    print('Test size =', len(test_dataset))

    word_field.build_vocab(train_dataset, min_freq=7)
    print('Vocab size =', len(word_field.vocab))

    save_word_field(word_field, "../data")

    save_dataset(train_dataset, "../data/train")
    save_dataset(val_dataset, "../data/val")
    save_dataset(test_dataset, "../data/test")

    with open(f"datasets_sizes.json", "w+") as f:
        json.dump({"Train size ": len(train_dataset), "Validation size ": len(val_dataset),
                   "Test size ": len(test_dataset), "Vocab size ": len(word_field.vocab)}, f)
        print("\n", file=f)

    return train_dataset, val_dataset, test_dataset, word_field
