import evaluate as evaluate
import torch
import json
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from torchtext.data import BucketIterator
import torch.nn.functional as F


from preprocess import data_preparing, load_dataset, BOS_TOKEN, load_word_field, EOS_TOKEN
from model import EncoderDecoder
from tools import DEVICE, convert_elem, convert_batch, make_mask, tokens_to_words, draw, visualize_attention


def summarization_generator(path_to_model, source, word_field, vocab_size):
    model = EncoderDecoder(source_vocab_size=vocab_size, target_vocab_size=vocab_size)
    model.load_state_dict(torch.load(path_to_model))
    model.to(DEVICE)

    source = source.transpose(0, 1)
    target = torch.tensor(word_field.vocab.stoi[BOS_TOKEN]).resize(1, 1).to(DEVICE)

    source_mask, target_mask = make_mask(source, target, pad_idx=1)
    source_emb, target_emb = model._emb
    encoder_output = model.encoder(source, source_mask)
    while word_field.vocab.stoi[EOS_TOKEN] not in target[0]:
        source_mask, target_mask = make_mask(source, target, pad_idx=1)
        decoder_output = model.decoder(target, encoder_output, source_mask, target_mask)
        out = decoder_output.contiguous().view(-1, decoder_output.shape[-1])
        target = torch.cat((target, torch.argmax(out, dim=1).resize(1, target.shape[-1])), dim=-1)

    return target[0]


def infer(path_to_model, word_field, vocab_size, iter):
    model = EncoderDecoder(source_vocab_size=vocab_size, target_vocab_size=vocab_size)
    model.load_state_dict(torch.load(path_to_model))
    model.to(DEVICE)

    # ROUGH метрика
    predictions = []
    references = []
    for i, elem in enumerate(iter):
        source_input, target_input_ref, source_mask, target_mask_ref = convert_batch(elem)
        references.append(' '.join(tokens_to_words(word_field, target_input_ref[0])))

        encoder_output = model.encoder(source_input, source_mask)
        target_input = torch.tensor(word_field.vocab.stoi[BOS_TOKEN]).resize(1, 1).to(DEVICE)
        count = target_input_ref.shape[1] // 2
        for j in range(count):
            source_mask, target_mask = make_mask(source_input, target_input, pad_idx=1)
            decoder_output = model.decoder(target_input, encoder_output, source_mask, target_mask)
            out = decoder_output.contiguous().view(-1, decoder_output.shape[-1])
            target_input = torch.cat((target_input, torch.argmax(out, dim=1).resize(1, target_input.shape[-1])), dim=-1)

        predictions.append(' '.join(tokens_to_words(word_field, target_input[0])))

    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, references=references)
    print(results)

    with open(f"ROUGE_metrics_emb.json", "w+") as f:
        json.dump({"rouges": results}, f)
        print("\n", file=f)


if __name__ == "__main__":
    # подгружаем сохранённый test_dataset
    test_dataset = load_dataset("../data/test")
    test_iter = BucketIterator(test_dataset, batch_size=1, device=DEVICE)

    word_field = load_word_field("../data")

    # подгружаем сохранённый размер словаря
    with open("datasets_sizes.json") as f_in:
        datasets_sizes = json.load(f_in)

    vocab_size = datasets_sizes["Vocab size "]
    print("Vocab size = ", vocab_size)

    train_dataset = load_dataset("../data/train")
    train_iter = BucketIterator(train_dataset, batch_size=1, device=DEVICE)

    model = EncoderDecoder(source_vocab_size=vocab_size, target_vocab_size=vocab_size)
    model.load_state_dict(torch.load("model_last.pt"))
    model.to(DEVICE)

    for i, elem in enumerate(test_iter):
        visualize_attention(model, word_field, elem, i+1)
        if i == 2:
            break

    # infer("model_last.pt", word_field, vocab_size, test_iter)

    #  примеры генератора суммаризации
    # data = []
    # for i, elem in enumerate(test_iter):
    #     text = ' '.join(tokens_to_words(word_field, elem.source))
    #     res = summarization_generator("model_metrics.pt", elem.source, word_field, vocab_size)
    #     data.append((text, ' '.join(tokens_to_words(word_field, res))))
    #     if i == 4:
    #         break
    #
    # with open(f"examples_from_tests.txt", "w") as f:
    #     for example in data:
    #         f.writelines(f"source: {example[0]}\n")
    #         f.writelines(f"model output: {example[1]}\n")
    #         f.write("\n")
