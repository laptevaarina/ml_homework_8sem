import evaluate as evaluate
import torch
import json
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from torchtext.data import BucketIterator


from preprocess import load_dataset, BOS_TOKEN, load_word_field, EOS_TOKEN
from model import EncoderDecoder
from tools import DEVICE, convert_batch, make_mask, tokens_to_words, draw, words_to_tokens


def summarization_generator(path_to_model, source, word_field, vocab_size):
    ''' Генератор суммаризации для модели (задание №1)
    :param path_to_model: путь до обученной модели в формате .pt
    :param source: torch.tensor с закодированными с помощью словаря токенами исходного текста
    :param word_field: словарь
    :param vocab_size: размер словаря '''
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
    ''' Оценка модели на тестовой выборке с помощью ROUGE метрики (задание №2)
    :param path_to_model: путь до обученной модели в формате .pt
    :param word_field: словарь
    :param vocab_size: размер словаря
    :param iter: итератор выборки, на которой будет происходить подсчёт метрики,
                в которой находятся закодированные токены '''
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


def example_summarization():
    ''' Сохранение результатов генератора суммаризации для 5 примеров из тестовой выборки
     и 5 собственных примеров из news_examples.txt (задание №1)'''

    # примеры генератора суммаризации для примеров из тестовой выборки
    data = []
    for i, elem in enumerate(test_iter):
        text = ' '.join(tokens_to_words(word_field, elem.source))
        res = summarization_generator("model_last.pt", elem.source, word_field, vocab_size)
        data.append((text, ' '.join(tokens_to_words(word_field, res))))
        if i == 4:
            break

    with open(f"examples_from_tests.txt", "w") as f:
        for example in data:
            f.writelines(f"source: {example[0]}\n")
            f.writelines(f"model output: {example[1]}\n")
            f.write("\n")

    # примеры генератора суммаризации для своих примеров
    with open("news_examples.txt", 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = []
    for i, news in enumerate(lines):
        source_news = word_field.preprocess(news)
        source_tokens = words_to_tokens(word_field, source_news).to(DEVICE)
        res = summarization_generator("model_last.pt", source_tokens, word_field, vocab_size)
        data.append((news, ' '.join(tokens_to_words(word_field, res))))
        if i == 4:
            break

    with open(f"our_examples.txt", "w") as f:
        for example in data:
            f.writelines(f"source: {example[0]}\n")
            f.writelines(f"model output: {example[1]}\n")
            f.write("\n")


def visualize_attention(model, word_field, elem, num):
    ''' Визуализаци механизма внимания (задание №3)
    :param model: обученная модель
    :param word_field: словарь
    :param elem: элемент выборки, с полями source и target
    :param num: номер примера '''
    source_input, target_input_ref, source_mask, target_mask_ref = convert_batch(elem)
    encoder_output = model.encoder(source_input, source_mask)

    words = tokens_to_words(word_field, elem.source)
    for layer in range(4):
        print("Encoder Layer", layer + 1)
        for h in range(4):
            print(model.encoder._blocks[layer]._self_attn._attn_probs)
            plt.title(f"Encoder Layer {layer + 1}, Attention Head {h + 1}")
            draw(model.encoder._blocks[layer]._self_attn._attn_probs[0, h].data.cpu(), words, words)
            plt.tick_params(labelsize=6)
            plt.savefig(f"../visualize_attention/example_{num}/encoder_layer_{layer+1}/attn_head_{h+1}.jpg")
            # plt.show()


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

    infer("model_last.pt", word_field, vocab_size, test_iter)



