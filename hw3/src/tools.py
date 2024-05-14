import torch
import seaborn as sns
import matplotlib.pyplot as plt

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


def subsequent_mask(size):
    mask = torch.ones(size, size, device=DEVICE).triu_()
    return mask.unsqueeze(0) == 0


def make_mask(source_inputs, target_inputs, pad_idx):
    # TODO: че с маской
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


# визуализация механизма внимания
def visualize_attention(model, word_field, elem, num):
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
