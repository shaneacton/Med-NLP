from os.path import join, exists
from pathlib import Path
from typing import List, Tuple

import dill
import torch
from tokenizers import Tokenizer, normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from torch import Tensor
from torch.utils.data import DataLoader

from main import device

normaliser = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
DIR = "Processed"
TOKENS = "tokenIDS.data"
LABELS = "labels.data"


def save_data(data, name):
    Path(DIR).mkdir(parents=True, exist_ok=True)
    print("saving tensor data for", name)
    torch.save(data, join(DIR, name), pickle_module=dill)


def load_splits(BATCH_SIZE, TRAIN_FRAC):
    data = load_dataset(BATCH_SIZE)
    length = len(data)
    train_size = int(length * TRAIN_FRAC)
    sets = torch.utils.data.random_split(data, [train_size, length - train_size])
    train, test = sets[0], sets[1]

    return train.dataset, test.dataset


def load_dataset(batch_size) -> DataLoader:  # loads, pads and batches the dataset, returns a dataloader
    if not exists(join(DIR, LABELS)):  # not previously saved. load through dataloader, then tokenise and save
        print("processing dataset")
        from dataloader import comment_text, label_text
        from medbert import tokeniser

        tokens, label_data = get_all_tokens_and_labels(comment_text, tokeniser, label_text)
        save_data(tokens, TOKENS)
        save_data(label_data, LABELS)
    else:  # previously saved. just load tokens
        print("loading processed datapoints")
        tokens = torch.load(join(DIR, TOKENS), pickle_module=dill)
        label_data = torch.load(join(DIR, LABELS), pickle_module=dill)

    combined = [(t, label_data[i]) for i, t in enumerate(tokens)]

    return DataLoader(combined, shuffle=True, batch_size=batch_size, collate_fn=PadCollate(0))  #


def pad_tensor(vec, pad, dim):  # Felix_Kreuk https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/7
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size).to(device)], dim=dim)


def get_all_tokens_and_labels(texts: List[List[str]], tokeniser: Tokenizer, labels: List[List[str]]) -> Tuple[List[Tensor], List[Tensor]]:
    label_tensors = [get_label_bools(l) for l in labels]
    return tokenise_all_texts(texts, tokeniser), label_tensors


def tokenise_all_texts(texts: List[List[str]], tokeniser: Tokenizer) -> List[Tensor]:
    tokenised_sequences = []
    for text in texts:
        # here text is a list of lines of comments for a particular image
        tokenised_sequences.append(tokenise_text(text, tokeniser))
    return tokenised_sequences


def tokenise_text(text: List[str], tokeniser: Tokenizer):
    text = [normaliser.normalize_str(t) for t in text]
    concat_text = "[SEP]".join(text)
    token_ids = tokeniser.encode(concat_text)
    print("tokenised text:", tokeniser.decode(token_ids))
    return torch.Tensor(token_ids).long().to(device)


def get_label_bools(label: List[str]):
    bools = [0 if l == "0" or l == "N" else 1 for l in label]
    return torch.Tensor(bools).long().to(device)


class PadCollate:  # Felix_Kreuk https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/7
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # print("batch:", batch)
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # pad according to max_len
        # print("max len:", max_len)
        batch = map(lambda x: (pad_tensor(x[0], pad=max_len, dim=self.dim), x[1]), batch)
        batch = list(batch)
        # stack all
        xs = torch.stack(list(map(lambda x: x[0], batch)), dim=0).long()
        ys = torch.stack(list(map(lambda x: x[1], batch)), dim=0).float()
        # print("xs:", xs.size(), "ys:", ys.size())

        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)


if __name__ == "__main__":
    from dataloader import comment_text, label_text
    from medbert import tokeniser

    tokenise_all_texts(comment_text, tokeniser)
