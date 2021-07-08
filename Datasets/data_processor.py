import math
import os
from os.path import join, exists
from pathlib import Path
from typing import List, Tuple

import dill
import torch
from tokenizers import Tokenizer, normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler

from Eval.analysis import get_inverse_label_frequencies
from Eval.config import RESAMPLE_TRAIN_DATA, TRAIN_FRAC, BATCH_SIZE
from Eval.device_settings import device

normaliser = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

dirname, _ = os.path.split(os.path.abspath(__file__))
print("data processor dirname")
DIR = join(dirname, "Data", "Processed")
TOKENS = "tokenIDS.data"
LABELS = "labels.data"


def save_data(data, name):
    Path(DIR).mkdir(parents=True, exist_ok=True)
    print("saving tensor data for", name)
    torch.save(data, join(DIR, name), pickle_module=dill)


def get_tokens_and_labels():
    if not exists(join(DIR, LABELS)):  # not previously saved. load through dataloader, then tokenise and save
        print("processing dataset at", DIR)
        from Models.medbert import tokeniser
        from Datasets.dataloader import comment_text, label_text

        tokens, label_data = get_all_tokens_and_labels(comment_text, tokeniser, label_text)
        save_data(tokens, TOKENS)
        save_data(label_data, LABELS)
    else:  # previously saved. just load tokens
        print("loading processed datapoints from", DIR)
        tokens = torch.load(join(DIR, TOKENS), pickle_module=dill)
        label_data = torch.load(join(DIR, LABELS), pickle_module=dill)
    return tokens, label_data


def load_full_dataset():
    tokens, label_data = get_tokens_and_labels()
    combined = [(t.to(device), label_data[i].to(device)) for i, t in enumerate(tokens)]
    return combined


def load_dataset_frac(batch_size, start_frac=0, end_frac=1, dataset=None, sample=False) -> DataLoader:  # loads, pads and batches the dataset, returns a dataloader
    if dataset is None:
        dataset = load_full_dataset()

    length = len(dataset) - 1
    start_id = math.ceil(start_frac * length)
    end_id = math.ceil(end_frac * length)
    dataset = dataset[start_id: end_id]
    print("making data loader with ids [", start_id, ":", end_id, "]")

    if sample:
        _, label_data = get_tokens_and_labels()
        label_frequenies = torch.from_numpy(get_inverse_label_frequencies(label_data)[start_id: end_id])
        sampler = WeightedRandomSampler(label_frequenies, len(dataset))
        return DataLoader(dataset, batch_size=batch_size, collate_fn=PadCollate(0), sampler=sampler)
    return DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=PadCollate(0))


def load_splits():
    dataset = load_full_dataset()
    train = load_dataset_frac(BATCH_SIZE, 0, TRAIN_FRAC, dataset=dataset, sample=RESAMPLE_TRAIN_DATA)
    test = load_dataset_frac(BATCH_SIZE, TRAIN_FRAC, 1, dataset=dataset, sample=False)
    return train, test


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
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # pad according to max_len
        batch = map(lambda x: (pad_tensor(x[0], pad=max_len, dim=self.dim), x[1]), batch)
        batch = list(batch)
        # stack all
        xs = torch.stack(list(map(lambda x: x[0], batch)), dim=0).long()
        ys = torch.stack(list(map(lambda x: x[1], batch)), dim=0).float()

        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)


if __name__ == "__main__":
    from dataloader import comment_text, label_text
    from Models.medbert import tokeniser

    tokenise_all_texts(comment_text, tokeniser)
