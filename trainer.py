import math
import time

import numpy
import torch
from numpy import mean
from tqdm import tqdm

import bert_classifier
from analysis import analyse_performance
from bert_classifier import BertBinaryClassifier
from data_processor import load_dataset, load_splits
from main import device
from medbert import tokeniser, model
from visualiser import plot_stats, coplot

BATCH_SIZE = 8
LR = 0.0001
CLASSIFIER_EPOCHS = 20  # 40 time before staged unfreezing. Just the classifier and any base bert layers
STAGED_LAYERS = ["layer.11", "layer.10", "layer.0"]
STAGE_EPOCHS = 10
FINISHING_EPOCHS = 0  # time after all layers have been unfrozen
MAX_BATCHES = 999  # for debug. makes epochs run faster
TRAIN_FRAC = 0.9

NUM_EPOCHS = CLASSIFIER_EPOCHS + len(STAGED_LAYERS) * STAGE_EPOCHS + FINISHING_EPOCHS

# data = load_dataset(BATCH_SIZE)
train, test = load_splits(BATCH_SIZE, TRAIN_FRAC)

cls = BertBinaryClassifier(model, tokeniser, 768, 30).to(device)
optim = torch.optim.Adam([c for c in cls.parameters() if c.requires_grad], lr=LR, weight_decay=1e-6)


last_activated_stage = -1
train_performances = []
test_performances = []


def train_batches(data_split, train=True):
    performance = None

    for b, batch in tqdm(enumerate(data_split)):
        if b >= MAX_BATCHES:
            break

        tokens, label = batch
        # print("tokens:", tokens.size(), "label:", label.size())
        if train:
            optim.zero_grad()
            logits, predicted, loss = cls(tokens, label)
            loss *= 5  # todo remove

            loss.backward()
            optim.step()

            predicted.detach()

        else:  #eval
            with torch.no_grad():
                logits, predicted, loss = cls(tokens, label)
                loss *= 5

        perf = analyse_performance(predicted, label, loss=loss.item())
        if performance is None:
            performance = perf
        else:
            performance += perf
    performance /= b
    return performance


for e in range(NUM_EPOCHS):
    e_start_time = time.time()
    next_stage = math.floor((e-CLASSIFIER_EPOCHS)/STAGE_EPOCHS)
    if 0 <= next_stage <= len(STAGED_LAYERS) and next_stage != last_activated_stage:  # activating a new stage
        last_activated_stage = next_stage
        print("activating", STAGED_LAYERS[next_stage])
        cls.activate_layers(STAGED_LAYERS[next_stage])
        optim = torch.optim.Adam([c for c in cls.parameters() if c.requires_grad], lr=LR, weight_decay=1e-6)

    # performance = train_batches(data)
    train_performance = train_batches(train)
    test_performance = train_batches(test, False)

    print("train - e:", e, "precision:", train_performance[0], "recall:", train_performance[2], "loss:", train_performance[-1])
    print("test - precision:", test_performance[0], "recall:", test_performance[2], "loss:", test_performance[-1])

    train_performances.append(train_performance)
    test_performances.append(test_performance)

run_string = "lr: " + repr(LR) + " num epochs: " + repr(NUM_EPOCHS)
run_string += "\ninitial unfrozen: " + repr(bert_classifier.FINE_TUNE_LAYERS) + \
              "\nstaged unfrozen: " + repr(STAGED_LAYERS)

# graph performances over epochs
plot_stats(train_performances, train=True, run_string=run_string)
plot_stats(test_performances, train=False, run_string=run_string)
coplot(train_performances, test_performances, run_string=run_string)