import math
import time

import torch
from tqdm import tqdm

from Models import bert_classifier
from analysis import analyse_performance
from Models.bert_classifier import BertBinaryClassifier
from config import BATCH_SIZE, LR, DROPOUT, CLASSIFIER_EPOCHS, STAGED_LAYERS, STAGE_EPOCHS, MAX_BATCHES, TRAIN_FRAC, \
    NUM_EPOCHS, WEIGHT_DECAY
from Datasets.data_processor import load_splits
from device_settings import device
from Models.medbert import tokeniser, model
from Viz.visualiser import coplot

# data = load_dataset(BATCH_SIZE)
train, test = load_splits(BATCH_SIZE, TRAIN_FRAC)

cls = BertBinaryClassifier(model, tokeniser, 768, 30, dropout=DROPOUT).to(device)
optim = torch.optim.Adam([c for c in cls.parameters() if c.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)


last_activated_stage = -1
train_performances = []
test_performances = []

run_string = "lr: " + repr(LR) + " epochs: " + repr(NUM_EPOCHS) + " drop: " + repr(DROPOUT) + " decay: " + repr(WEIGHT_DECAY)
run_string += "\ninit unfrozen: " + repr(bert_classifier.FINE_TUNE_LAYERS) + \
              "\nstaged unfrozen: " + repr(STAGED_LAYERS)


def train_batches(data_split, train=True):
    performance = None
    global cls
    global optim
    cls.train(mode=train)

    for b, batch in tqdm(enumerate(data_split)):
        if b >= MAX_BATCHES:
            break

        tokens, label = batch
        # print("tokens:", tokens.size(), "label:", label.size())
        if train:
            optim.zero_grad()
            logits, predicted, loss = cls(tokens, label)

            loss.backward()
            optim.step()

            predicted.detach()

        else:  #eval
            with torch.no_grad():
                logits, predicted, loss = cls(tokens, label)

        perf = analyse_performance(predicted, label, loss=loss.item())
        if performance is None:
            performance = perf
        else:
            performance += perf
    performance /= b
    return performance


for e in range(NUM_EPOCHS):
    e_start_time = time.time()
    next_stage = math.floor((e - CLASSIFIER_EPOCHS) / STAGE_EPOCHS)
    if 0 <= next_stage < len(STAGED_LAYERS) and next_stage != last_activated_stage:  # activating a new stage
        last_activated_stage = next_stage
        print("activating", STAGED_LAYERS[next_stage])
        cls.activate_layers(STAGED_LAYERS[next_stage])
        optim = torch.optim.Adam([c for c in cls.parameters() if c.requires_grad], lr=LR*0.5)

    # performance = train_batches(data)
    train_performance = train_batches(train)
    test_performance = train_batches(test, False)

    print("train - e:", e, "precision:", train_performance[0], "recall:", train_performance[2], "loss:", train_performance[-1])
    print("test - precision:", test_performance[0], "recall:", test_performance[2], "loss:", test_performance[-1])

    train_performances.append(train_performance)
    test_performances.append(test_performance)

    coplot(train_performances, test_performances, run_string=run_string, display=False)


# graph performances over epochs
# plot_stats(train_performances, train=True, run_string=run_string)
# plot_stats(test_performances, train=False, run_string=run_string)
coplot(train_performances, test_performances, run_string=run_string, display=True)
