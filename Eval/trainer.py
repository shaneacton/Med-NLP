import math
import time

import torch
from tqdm import tqdm

from Datasets.data_processor import load_splits
from Models.bert_classifier import BertBinaryClassifier
from Models.medbert import tokeniser, model
from Viz.visualiser import coplot, plot_class_stats, coplot_class_stats
from analysis import analyse_performance, classwise_confusion_matrix, confusion_to_precision
from config import LR, DROPOUT, CLASSIFIER_EPOCHS, STAGED_LAYERS, STAGE_EPOCHS, MAX_BATCHES, NUM_EPOCHS, WEIGHT_DECAY, \
    RESAMPLE_TRAIN_DATA, RESAMPLE_METHOD, INITIAL_LAYERS
from device_settings import device

train, test = load_splits()

cls = BertBinaryClassifier(model, tokeniser, 768, 30, dropout=DROPOUT).to(device)
optim = torch.optim.Adam([c for c in cls.parameters() if c.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)


last_activated_stage = -1
train_performances = []
train_class_stats = []
test_performances = []
test_class_stats = []

run_string = ""
if RESAMPLE_TRAIN_DATA:
    run_string = RESAMPLE_METHOD + "_sampler_"
run_string += "lr: " + repr(LR) + " epochs: " + repr(NUM_EPOCHS) + " drop: " + repr(DROPOUT) + " decay: " + repr(WEIGHT_DECAY)
run_string += "\ninit unfrozen: " + repr(INITIAL_LAYERS) + \
              "\nstaged unfrozen: " + repr(STAGED_LAYERS)


def train_batches(data_split, train=True):
    performance = None
    classwise_confusion = None
    global cls
    global optim
    cls.train(mode=train)
    # label_sum = None

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
        cwc = classwise_confusion_matrix(predicted, label)

        if performance is None:
            performance = perf
            classwise_confusion = cwc
            # label_sum = label
        else:
            performance += perf
            classwise_confusion += cwc
            # if label_sum.size(0) == label.size(0):
            #     label_sum += label
    performance /= b
    # label_sum = torch.sum(label_sum, dim=0)
    # print("label sum:", label_sum.size(), label_sum)

    return performance, classwise_confusion


for e in range(NUM_EPOCHS):
    e_start_time = time.time()
    next_stage = math.floor((e - CLASSIFIER_EPOCHS) / STAGE_EPOCHS)
    if 0 <= next_stage < len(STAGED_LAYERS) and next_stage != last_activated_stage:  # activating a new stage
        last_activated_stage = next_stage
        print("activating", STAGED_LAYERS[next_stage])
        cls.activate_layers(STAGED_LAYERS[next_stage])
        optim = torch.optim.Adam([c for c in cls.parameters() if c.requires_grad], lr=LR*0.5)

    train_performance, train_confusion = train_batches(train)
    test_performance, test_confusion = train_batches(test, False)
    train_class_stats.append(confusion_to_precision(train_confusion))
    test_class_stats.append(confusion_to_precision(test_confusion))

    print("train - e:", e, "precision:", train_performance[0], "recall:", train_performance[2], "loss:", train_performance[-1])
    print("test - precision:", test_performance[0], "recall:", test_performance[2], "loss:", test_performance[-1])

    train_performances.append(train_performance)
    test_performances.append(test_performance)

    coplot(train_performances, test_performances, run_string=run_string, display=False)
    plot_class_stats(test_class_stats, run_string=run_string, display=False, train=False)
    plot_class_stats(train_class_stats, run_string=run_string, display=False, train=True)
    coplot_class_stats(train_class_stats, test_class_stats, run_string=run_string, display=False)


# graph performances over epochs
# plot_stats(train_performances, train=True, run_string=run_string)
# plot_stats(test_performances, train=False, run_string=run_string)
coplot(train_performances, test_performances, run_string=run_string, display=True)

