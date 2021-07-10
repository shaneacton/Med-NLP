import os
from os.path import join
from typing import List
import matplotlib.pyplot as plt

import numpy

from Datasets.data_processor import train_label_distribution, test_label_distribution
from Datasets.dataloader import col_headings
from Eval.config import CLASSIFIER_EPOCHS, STAGE_EPOCHS, NUM_EPOCHS

id_to_str = {0: "precision", 1: "recall", 2: "accuracy", 3: "tru pos", 4: "tru neg", 5: "fls pos", 6: "fls neg", 7: "loss"}
str_to_id = {v: k for k, v in id_to_str.items()}
DIR, _ = os.path.split(os.path.abspath(__file__))


def plot_stats(performances: List[numpy.array], metrics=["precision", "loss"], train=True, run_string=""):
    perf = numpy.stack(performances, axis=0)  # ~ (E, Metrics)
    print("perf:", perf.shape)
    epochs = [i for i in range(len(performances))]
    for met in metrics:
        plt.plot(epochs, perf[:, str_to_id[met]], label=met)
    plt.xlabel('Epoch')
    plt.legend()
    train_str = "train" if train else "test"
    plt.title(train_str + ": " + run_string)
    plt.show()


def add_layer_lines(epochs):
    if len(epochs) < CLASSIFIER_EPOCHS:
        return
    plt.axvline(CLASSIFIER_EPOCHS, linewidth=2, color='r')
    next_stage_line = CLASSIFIER_EPOCHS + STAGE_EPOCHS
    while next_stage_line < len(epochs) and next_stage_line < NUM_EPOCHS:
        plt.axvline(next_stage_line, linewidth=2, color='r')
        next_stage_line += STAGE_EPOCHS


def coplot(train_performances, test_performances, metrics=["precision", "loss"], run_string="", display=True):
    train_perf = numpy.stack(train_performances, axis=0)  # ~ (E, Metrics)
    test_perf = numpy.stack(test_performances, axis=0)  # ~ (E, Metrics)
    epochs = [i for i in range(len(train_perf))]

    for met in metrics:
        plt.plot(epochs, train_perf[:, str_to_id[met]], label="train_"+met)
        plt.plot(epochs, test_perf[:, str_to_id[met]], label="test_"+met)
        plt.xlabel('Epoch')
        plt.ylabel(met)
        plt.legend()
        plt.title(run_string)
        add_layer_lines(epochs)
        if display:
            plt.show()
        else:
            name = met + "_" + run_string.replace(" ", "_") + ".png"
            path = join(DIR, name)
            plt.savefig(path)
            plt.clf()


def rolling_average(values, alpha):
    avs = []
    average = values[0]
    for i, val in enumerate(values):
        average = average * alpha + (1-alpha) * val
        avs.append(average)
    return avs


def plot_class_stats(stats, run_string="", display=True, alpha=0.9, train=True):
    if train:
        run_string = "train_" + run_string
        label_dist = train_label_distribution
    else:
        run_string = "test_" + run_string
        label_dist = test_label_distribution

    epochs = [i for i in range(len(stats))]
    stats = numpy.stack(stats, axis=0)  # ~ (E, CLASSES, 2)
    stats = stats.swapaxes(0, 1)  # ~ (C, E, 2)
    plt.figure(figsize=(23, 12))

    for cls in range(stats.shape[0]):
        y = stats[cls, :, 0]
        y = rolling_average(y, alpha)
        last_val = "{:.1f}".format(y[-1])
        distribution = label_dist[cls]
        if distribution == 0:  # don't plot classes with no representation
            continue
        label = col_headings[cls] + " (" + last_val + ") D:" + repr(distribution)
        plt.plot(epochs, y, label=label)

    plt.xlabel('Epoch')
    plt.ylabel("precision")
    plt.title(run_string)
    add_layer_lines(epochs)

    handles, labels = plt.gca().get_legend_handles_labels()
    # sort both labels and handles by labels
    def extract_metric(label):
        val = float(label.split("(")[1].split(")")[0])
        if val == 0:
            distro = float(label.split(") D:")[1]) / 10000
            return distro
        return val

    # sort labels by metric value
    labhandles = sorted(zip(labels, handles), key=lambda t: extract_metric(t[0]), reverse=True)
    labels, handles = zip(*labhandles)
    plt.legend(labels=labels, handles=handles, fontsize='small', bbox_to_anchor=(1, 1), loc='upper left')

    if display:
        plt.show()
    else:
        name = "cls_prec" + "_" + run_string.replace(" ", "_") + ".png"
        path = join(DIR, name)
        plt.savefig(path)
        plt.clf()






