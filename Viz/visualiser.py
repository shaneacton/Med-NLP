import os
from os.path import join
from typing import List
import matplotlib.pyplot as plt

import numpy

from Artifacts.artifact_manager import get_folder_path
from Datasets.data_processor import train_label_distribution, test_label_distribution
from Datasets.dataloader import col_headings
from Eval.config import CLASSIFIER_EPOCHS, STAGE_EPOCHS, NUM_EPOCHS, STAGED_LAYERS

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
    if len(STAGED_LAYERS) == 0:
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

    run_name = run_string.replace(" ", "_")
    artifacts_path = get_folder_path(run_name)

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
            name = met + ".png"
            path = join(artifacts_path, name)
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
    run_name = run_string.replace(" ", "_")

    if train:
        split_string = "train_"
        label_dist = train_label_distribution
    else:
        split_string = "test_"
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
    plt.title(split_string + run_string)
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

    artifacts_path = get_folder_path(run_name)
    if display:
        plt.show()
    else:
        name = split_string + "cls_prec.png"
        path = join(artifacts_path, name)
        plt.savefig(path)
        plt.clf()


def coplot_class_stats(train_stats: List, test_stats: List, run_string="", display=True, alpha=0.9):
    groups = group_classes_by_precision(train_stats[-1], test_stats[-1])
    for i, group in enumerate(groups):
        # print("plotting for group:", group)
        coplot_class_stats_group(train_stats, test_stats, group, i, run_string, display, alpha)


def group_classes_by_precision(last_train_stats, last_test_stats, max_group_size=4):
    """
        ~ (CLASSES, 2) where the 2 is precision and recall
    """

    assert last_test_stats.shape[0] == last_train_stats.shape[0]
    class_ids = list(range(last_test_stats.shape[0]))
    class_ids = [id for id in class_ids if train_label_distribution[id] > 0 and test_label_distribution[id] > 0]
    precisions = list(last_test_stats[:, 0])
    precisions = [p + last_train_stats[i: 0] for i, p in enumerate(precisions) if i in class_ids]
    sorted_class_ids = [cls_id for _, cls_id in sorted(zip(precisions, class_ids), reverse=True)]
    groups = []
    i = 0
    while True:
        if i >= len(sorted_class_ids):
            break

        group = []
        for _ in range(max_group_size):
            if i < len(sorted_class_ids):
                group.append(sorted_class_ids[i])
                i += 1
        groups.append(group)
    return groups


def coplot_class_stats_group(train_stats: List, test_stats: List, class_ids: List, group_num, run_string="", display=True, alpha=0.9):
    run_name = run_string.replace(" ", "_")

    epochs = [i for i in range(len(train_stats))]
    train_stats = numpy.stack(train_stats, axis=0)  # ~ (E, CLASSES, 2)
    test_stats = numpy.stack(test_stats, axis=0)  # ~ (E, CLASSES, 2)
    train_stats = numpy.stack(train_stats, axis=0)  # ~ (E, CLASSES, 2)
    test_stats = test_stats.swapaxes(0, 1)  # ~ (C, E, 2)
    train_stats = train_stats.swapaxes(0, 1)  # ~ (C, E, 2)

    plt.figure(figsize=(23, 12))
    #print("plotting groups", class_ids)
    for i, cls in enumerate(class_ids):
        ytrain = rolling_average(train_stats[cls, :, 0], alpha)
        ytest = rolling_average(test_stats[cls, :, 0], alpha)

        train_distribution = train_label_distribution[cls]
        test_distribution = test_label_distribution[cls]

        if train_distribution == 0 or test_distribution == 0:  # don't plot classes with no representation
            continue

        train_label = col_headings[cls] + " (" + "{:.1f}".format(ytrain[-1]) + ") D:" + repr(train_distribution) + "," + repr(test_distribution)
        plt.plot(epochs, ytrain, label=train_label)

        if i == 0:
            plt.plot(epochs, ytest, dashes=[30, 5, 10, 5], label="Test Precision", color=plt.gca().lines[-1].get_color())
        else:
            plt.plot(epochs, ytest, dashes=[30, 5, 10, 5], color=plt.gca().lines[-1].get_color())

    plt.xlabel('Epoch')
    plt.ylabel("precision")
    group_string = "top_"+ repr(group_num) + "_classes_"
    plt.title(group_string + run_string)
    add_layer_lines(epochs)

    handles, labels = plt.gca().get_legend_handles_labels()
    # sort both labels and handles by labels
    def extract_metric(label):
        if "Test Precision" in label:
            return -1
        val = float(label.split("(")[1].split(")")[0])
        if val == 0:
            distro = float(label.split(") D:")[1].split(",")[0]) / 10000
            return distro
        return val

    # sort labels by metric value
    labhandles = sorted(zip(labels, handles), key=lambda t: extract_metric(t[0]), reverse=True)
    labels, handles = zip(*labhandles)
    plt.legend(labels=labels, handles=handles, fontsize='small', bbox_to_anchor=(1, 1), loc='upper left')

    artifacts_path = get_folder_path(run_name)
    if display:
        plt.show()
    else:
        name = group_string + "cls_prec.png"
        path = join(artifacts_path, name)
        plt.savefig(path)
        plt.clf()






