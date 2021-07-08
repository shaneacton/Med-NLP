import numpy as np
import torch
from torch import Tensor


def classwise_confusion_matrix(predicted: Tensor, label: Tensor):
    """
        pred, label ~ (batch, num_classes)
        :return: a (num_classes, 4) np array with confusion data for each class for the given batch
    """
    assert predicted.size() == label.size()
    num_classes = predicted.size(1)
    confusions = []
    for cls_id in range(num_classes):
        confusion = get_confusion_matrix(predicted[:, cls_id:cls_id+1], label[:, cls_id:cls_id+1], return_list=True)
        confusions.append(confusion)
    class_wise_confusion = np.array(confusions)
    # print("cwc:", class_wise_confusion.shape, class_wise_confusion)
    return class_wise_confusion


def confusion_to_precision(confusion_matrix):
    """

    :param confusion_matrix: (*, 4)
    :return: precision and recall stats ~ (*, 2)
    """

    # print("confusion matrx:", confusion_matrix.shape)
    if confusion_matrix.ndim == 1:
        confusion_matrix.reshape((1, confusion_matrix.shape[0]))

    num_classes = confusion_matrix.shape[0]
    stats = []
    for cls_id in range(num_classes):
        cfn = confusion_matrix[cls_id, :]
        # print("cfn:", cfn)
        true_positives, true_negatives, false_positives, false_negatives = cfn[0], cfn[1], cfn[2], cfn[3]
        if true_positives == 0:
            precision = 0
            recall = 0
        else:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
        stats.append([precision, recall])
    stats = np.array(stats)
    return stats


def get_confusion_matrix(predicted: Tensor, label: Tensor, return_list=False):
    """
        pred, label ~ (batch, num_classes). Here num classes can be 1, making the analysis classwise
        :return: a 4-tuple with (tp, tn, fp, fn)
    """
    matches = predicted == label
    mistakes = matches == False
    label_zeros = label <= 0
    label_ones = label >= 0.999
    # logical and gives us the elements which are for eg, both true and predicted true
    # sum tells us how many such elements
    true_positives = torch.sum(torch.logical_and(label_ones, matches)).item()
    true_negatives = torch.sum(torch.logical_and(label_zeros, matches)).item()

    false_positives = torch.sum(torch.logical_and(label_zeros, mistakes)).item()  # we were wrong, it was neg
    false_negatives = torch.sum(torch.logical_and(label_ones, mistakes)).item()  # we were wrong, it was pos

    if return_list:
        return [true_positives, true_negatives, false_positives, false_negatives]
    return true_positives, true_negatives, false_positives, false_negatives


def analyse_performance(predicted: Tensor, label: Tensor, loss:float=None):
    """
    :param class_id: if given, the analysis will only be performed for this conditions (predicted, label) pairs\
    pred, label ~ (batch, num_classes)
    """
    assert predicted.size() == label.size()
    true_positives, true_negatives, false_positives, false_negatives = get_confusion_matrix(predicted, label)

    total = label.size(0) * label.size(1)
    accuracy = (true_positives + true_negatives)/ total
    if true_positives == 0:
        precision = 0
        recall = 0
    else:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

    metrics = [precision, recall, accuracy, true_positives, true_negatives, false_positives, false_negatives]
    if loss is not None:
        metrics.append(loss)
    performance = np.array(metrics)
    return performance


def get_inverse_label_frequencies(label_data):
    scores = np.array(get_label_frequency_scores(label_data))
    inverse = 1/scores
    inverse **= 0.25
    return inverse


def get_label_frequency_scores(label_data):
    normalised_class_counts = get_normalised_class_counts(label_data)
    scores = []
    num_negative_examples = 0
    for label in label_data:
        label = label.cpu().detach().numpy()
        mask = label == 1
        class_frequencies = normalised_class_counts[mask]
        total_frequency = np.sum(class_frequencies)
        scores.append(total_frequency)
        if total_frequency == 0:
            num_negative_examples += 1

    sum_label_freq = sum(scores)
    for i in range(len(scores)):  # makes the total amount of weight for positive and negative examples the same
        if scores[i] == 0:
            scores[i] = sum_label_freq/num_negative_examples
    return scores


def get_normalised_class_counts(label_data ):
    labels = torch.stack(label_data, dim=0).cpu().detach().numpy()
    print(labels)
    class_counts = np.sum(labels, axis=0)  # per class total  examples in dataset
    total_conditions = np.sum(class_counts)
    _normalised_class_counts = class_counts/total_conditions
    return _normalised_class_counts
