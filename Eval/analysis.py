import numpy as np
import torch
from torch import Tensor


def classwise_analysis(predicted: Tensor, label: Tensor):
    assert predicted.size() == label.size()
    print("predicted:", predicted.size())
    num_classes = predicted.size(1)
    for cls_id in range(num_classes):
        analyse_performance(predicted, label, cls_id)


def analyse_performance(predicted: Tensor, label: Tensor, class_id:int=None, loss:float=None):
    """
    :param class_id: if given, the analysis will only be performed for this conditions (predicted, label) pairs\
    pred, label ~ (batch, num_classes)
    """
    assert predicted.size() == label.size()

    if class_id is not None:
        pass

    matches = predicted == label
    zeros = label <=0
    ones = label >= 0.999

    # logical and gives us the elements which are for eg, both true and predicted true
    # sum tells us how many such elements
    true_positives = torch.sum(torch.logical_and(ones, matches)).item()
    true_negatives = torch.sum(torch.logical_and(zeros, matches)).item()
    false_positives = torch.sum(torch.logical_and(ones, matches == False)).item()
    false_negatives = torch.sum(torch.logical_and(zeros, matches == False)).item()

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