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


def get_inverse_label_frequencies(label_data):
    scores = np.array(get_label_frequency_scores(label_data))
    print("scores:", len(scores), scores)
    inverse = 1/scores
    inverse **= 0.25
    print("inverse:", inverse)
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
        # print("label:", label)
        # print("mask:", mask)
        # print("freqs:", class_frequencies)
        print("total freq:", total_frequency)
    sum_label_freq = sum(scores)
    for i in range(len(scores)):  # makes the total amount of weight for positive and negative examples the same
        if scores[i] == 0:
            scores[i] = sum_label_freq/num_negative_examples
    return scores


def get_normalised_class_counts(label_data ):
    labels = torch.stack(label_data, dim=0).cpu().detach().numpy()
    print("num labels:", len(label_data))
    print("labels:", labels.shape)
    print(labels)
    class_counts = np.sum(labels, axis=0)  # per class total  examples in dataset
    print("class counts:", class_counts)
    total_conditions = np.sum(class_counts)
    print("total conditions:", total_conditions)
    _normalised_class_counts = class_counts/total_conditions
    print("norm counts:", _normalised_class_counts)
    return _normalised_class_counts


def get_label_total_frequency(label):
    print(label)
    normalised_class_counts = get_normalised_class_counts()


if __name__ == "__main__":
    get_label_frequency_scores()