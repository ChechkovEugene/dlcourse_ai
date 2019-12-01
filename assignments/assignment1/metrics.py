import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    tp = np.sum(np.logical_and(prediction, ground_truth))
    fp = np.sum(np.greater(prediction, ground_truth))
    fn = np.sum(np.less(prediction, ground_truth))

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy = np.sum(prediction == ground_truth)/len(prediction)
    f1 = 2*(precision * recall)/(precision + recall)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    accuracy = np.sum(prediction == ground_truth) / len(prediction)
    return accuracy
