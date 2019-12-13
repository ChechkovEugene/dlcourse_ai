def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    
    tp = np.sum(np.logical_and(prediction, ground_truth))
    fp = np.sum(np.greater(prediction, ground_truth))
    fn = np.sum(np.less(prediction, ground_truth))

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    accuracy = np.sum(prediction == ground_truth)/len(prediction)
    f1 = 2*(precision * recall)/(precision + recall)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    accuracy = np.sum(prediction == ground_truth) / len(prediction)
    return accuracy
