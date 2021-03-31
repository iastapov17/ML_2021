import numpy as np


def compute_confusion_matrix(y_true, y_pred):
    classes_count = len(np.unique(y_true))
    confusion_matrix = np.zeros((classes_count, classes_count))

    for i in range(len(y_true)):
        confusion_matrix[int(y_true[i])][int(y_pred[i])] += 1
    return confusion_matrix


def get_classes(y_predict, percent):
    y_predict = y_predict[:percent, :]
    return np.array([np.argmax(y_predict[i, :]) for i in range(y_predict.shape[0])])


def accuracy_score(y_true, y_predict, percent=None):
    percent = 50 if percent is None else percent
    assert (1 <= percent <= 100) and isinstance(percent, int)
    
    percent = int((percent / 100) * y_true.shape[0])
    pred_classes = get_classes(y_predict, percent)
    count = pred_classes == y_true[:percent]
    return count.sum() / np.size(y_true)


def precision_score(y_true, y_predict, percent=None):
    percent = 50 if percent is None else percent
    assert (1 <= percent <= 100) and isinstance(percent, int)
    
    classes_count = len(np.unique(y_true))
    percent = int((percent / 100) * y_true.shape[0])
    y_true = y_true[:percent]
    predict = get_classes(y_predict, percent)
    
    confusion_matrix = compute_confusion_matrix(y_true, predict)
    return np.array([confusion_matrix[i, i] / confusion_matrix[:, i].sum() for i in range(classes_count)])


def recall_score(y_true, y_predict, percent=None):
    percent = 50 if percent is None else percent
    assert (1 <= percent <= 100) and isinstance(percent, int)
    
    classes_count = len(np.unique(y_true))
    percent = int((percent / 100) * y_true.shape[0])
    y_true = y_true[:percent]
    predict = get_classes(y_predict, percent)
    
    confusion_matrix = compute_confusion_matrix(y_true, predict)
    return np.array([confusion_matrix[i, i] / confusion_matrix[i, :].sum() for i in range(classes_count)])


def f1_score(y_true, y_predict, percent=None):
    precision = precision_score(y_true, y_predict, percent)
    recall = recall_score(y_true, y_predict, percent)
    return 2 * precision * recall / (precision + recall)


def lift_score(y_true, y_predict, percent = None):
    percent = 50 if percent is None else percent
    assert (1 <= percent <= 100) and isinstance(percent, int)
    
    unique_classes = np.unique(y_true)
    precision = precision_score(y_true, y_predict, percent)

    percent = int((percent / 100) * y_true.shape[0])
    y_true = y_true[:percent]
    
    return np.array([precision[int(u_class)] * y_true.shape[0] / (y_true == u_class).sum() for u_class in unique_classes])
