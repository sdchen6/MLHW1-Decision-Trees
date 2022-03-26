import numpy as np

def confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the confusion matrix. The confusion 
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    YOU DO NOT NEED TO IMPLEMENT CONFUSION MATRICES THAT ARE FOR MORE THAN TWO 
    CLASSES (binary).
    
    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    true_neg = 0
    false_pos = 0
    false_neg = 0
    true_pos = 0
    N = actual.shape[0]

    for i in range(0, N):
        if (not actual[i]) and (not predictions[i]):
            true_neg += 1
        elif (not actual[i]) and (predictions[i]):
            false_pos += 1
        elif (actual[i]) and (not predictions[i]):
            false_neg += 1
        else:
            true_pos += 1

    matrix = np.array([[true_neg, false_pos], [false_neg, true_pos]])
    return matrix
    
def accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the accuracy:

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    matrix = confusion_matrix(actual, predictions)
    true_neg = matrix[0,0]
    false_pos = matrix[0,1]
    false_neg = matrix[1,0]
    true_pos = matrix[1,1]

    correct = true_neg+true_pos
    wrong = false_pos+false_neg
    headshots = correct/(correct+wrong)
    return headshots



def precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    matrix = confusion_matrix(actual, predictions)
    true_neg = matrix[0,0]
    false_pos = matrix[0,1]
    false_neg = matrix[1,0]
    true_pos = matrix[1,1]

    selected = true_pos + false_pos
    relevant = true_pos + false_neg

    precision = true_pos/selected
    recall = true_pos/relevant
    
    return precision, recall

def f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the F1-measure:

   https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Hint: implement and use the precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and 
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    precision, recall = precision_and_recall(actual, predictions)
    f1 = 2* (precision*recall)/(precision+recall)
    return f1
