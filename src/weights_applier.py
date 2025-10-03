import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def get_class_weights(y):
    """
    Calculate class weights for handling class imbalance

    Parameters:
    - y: 1D numpy array or Pandas Series of target labels

    Returns:
    - dict: {class_label: weight}
    """
    unique_classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced',
                                   classes=unique_classes,
                                   y=y)
    return {cls: w for cls, w in zip(unique_classes, weights)}