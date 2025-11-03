def clinical_grading(predictions, ground_truth):
    """
    Function to evaluate clinical grading based on model predictions.

    Parameters:
    predictions (list): List of predicted grades from the model.
    ground_truth (list): List of actual grades for comparison.

    Returns:
    dict: A dictionary containing evaluation metrics such as accuracy and Kappa score.
    """
    from sklearn.metrics import accuracy_score, cohen_kappa_score

    # Calculate accuracy
    accuracy = accuracy_score(ground_truth, predictions)

    # Calculate Cohen's Kappa score
    kappa = cohen_kappa_score(ground_truth, predictions)

    return {
        'accuracy': accuracy,
        'kappa': kappa
    }