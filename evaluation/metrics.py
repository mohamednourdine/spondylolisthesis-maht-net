def calculate_mre(predictions, targets):
    return ((predictions - targets) ** 2).mean().sqrt()

def calculate_sdr(predictions, targets, threshold):
    return ((predictions - targets).abs() <= threshold).float().mean()

def calculate_cohens_kappa(predictions, targets):
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(predictions, targets)

def calculate_accuracy(predictions, targets):
    return (predictions == targets).float().mean()