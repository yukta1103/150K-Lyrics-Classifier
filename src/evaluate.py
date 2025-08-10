from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from scipy.stats import pearsonr, spearmanr

def regression_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    try:
        pearson_r, _ = pearsonr(y_true, y_pred)
    except Exception:
        pearson_r = float('nan')
    try:
        spearman_r, _ = spearmanr(y_true, y_pred)
    except Exception:
        spearman_r = float('nan')
    return {"mse": mse, "rmse": rmse, "mae": mae, "pearson_r": pearson_r, "spearman_r": spearman_r}

if __name__ == "__main__":
    y_true = [0.1, 0.5, 0.9]
    y_pred = [0.2, 0.45, 0.8]
    print(regression_metrics(y_true, y_pred))
