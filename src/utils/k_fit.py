import numpy as np
from typing import Dict, Optional, Tuple
from scipy.optimize import least_squares
from sklearn.metrics import r2_score


TIME_HOURS_DEFAULT = [1, 3, 6, 10, 16, 24, 34, 48]


def _model_loss(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    a, b, k = params
    return a * np.exp(-k * t) + b


def _model_incorp(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    a, b, k = params
    return a * (1.0 - np.exp(-k * t)) + b


def _residuals(model_fn, params: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return model_fn(params, t) - y


def _compute_initial_params(y: np.ndarray) -> Tuple[float, float, float]:
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    a0 = max(y_max - y_min, 0.0)
    b0 = y_min
    k0 = float(np.log(2) / np.median([h for h in TIME_HOURS_DEFAULT if h > 0]))
    return a0, b0, k0


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return np.nan
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return np.nan


def fit_k_loss(y: np.ndarray, t: np.ndarray) -> Optional[Dict[str, float]]:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    mask = ~np.isnan(y)
    y_used = y[mask]
    t_used = t[mask]
    if y_used.size < 3:
        return None
    a0, b0, k0 = _compute_initial_params(y_used)
    x0 = np.array([a0, b0, k0], dtype=float)
    lb = np.array([0.0, -np.inf, 0.0], dtype=float)
    ub = np.array([np.inf, np.inf, np.inf], dtype=float)
    res = least_squares(
        fun=lambda p: _residuals(_model_loss, p, t_used, y_used),
        x0=x0,
        bounds=(lb, ub),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=2000,
    )
    a_hat, b_hat, k_hat = res.x
    y_pred = _model_loss(res.x, t_used)
    return {
        "A": float(a_hat),
        "B": float(b_hat),
        "K": float(max(k_hat, 0.0)),
        "success": bool(res.success),
        "y_pred": y_pred.astype(float),
        "r2": _r2(y_used, y_pred),
    }


def fit_k_incorp(y: np.ndarray, t: np.ndarray) -> Optional[Dict[str, float]]:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    mask = ~np.isnan(y)
    y_used = y[mask]
    t_used = t[mask]
    if y_used.size < 3:
        return None
    a0, b0, k0 = _compute_initial_params(y_used)
    x0 = np.array([a0, b0, k0], dtype=float)
    lb = np.array([0.0, -np.inf, 0.0], dtype=float)
    ub = np.array([np.inf, np.inf, np.inf], dtype=float)
    res = least_squares(
        fun=lambda p: _residuals(_model_incorp, p, t_used, y_used),
        x0=x0,
        bounds=(lb, ub),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=2000,
    )
    a_hat, b_hat, k_hat = res.x
    y_pred = _model_incorp(res.x, t_used)
    return {
        "A": float(a_hat),
        "B": float(b_hat),
        "K": float(max(k_hat, 0.0)),
        "success": bool(res.success),
        "y_pred": y_pred.astype(float),
        "r2": _r2(y_used, y_pred),
    }


