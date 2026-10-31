"""Regression metrics and plots corresponding to RegressionPlt.m."""
from pathlib import Path
import numpy as np
from scipy.stats import t


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if actual.size != predicted.size or actual.size == 0:
        raise ValueError('Actual and predicted values must have equal, nonzero lengths.')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Regression values must be finite.')
    residual = predicted - actual
    total = np.sum((actual - actual.mean()) ** 2)
    rho = float(np.corrcoef(actual, predicted)[0, 1]) if actual.std() > 0 and predicted.std() > 0 else float('nan')
    blanks = predicted[actual == 0]
    lod = 3 * float(np.std(blanks, ddof=1)) if blanks.size > 1 else (0.0 if blanks.size == 1 else float('nan'))
    return dict(rho=rho, R2=float(1 - np.sum(residual ** 2) / total) if total > 0 else float('nan'),
                BIAS=float(residual.mean()), RMSEP=float(np.sqrt(np.mean(residual ** 2))), LOD=lod)


def regression_intervals(actual, predicted, line):
    """OLS line with pointwise 95% mean-confidence and observation intervals."""
    x, y, line = [np.asarray(a, dtype=float).reshape(-1) for a in (actual, predicted, line)]
    design = np.column_stack((np.ones(x.size), x))
    if x.size <= 2 or np.linalg.matrix_rank(design) < 2:
        return None
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    mse = np.sum((y - design @ beta) ** 2) / (x.size - 2)
    query = np.column_stack((np.ones(line.size), line))
    fit = query @ beta
    variance = mse * np.einsum('ij,jk,ik->i', query, np.linalg.inv(design.T @ design), query)
    q = t.ppf(0.975, x.size - 2)
    ci, pi = q * np.sqrt(np.maximum(variance, 0)), q * np.sqrt(np.maximum(variance + mse, 0))
    return fit, np.column_stack((fit-ci, fit+ci)), np.column_stack((fit-pi, fit+pi))


def RegressionPlt(Y_tra, Y_tra_pre, Y_tst, Y_tst_pre, *, plot=True, output_dir=None):
    """Return rho, R2, BIAS, RMSEP, LOD arrays: calibration then test."""
    pairs = [(np.asarray(a).reshape(-1), np.asarray(p).reshape(-1)) for a, p in
             ((Y_tra, Y_tra_pre), (Y_tst, Y_tst_pre))]
    metrics = [regression_metrics(a, p) for a, p in pairs]
    if plot:
        import matplotlib.pyplot as plt
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        for i, ((a, p), metric) in enumerate(zip(pairs, metrics)):
            fig, ax = plt.subplots(figsize=(7, 5))
            if i == 1:
                line = np.linspace(a.min(), a.max(), 200)
                intervals = regression_intervals(a, p, line)
                if intervals is not None:
                    fit, ci, pi = intervals
                    ax.fill_between(line, pi[:, 0], pi[:, 1], color='lightcoral', alpha=.3, label='95% PI')
                    ax.fill_between(line, ci[:, 0], ci[:, 1], color='red', alpha=.3, label='95% CI')
                    ax.plot(line, fit, 'r-', label='Fit Line')
            ax.scatter(a, p, s=30, color='blue' if i == 0 else 'black', label='Calibration Points' if i == 0 else 'Test Points')
            limits = pairs[1][0] if i == 0 else a
            ax.plot([limits.min(), limits.max()], [limits.min(), limits.max()], 'k--', label='Ideal Line')
            ax.text(.05, .95, f"Corr. Coeff.: {metric['rho']:.3f}\nR²: {metric['R2']:.3f}", transform=ax.transAxes, va='top')
            ax.set(xlabel='Actual Concentration (µg/mL)', ylabel='Predicted Concentration (µg/mL)', title='Calibration set' if i == 0 else 'Test set')
            ax.legend(loc='lower right', frameon=False)
            fig.tight_layout()
            if output_dir is not None:
                fig.savefig(Path(output_dir) / ('calibration.png' if i == 0 else 'test.png'), dpi=150)
                plt.close(fig)
        if output_dir is None:
            plt.show()
    return tuple(np.array([m[key] for m in metrics]) for key in ('rho', 'R2', 'BIAS', 'RMSEP', 'LOD'))
