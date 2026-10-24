"""Isolated evaluator: deterministic Sim(3), fixed GT domain, missing=error.

This module must never be imported by a reconstruction tool or controller.
Inputs must already share the verified public pixel canvas. It does no resizing,
ROI selection, confidence filtering, robust fitting, or fallback alignment.
"""
import numpy as np


class AlignmentFailure(ValueError):
    pass


def fit_similarity(source, target):
    """Least-squares proper similarity, target = scale * source @ R.T + t."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError('Correspondences must have matching N x 3 shapes')
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise AlignmentFailure('At least three finite correspondences required')
    source_mean, target_mean = source.mean(0), target.mean(0)
    source_centered, target_centered = source - source_mean, target - target_mean
    if np.linalg.matrix_rank(source_centered) < 2 or np.linalg.matrix_rank(target_centered) < 2:
        raise AlignmentFailure('Correspondences must be non-collinear')
    covariance = target_centered.T @ source_centered / len(source)
    try:
        u, singular, vh = np.linalg.svd(covariance)
    except np.linalg.LinAlgError as error:
        raise AlignmentFailure('Similarity decomposition failed') from error
    parity = np.ones(3)
    parity[-1] = np.linalg.det(u @ vh)
    rotation = (u * parity) @ vh
    variance = np.square(source_centered).sum() / len(source)
    scale = float(singular @ parity / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    if scale <= 0 or not np.isfinite(scale) or not np.isfinite(translation).all():
        raise AlignmentFailure('Invalid fitted similarity')
    return scale, rotation, translation


def score_pointmaps(prediction, prediction_valid, ground_truth, ground_truth_valid):
    prediction = np.asarray(prediction, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    prediction_valid, ground_truth_valid = np.asarray(prediction_valid), np.asarray(ground_truth_valid)
    if prediction.shape != ground_truth.shape or prediction.shape[-1:] != (3,):
        raise ValueError('Prediction and GT must share the exact point-map canvas')
    if prediction_valid.shape != prediction.shape[:-1] or ground_truth_valid.shape != prediction.shape[:-1]:
        raise ValueError('Validity masks must match the canvas')
    if prediction_valid.dtype != bool or ground_truth_valid.dtype != bool:
        raise ValueError('Validity masks must be boolean')
    if not ground_truth_valid.any() or not np.isfinite(ground_truth[ground_truth_valid]).all():
        raise ValueError('Evaluator GT valid domain must be nonempty and finite')
    usable = prediction_valid & np.isfinite(prediction).all(-1)
    fitting = ground_truth_valid & usable
    domain_size = int(ground_truth_valid.sum())
    errors = np.full(prediction_valid.shape, np.inf, dtype=np.float64)
    report = {'gt_valid_pixels': domain_size, 'fit_pixels': int(fitting.sum()),
              'missing_prediction_pixels': int((ground_truth_valid & ~usable).sum()),
              'alignment': 'least_squares_Sim3_Umeyama', 'alignment_failure': None}
    transform = None
    try:
        scale, rotation, translation = fit_similarity(prediction[fitting], ground_truth[fitting])
        aligned = scale * (prediction[fitting] @ rotation.T) + translation
        errors[fitting] = np.linalg.norm(aligned - ground_truth[fitting], axis=-1)
        transform = {'scale': scale, 'rotation': rotation, 'translation': translation}
    except AlignmentFailure as error:
        report['alignment_failure'] = str(error)
    thresholds = np.arange(1, 11, dtype=np.float64) / 100
    curve = np.array([(errors[ground_truth_valid] < threshold).mean() for threshold in thresholds])
    report.update(auc_5cm=float(curve[:5].mean()), auc_10cm=float(curve.mean()),
                  thresholds_m=thresholds.tolist(), fractions_below_threshold=curve.tolist())
    return report, errors, transform
