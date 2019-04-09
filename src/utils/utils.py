import numpy as np
from scipy.stats import mannwhitneyu


def detection_score(y_pred, y_true, alternative='greater'):
    """
    Computes the AUC of a change detection prediction using the Mann-Whitney U
    test.
    Given the U statistic of the Mann-Whitney test, AUC is computed as
    U / (n1 * n2) where n1 and n2 are the sample sizes of the run lenght
    vectors, computed as the times between alarms in the nominal and non-nominal
    regimes.
    :param y_pred: np.ndarray of 0s and 1s, of shape (n_samples,) or
    (n_samples, 1). The predictions of a change detection test;
    :param y_true: np.ndarray of 0s and 1s, of shape (n_samples,) or
    (n_samples, 1). All 0s must be in the first half of the array, all 1s in the
    second half (i.e. the metric only works for a setting with a single change);
    :param alternative: type of Mann-Whithney U-test to run (see Scipy docs).
    :return: a tuple (AUC score, p-value).
    """
    if y_pred.ndim > 2:
        raise ValueError('Expected y_pred of shape (n_samples,) or '
                         '(n_samples, 1), got {}.'.format(y_pred.shape))
    if y_true.ndim > 2:
        raise ValueError('Expected y_true of shape (n_samples,) or '
                         '(n_samples, 1), got {}.'.format(y_true.shape))

    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    # Check if all equal
    if y_pred.sum() == y_pred.shape[0]:
        # All 1
        return 0.5, 0.
    elif y_pred.sum() == 0.:
        # All 0
        return 0.5, 0.
    else:
        regime_changes = np.diff(y_true)
        dummy_alarms_0 = np.hstack(([True], regime_changes == -1))
        alarms_0 = np.logical_and(y_pred == 1, y_true == 0)
        alarms_0 += dummy_alarms_0

        dummy_alarms_1 = np.hstack(([False], regime_changes == 1))
        alarms_1 = np.logical_and(y_pred == 1, y_true == 1)
        alarms_1 += dummy_alarms_1

        if alarms_0.sum() == dummy_alarms_0.sum():
            # No false positive in nominal regime
            if alarms_1.sum() >= dummy_alarms_1.sum():
                # A non-dummy alarm was given, change is surely detected
                return 1., 0.
            else:
                # All 0 (shouldn't get here)
                return 0.5, 0.

        if alarms_1.sum() == dummy_alarms_1.sum():
            # No true positive in non-nominal regime
            if alarms_0.sum() >= dummy_alarms_0.sum():
                # A non-dummy FP alarm was given, change is never detected
                return 0., 0.
            else:
                # All 0 (shouldn't get here)
                return 0.5, 0.

        times_0 = np.argwhere(alarms_0).reshape(-1)
        times_1 = np.argwhere(alarms_1).reshape(-1)

        rl_0 = np.diff(times_0)
        rl_1 = np.diff(times_1)
        if rl_0.shape[0] <= 20:
            rl_0 = np.random.choice(rl_0, 21, replace=True)
        if rl_1.shape[0] <= 20:
            rl_1 = np.random.choice(rl_1, 21, replace=True)

        try:
            mwu_stat, mwu_pval = mannwhitneyu(rl_0, rl_1, alternative=alternative)
        except ValueError:
            # TODO edge case
            return 0.5, 0.

        return mwu_stat / (rl_0.shape[0] * rl_1.shape[0]), mwu_pval


def dataset_load(data):
    # Load data from file
    if isinstance(data, str):
        if data.endswith('.npy') or data.endswith('.npz'):
            # Load from .npy/.npz format
            data = np.load(data)
        elif data.endswith('.txt'):
            # Load as text data
            data = np.loadtxt(data)
        else:
            import joblib
            data = joblib.load(data)

    # Split data into embeddings and labels
    if isinstance(data, list):
        if len(data) == 3:
            nominal, live, labels = data[0], data[1], data[2]
            labels = labels.reshape(-1)
            return nominal, live, labels
        else:
            live, labels = data[0], data[1]
            labels = labels.reshape(-1)
            return live, labels
    else:
        raise TypeError('Unsupported data format: {}.'.format(type(data)))
