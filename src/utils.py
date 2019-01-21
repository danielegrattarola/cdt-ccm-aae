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


def dataset_bootstrap(data, classes=None, n_train_samples=1000,
                      n_test_samples=1000):
    """
    Randomly bootstraps a dataset of embeddings into a train and a test set.
    Integer class labels must be provided as part of the data. The method
    expects to find at least 1 embedding of class 0, and at least 1 embedding
    for any other class in `classes` for test data.
    :param data: data from which to sample the training and test sets.
        Can be:
        - list of np.ndarrays [embeddings, labels], where embeddings.shape ==
        (n_samples, latent_space) and labels.shape == (n_samples, 1) or
        (n_samples, );
        - np.ndarray of shape (n_samples, latent_space + 1) in which the last
          column represents the labels;
        - path to numpy serialized file (e.g. .npy, .npz, or .txt files) storing
          data in the np.ndarray format described above;
        - path to pickled file storing data in any of the first two formats
          described above.
    :param classes: classes to use for testing. The values must match the labels
        column of `data`.
        Possible values:
        - list of class indices (e.g. [0, 1, 2, 3]);
        - int, converted to range(classes);
        - None, use all labels in the dataset.
    :param n_train_samples: number of embeddings of class 0 to samples as train
        data.
    :param n_test_samples: number of embeddings per class in `classes` to sample
        as test data.
    :return: a tuple containing:
        - train data, np.ndarray of shape (n_train_samples, latent space);
        - test data, np.ndarray of shape (n_test_samples * n_classes,
          latent_space);
    """
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
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError('data: too many dimensions (expected ndim == 2).')
        embeddings, labels = data[..., :-1], data[..., -1:]
    elif isinstance(data, list):
        if len(data) != 2:
            raise ValueError('Expected a list [embeddings, labels], but got '
                             'a list of length {}.'.format(len(data)))
        for i in range(len(data)):
            if not isinstance(data[i], np.ndarray):
                raise TypeError('data[{}]: expected a numpy array.'.format(i))

        embeddings, labels = data[0], data[1]
    else:
        raise TypeError('Unsupported data format: {}.'.format(type(data)))
    labels = labels.reshape(-1)

    if classes is None:
        classes = list(set(labels))
    elif isinstance(classes, int):
        classes = range(classes)
    else:
        if not isinstance(classes, list):
            raise TypeError('classes must be list, int, or None.')

    # Keep only requested data
    mask = np.isin(labels, [0] + classes)
    embeddings = embeddings[mask]
    labels = labels[mask]

    # Collect test indices
    test_idx = np.array([])
    for c in classes:
        c_range = np.argwhere(labels == c)
        if c_range.shape[0] < 1:
            raise ValueError('Not enough samples for class {}.'.format(c))
        t_idx = np.random.choice(c_range.reshape(-1), n_test_samples)
        test_idx = np.concatenate((test_idx, t_idx))
    test_idx = test_idx.astype(int)

    # Collect train indices
    c_range = np.argwhere(labels == 0)
    if c_range.shape[0] < 1:
        raise ValueError('Not enough samples for class 0.')
    train_idx = np.random.choice(c_range.reshape(-1), n_train_samples)
    train_idx = train_idx.astype(int)

    return embeddings[train_idx], embeddings[test_idx]