import argparse
from itertools import product

import numpy as np
import pandas as pd
from cdg.changedetection.cusum import GaussianCusum, ManifoldCLTCusum, \
    BonferroniCusum
from cdg.embedding.manifold import SphericalManifold, HyperbolicManifold, \
    EuclideanManifold
from joblib import Parallel, delayed
from spektral.geometric import hyperbolic_clip
from spektral.utils import init_logging, log
from utils import detection_score, dataset_bootstrap

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, default=None, help='Path to dataset .pkl or log folder with datasets')
parser.add_argument('--output', type=str, default=None, help='Path to create output folder')
parser.add_argument('--baseline', action='store_true', help='Use dataset_baseline instead of dataset_geom')
parser.add_argument('-r', nargs='*', type=float, default=None, help='Radius, or list of radii')

parser.add_argument('--dcdt', action='store_true', help='Run distance-based CDT')
parser.add_argument('--rcdt', action='store_true', help='Run Riemannian CDT')
args = parser.parse_args()

CUSUM_WINDOW_SIZE = 20
CUSUM_ARL = 10000
CUSUM_SIM_LEN = int(1e3)
n_train_samples = 5000
n_test_samples = 10000
latent_space = 3
radius = args.r
N_RUNS = 20
classes = list(range(1, 21))

paths = []
if args.path.endswith('.pkl'):
    # Single dataset
    paths.append(args.path)
else:
    # Dataset from normal training
    if args.baseline:
        paths.append(args.path + 'dataset_baseline/dataset.pkl')
    else:
        paths.append(args.path + 'dataset_geom/dataset.pkl')
    paths.append(args.path + 'dataset_prior/dataset.pkl')


def _d_cdt(_path, _c):
    _id = _path.split('/')[-2]
    tpr_avg = []
    fpr_avg = []
    auc_avg = []
    run = 0
    skipped = 0
    crashed = False
    while run <= N_RUNS and (skipped < 100 or skipped / (run + skipped) < 0.9):
        try:
            nominal, test = dataset_bootstrap(_path,
                                              classes=[0, _c],
                                              n_train_samples=n_train_samples,
                                              n_test_samples=n_test_samples)
        except FileNotFoundError:
            crashed = True
            break

        distances_nom = []
        distances_test = []
        try:
            for i_, r_ in enumerate(radius):
                start = i_ * latent_space
                stop = start + latent_space
                if r_ > 0.:
                    # Spherical
                    s_mean = SphericalManifold.sample_mean(nominal[:, start:stop], radius=r_)
                    d_nom = SphericalManifold.distance(nominal[:, start:stop], s_mean, radius=r_)
                    d_test = SphericalManifold.distance(test[:, start:stop], s_mean, radius=r_)
                elif r_ < 0.:
                    # Hyperbolic
                    nominal[:, start:stop] = hyperbolic_clip(nominal[:, start:stop], r=-r_)
                    s_mean = HyperbolicManifold.sample_mean(nominal[:, start:stop], radius=-r_)
                    d_nom = HyperbolicManifold.distance(nominal[:, start:stop], s_mean, radius=-r_)
                    d_test = HyperbolicManifold.distance(test[:, start:stop], s_mean, radius=-r_)
                else:
                    # Euclidean
                    s_mean = np.mean(nominal[:, start:stop], 0)
                    d_nom = np.linalg.norm(nominal[:, start:stop] - s_mean, axis=-1)[..., None]
                    d_test = np.linalg.norm(test[:, start:stop] - s_mean, axis=-1)[..., None]
                distances_nom.append(d_nom)
                distances_test.append(d_test)
        except FloatingPointError:
            # Hyperbolic mean crashed
            skipped += 1
            continue

        # Combined
        distances_nom = np.concatenate(distances_nom, -1)
        distances_test = np.concatenate(distances_test, -1)

        # Change detection
        cdt = GaussianCusum(arl=CUSUM_ARL, window_size=CUSUM_WINDOW_SIZE)
        cdt.fit(distances_nom, estimate_threshold=True, len_simulation=CUSUM_SIM_LEN)

        pred, cum_sum = cdt.predict(distances_test, reset=True)
        pred = np.array(pred).astype(int)

        true_positives = pred[n_test_samples:].mean()
        false_positives = pred[:n_test_samples].mean()
        y_pred = pred.reshape(-1, CUSUM_WINDOW_SIZE)[:, 0].reshape(-1)
        y_true = np.array([0.] * n_test_samples + [1.] * n_test_samples).reshape(-1, CUSUM_WINDOW_SIZE)[:, 0].reshape(-1)
        auc, _ = detection_score(y_pred, y_true)

        if auc > 0.:
            tpr_avg.append(true_positives)
            fpr_avg.append(false_positives)
            auc_avg.append(auc)
            run += 1
        else:
            # No true positive predictions
            skipped += 1

    if len(auc_avg) == 0 or np.isnan(np.mean(auc_avg)):
        crashed = True

    result_str = 'crashed' if crashed else 'TPR: {:.5f} FPR: {:.5f} - AUC: {:.3f}'.format(np.mean(tpr_avg), np.mean(fpr_avg), np.mean(auc_avg))
    log('Done: {} {} - {}'.format(_id, _c, result_str))

    if not crashed:
        return (_id, _c,
                np.mean(tpr_avg), np.std(tpr_avg),
                np.mean(fpr_avg), np.std(fpr_avg),
                np.mean(auc_avg), np.std(auc_avg))
    else:
        return _id, _c, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def _r_cdt(_path, _c):
    _id = _path.split('/')[-2]
    tpr_avg = []
    fpr_avg = []
    auc_avg = []
    run = 0
    skipped = 0
    crashed = False
    while run <= N_RUNS and (skipped < 100 or skipped / (run + skipped) < 0.9):
        try:
            nominal, test = dataset_bootstrap(_path,
                                              classes=[0, _c],
                                              n_train_samples=n_train_samples,
                                              n_test_samples=n_test_samples)
        except FileNotFoundError:
            crashed = True
            break

        # Change detection
        cusum_list = []
        indices = []
        for i_, r_ in enumerate(radius):
            start = i_ * latent_space
            stop = start + latent_space
            indices.append((start, stop))
            if r_ < 0.:
                # Hyperbolic
                man_tmp = HyperbolicManifold()
                man_tmp.set_radius(-r_)
                cusum_list.append(ManifoldCLTCusum(arl=CUSUM_ARL, manifold=man_tmp,
                                                   window_size=CUSUM_WINDOW_SIZE))
            elif r_ > 0.:
                # Spherical
                man_tmp = SphericalManifold()
                man_tmp.set_radius(r_)
                cusum_list.append(ManifoldCLTCusum(arl=CUSUM_ARL, manifold=man_tmp,
                                                   window_size=CUSUM_WINDOW_SIZE))
            else:
                # Euclidean
                man_tmp = EuclideanManifold()
                cusum_list.append(ManifoldCLTCusum(arl=CUSUM_ARL, manifold=man_tmp,
                                                   window_size=CUSUM_WINDOW_SIZE))

        # Bonferroni on different
        cdt = BonferroniCusum(arl=CUSUM_ARL // len(radius),
                              window_size=CUSUM_WINDOW_SIZE,
                              cusum_list=cusum_list)
        try:
            cdt.fit([nominal[..., start:stop] for start, stop in indices],
                    estimate_threshold=True, len_simulation=CUSUM_SIM_LEN,
                    radia=radius)
        except FloatingPointError:
            skipped += 1
            continue

        pred, _ = cdt.predict([test[..., start:stop] for start, stop in indices],
                              reset=True)
        pred = np.array(pred).astype(int)

        true_positives = pred[n_test_samples:].mean()
        false_positives = pred[:n_test_samples].mean()
        y_pred = pred.reshape(-1, CUSUM_WINDOW_SIZE)[:, 0].reshape(-1)
        y_true = np.array([0.] * n_test_samples + [1.] * n_test_samples).reshape(-1, CUSUM_WINDOW_SIZE)[:, 0].reshape(-1)
        auc, _ = detection_score(y_pred, y_true)

        if auc > 0.:
            tpr_avg.append(true_positives)
            fpr_avg.append(false_positives)
            auc_avg.append(auc)
            run += 1
        else:
            # No true positive predictions
            skipped += 1

    if len(auc_avg) == 0 or np.isnan(np.mean(auc_avg)):
        crashed = True

    result_str = 'crashed' if crashed else 'TPR: {:.5f} FPR: {:.5f} - AUC: {:.3f}'.format(np.mean(tpr_avg), np.mean(fpr_avg), np.mean(auc_avg))
    log('Done: {} {} - {}'.format(_id, _c, result_str))

    if not crashed:
        return (_id, _c,
                np.mean(tpr_avg), np.std(tpr_avg),
                np.mean(fpr_avg), np.std(fpr_avg),
                np.mean(auc_avg), np.std(auc_avg))
    else:
        return _id, _c, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


cdts_to_run = []
suffixes = []
log_dir_name = 'cdt'
if args.dcdt:
    suffix = 'D-CDT'
    cdts_to_run.append(_d_cdt)
    log_dir_name += '_' + suffix
    suffixes.append(suffix)
if args.rcdt:
    cdts_to_run.append(_r_cdt)
    suffix = 'R-CDT'
    log_dir_name += '_' + suffix
    suffixes.append(suffix)

if args.output is not None:
    log_dir_name = args.output + log_dir_name
log_dir = init_logging(log_dir_name)

df_columns = ['id', 'c', '1_TPR', '2_TPR_std', '3_FPR', '4_FPR_std', '5_AUC', '6_AUC_std']
for sfx_, cdt_ in zip(suffixes, cdts_to_run):
    print('{}'.format(sfx_))
    output = Parallel(1)(delayed(cdt_)(path_, c_) for path_, c_ in
                          product(paths, classes))
    df = pd.DataFrame(output)
    df.columns = df_columns
    val = ['1_TPR', '2_TPR_std', '3_FPR', '4_FPR_std', '5_AUC', '6_AUC_std']
    out = df.pivot_table(values=val, index=['id'], columns='c')
    out = out.stack(level=0)
    out.to_csv(log_dir + '{}_results.csv'.format(sfx_))

    # Exec summary
    filtered_cols = ['id', 'c', '1_TPR', '5_AUC']
    val = ['1_TPR', '5_AUC']
    exec_summary = df[filtered_cols].pivot_table(values=val, index=['id'],
                                                 columns='c')
    exec_summary = exec_summary.stack(level=0)
    exec_summary.to_csv(log_dir + '{}_exec_summary.csv'.format(sfx_))

