import argparse
from collections import OrderedDict
from itertools import product

import numpy as np
import pandas as pd
from cdg.changedetection import GaussianCusum, ManifoldCLTCusum, BonferroniCusum
from cdg.geometry import SphericalManifold, HyperbolicManifold
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix

from src.utils.utils import detection_score, dataset_load
from src.utils.logging import init_logging

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, default=None, help='Path to dataset .pkl or log folder with datasets')
parser.add_argument('--dcdt', action='store_true', help='Run distance-based CDT')
parser.add_argument('--rcdt', action='store_true', help='Run Riemannian CDT')
args = parser.parse_args()

P = OrderedDict(
    CUSUM_WINDOW_RATIO=0.001,
    CUSUM_ARL=100,              # Expected average run lenght
    CUSUM_SIM_LEN=int(1e5),     # Length of the simulations run by CUSUM to estimate the threshold
    latent_space=3,             # Dimension of each manifold
    radius=[-1., 0., 1.],       # List of radii (one for eacch manifold)
    N_RUNS=1,                   # Number of repeated runs for each CUSUM
    classes=list(range(1, 21))  # Classes to test for
)
print(P)

paths = []
if args.path.endswith('.pkl'):
    # Single dataset
    paths.append(args.path)
else:
    # Dataset from normal training
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
    while run < P['N_RUNS'] and (skipped < 100 or skipped / (run + skipped) < 0.9):
        # Read data
        data = dataset_load(_path)
        try:
            nominal, live, labels = data
        except:
            live, labels = data
            nominal = live[labels == 0].copy()
        live = live[(labels == 0) | (labels == _c)]
        labels = labels[(labels == 0) | (labels == _c)]
        labels[labels != 0] = 1
        CUSUM_WINDOW_SIZE = int(nominal.shape[0] * P['CUSUM_WINDOW_RATIO'])
        cut = CUSUM_WINDOW_SIZE * (nominal.shape[0] // CUSUM_WINDOW_SIZE)
        nominal = nominal[:cut]
        cut = CUSUM_WINDOW_SIZE * (labels.shape[0] // CUSUM_WINDOW_SIZE)
        live = live[:cut]
        labels = labels[:cut]
        live_n = live[labels == 0].copy()
        live_nn = live[labels == 1].copy()
        live = np.vstack((live_n, live_nn))

        # Compute distances
        distances_nom = []
        distances_test = []
        try:
            for i_, r_ in enumerate(P['radius']):
                start = i_ * P['latent_space']
                stop = start + P['latent_space']
                if r_ > 0.:
                    # Spherical
                    s_mean = SphericalManifold.sample_mean(nominal[:, start:stop], radius=r_)
                    d_nom = SphericalManifold.distance(nominal[:, start:stop], s_mean, radius=r_)
                    d_test = SphericalManifold.distance(live[:, start:stop], s_mean, radius=r_)
                elif r_ < 0.:
                    # Hyperbolic
                    s_mean = HyperbolicManifold.sample_mean(nominal[:, start:stop], radius=-r_)
                    d_nom = HyperbolicManifold.distance(nominal[:, start:stop], s_mean, radius=-r_)
                    d_test = HyperbolicManifold.distance(live[:, start:stop], s_mean, radius=-r_)
                else:
                    # Euclidean
                    s_mean = np.mean(nominal[:, start:stop], 0)
                    d_nom = np.linalg.norm(nominal[:, start:stop] - s_mean, axis=-1)[..., None]
                    d_test = np.linalg.norm(live[:, start:stop] - s_mean, axis=-1)[..., None]
                distances_nom.append(d_nom)
                distances_test.append(d_test)
        except FloatingPointError:
            print('D-CDT: FloatingPointError')
            skipped += 1
            continue

        # Combined
        distances_nom = np.concatenate(distances_nom, -1)
        distances_test = np.concatenate(distances_test, -1)

        # Change detection
        cdt = GaussianCusum(arl=P['CUSUM_ARL'], window_size=CUSUM_WINDOW_SIZE)
        cdt.fit(distances_nom, estimate_threshold=True, len_simulation=P['CUSUM_SIM_LEN'])

        pred, cum_sum = cdt.predict(distances_test, reset=True)
        pred = np.array(pred).astype(int)

        y_true = labels.reshape(-1, CUSUM_WINDOW_SIZE).mean(-1).round().reshape(-1)
        y_pred = pred.reshape(-1, CUSUM_WINDOW_SIZE).mean(-1).round().reshape(-1)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        auc, _ = detection_score(y_pred, y_true)

        if auc > 0.:
            tpr_avg.append(tpr)
            fpr_avg.append(fpr)
            auc_avg.append(auc)
            run += 1
        else:
            print('No true positive predictions')
            skipped += 1

    if len(auc_avg) == 0 or np.isnan(np.mean(auc_avg)):
        crashed = True

    result_str = 'crashed' if crashed else 'TPR: {:.5f} FPR: {:.5f} - AUC: {:.3f}'.format(np.mean(tpr_avg), np.mean(fpr_avg), np.mean(auc_avg))
    print('Done: {} {} - {}'.format(_id, _c, result_str))

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
    while run < P['N_RUNS'] and (skipped < 100 or skipped / (run + skipped) < 0.9):
        # Read data
        data = dataset_load(_path)
        try:
            nominal, live, labels = data
        except:
            live, labels = data
            nominal = live[labels == 0].copy()
        live = live[(labels == 0) | (labels == _c)]
        labels = labels[(labels == 0) | (labels == _c)]
        labels[labels != 0] = 1
        CUSUM_WINDOW_SIZE = int(nominal.shape[0] * P['CUSUM_WINDOW_RATIO'])
        cut = CUSUM_WINDOW_SIZE * (nominal.shape[0] // CUSUM_WINDOW_SIZE)
        nominal = nominal[:cut]
        cut = CUSUM_WINDOW_SIZE * (labels.shape[0] // CUSUM_WINDOW_SIZE)
        live = live[:cut]
        labels = labels[:cut]
        live_n = live[labels == 0].copy()
        live_nn = live[labels == 1].copy()
        live = np.vstack((live_n, live_nn))

        # Change detection
        cusum_list = []
        indices = []
        for i_, r_ in enumerate(P['radius']):
            start = i_ * P['latent_space']
            stop = start + P['latent_space']
            indices.append((start, stop))
            if r_ < 0.:
                # Hyperbolic
                man_tmp = HyperbolicManifold(radius=-r_)
                cusum_list.append(ManifoldCLTCusum(arl=P['CUSUM_ARL'], manifold=man_tmp,
                                                   window_size=CUSUM_WINDOW_SIZE))
            elif r_ > 0.:
                # Spherical
                man_tmp = SphericalManifold(radius=r_)
                cusum_list.append(ManifoldCLTCusum(arl=P['CUSUM_ARL'], manifold=man_tmp,
                                                   window_size=CUSUM_WINDOW_SIZE))
            else:
                # Euclidean
                cusum_list.append(GaussianCusum(arl=P['CUSUM_ARL'], window_size=CUSUM_WINDOW_SIZE))

        # Bonferroni on different
        cdt = BonferroniCusum(cusum_list=cusum_list, arl=P['CUSUM_ARL'] // len(P['radius']))
        try:
            cdt.fit([nominal[..., start:stop] for start, stop in indices],
                    estimate_threshold=True, len_simulation=P['CUSUM_SIM_LEN'],
                    radia=P['radius'])
        except FloatingPointError:
            print('R-CDT: FloatingPointError')
            skipped += 1
            continue

        pred, cum_sum = cdt.predict([live[..., start:stop] for start, stop in indices], reset=True)
        pred = np.array(pred).astype(int)

        y_true = labels.reshape(-1, CUSUM_WINDOW_SIZE).mean(-1).round().reshape(-1)
        y_pred = pred.reshape(-1, CUSUM_WINDOW_SIZE).mean(-1).round().reshape(-1)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        auc, _ = detection_score(y_pred, y_true)

        if auc > 0.:
            tpr_avg.append(tpr)
            fpr_avg.append(fpr)
            auc_avg.append(auc)
            run += 1
        else:
            print('No true positive predictions')
            skipped += 1

    if len(auc_avg) == 0 or np.isnan(np.mean(auc_avg)):
        crashed = True

    result_str = 'crashed' if crashed else 'TPR: {:.5f} FPR: {:.5f} - AUC: {:.3f}'.format(np.mean(tpr_avg), np.mean(fpr_avg), np.mean(auc_avg))
    print('Done: {} {} - {}'.format(_id, _c, result_str))

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
    cdts_to_run.append(_d_cdt)
    suffixes.append('D-CDT')
if args.rcdt:
    cdts_to_run.append(_r_cdt)
    suffixes.append('R-CDT')
log_dir = init_logging('cdt')

df_columns = ['id', 'c', '1_TPR', '2_TPR_std', '3_FPR', '4_FPR_std', '5_AUC', '6_AUC_std']
for sfx_, cdt_ in zip(suffixes, cdts_to_run):
    print('{}'.format(sfx_))
    output = Parallel(1)(delayed(cdt_)(path_, c_)
                          for path_, c_ in product(paths, P['classes']))
    df = pd.DataFrame(output)
    df.columns = df_columns
    val = ['1_TPR', '2_TPR_std', '3_FPR', '4_FPR_std', '5_AUC', '6_AUC_std']
    out = df.pivot_table(values=val, index=['id'], columns='c')
    out = out.stack(level=0)
    out.to_csv(log_dir + '{}_results.csv'.format(sfx_))
