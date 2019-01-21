import argparse

import joblib
import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
from model import GAE_CCM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from spektral.datasets import delaunay
from spektral.utils import localpooling_filter
from spektral.utils.logging import log, model_to_str, init_logging, tic, toc

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=int, default=None, help='Save logs in an experiment folder')
args = parser.parse_args()

SEED = np.random.randint(1000000)
N_SAMPLES_IN_BASE = 10000
N_SAMPLES_IN_CLASS = 1000
SAVE_DATASET = True
latent_space = 3
radius = [0.]
learning_rate = 1e-3
l2_reg = 5e-4
epochs = 2000
batch_size = 64
es_patience = 20
optimizer = Adam(lr=learning_rate)
losses = ['binary_crossentropy', 'mse']
log_dir = init_logging('dataset_baseline')

print('Generating training data.')
adj, nf, y = delaunay.generate_data(return_type='numpy',
                                    classes=[0],
                                    n_samples_in_class=N_SAMPLES_IN_BASE,
                                    one_hot_labels=False,
                                    seed=SEED)
print('Generating operational data.')
live_classes = list(range(1, 21))
# It's OK to reuse seed because live samples of class 0 are never used.
# This ensures that the support is the same as the training data.
adj_live, nf_live, y_live = delaunay.generate_data(return_type='numpy',
                                                   classes=[0] + live_classes,
                                                   n_samples_in_class=N_SAMPLES_IN_CLASS,
                                                   one_hot_labels=False,
                                                   seed=SEED)

# Parameters
N = nf.shape[-2]
F = nf.shape[-1]

# Log variables
log(__file__)
vars_to_log = ['SEED', 'N_SAMPLES_IN_BASE', 'N_SAMPLES_IN_CLASS', 'N',
               'F', 'latent_space', 'radius', 'learning_rate', 'epochs',
               'batch_size', 'es_patience', 'live_classes', 'optimizer', 'losses']
log(''.join('- {}: {}\n'.format(v, str(eval(v))) for v in vars_to_log))

# Data normalization
print('Preprocessing data.')
ss = StandardScaler()
nf = ss.fit_transform(nf.reshape(-1, F)).reshape(-1, N, F)
nf_live = ss.transform(nf_live.reshape(-1, F)).reshape(-1, N, F)
fltr = localpooling_filter(adj.copy())
fltr_live = localpooling_filter(adj_live.copy())

# Train/test split
adj_train, adj_test, \
fltr_train, fltr_test, \
nf_train, nf_test = train_test_split(adj, fltr, nf, test_size=0.1)

# Train/val split
adj_train, adj_val, \
fltr_train, fltr_val, \
nf_train, nf_val = train_test_split(adj_train, fltr_train, nf_train, test_size=0.1)

# Autoencoder
model = GAE_CCM(N, F, latent_space=latent_space, radius=radius, l2_reg=l2_reg)
model.compile(optimizer=optimizer, loss=losses)

# Log models
log(model_to_str(model), print_string=False)

# Callbacks
es_callback = EarlyStopping(monitor='val_loss', patience=es_patience)
tb_callback = TensorBoard(log_dir=log_dir, batch_size=batch_size)
mc_callback = ModelCheckpoint(log_dir + 'best_model.h5', save_best_only=True, save_weights_only=True)

# Fit AE
tic('Fitting network.')
validation_data = ([adj_val, fltr_val, nf_val], [adj_val, nf_val])
ae_loss = model.fit([adj_train, fltr_train, nf_train],
                    [adj_train, nf_train],
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=validation_data,
                    callbacks=[es_callback, tb_callback, mc_callback])

# Post-training
toc()
log('Loading best weights')
model.load_weights(log_dir + 'best_model.h5')
test_results = model.evaluate([adj_test, fltr_test, nf_test],
                              [adj_test, nf_test],
                              batch_size=batch_size,
                              verbose=0)
log('Test loss: {}'.format(test_results))

# Embeddings
print('Computing operational stream.')
embeddings = model.encode([adj_live, fltr_live, nf_live])

# Save embeddings dataset
print('Saving operational stream.')
joblib.dump([embeddings, y_live], log_dir + 'dataset.pkl')
