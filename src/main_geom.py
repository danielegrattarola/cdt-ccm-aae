import sys
import time
import warnings

import joblib
import keras.backend as K
import numpy as np
from keras.models import Sequential, Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from spektral.layers import CCMMembership
from spektral.utils import localpooling_filter, batch_iterator

from src.utils import delaunay
from src.utils.logging import log, model_to_str, init_logging, tic, toc
from src.utils.model import GAE_CCM

# Keras 2.2.2 throws UserWarnings all over the place during training
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


SEED = np.random.randint(1000000)
N_SAMPLES_IN_BASE = 10000
N_SAMPLES_IN_CLASS = 10000
latent_space = 3
radius = [-1., 0., 1.]
sigma = 5
full_latent_space = latent_space * len(radius)
learning_rate = 1e-3
l2_reg = 5e-4
epochs = 20000
batch_size = 128
es_patience = 100
optimizer = Adam(lr=learning_rate)
losses = ['binary_crossentropy', 'mse']
log_dir = init_logging('dataset_geom')

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
               'F', 'latent_space', 'radius', 'sigma', 'learning_rate', 'epochs',
               'batch_size', 'es_patience', 'live_classes', 'optimizer', 'losses']
log(''.join('- {}: {}\n'.format(v, str(eval(v))) for v in vars_to_log))

# Data normalization
print('Preprocessing data.')
ss = StandardScaler()
nf = ss.fit_transform(nf.reshape(-1, F)).reshape(-1, N, F)
nf_live = ss.transform(nf_live.reshape(-1, F)).reshape(-1, N, F)
fltr = localpooling_filter(adj)
fltr_live = localpooling_filter(adj_live)

# Train/test split
adj_train, adj_test, \
fltr_train, fltr_test, \
nf_train, nf_test = train_test_split(adj, fltr, nf, test_size=0.1)

# Train/val split
adj_train, adj_val, \
fltr_train, fltr_val, \
nf_train, nf_val = train_test_split(adj_train, fltr_train, nf_train, test_size=0.1)

# Autoencoder
model = GAE_CCM(N, F, latent_space=latent_space, radius=radius, l2_reg=l2_reg, multi_gpu=False)
model.compile(optimizer=optimizer, loss=losses)

# Discriminator
discriminator = Sequential(name='discriminator')
discriminator.add(CCMMembership(r=radius, sigma=sigma, input_shape=(full_latent_space,),
                                name='geom_critic'))

# Frozen discriminator + Encoder
enc_discriminator = Model(inputs=model.encoder.input,
                          outputs=discriminator(model.encoder.output),
                          name='enc_discriminator')
enc_discriminator.compile('adam', 'binary_crossentropy', [mean_pred])

# Log models
log(model_to_str(model), print_string=False)
log(model_to_str(discriminator), print_string=False)

# Train model
tic('Fitting AAE')
t = time.time()
current_batch = 0
model_loss = 0  # Loss of the autoencoder
adv_fool = 0    # Mean prediction of the discriminator on positive samples
best_val_loss = np.inf
patience = es_patience
batches_in_epoch = 1 + adj_train.shape[0] // batch_size
total_batches = batches_in_epoch * epochs

for batch in batch_iterator([adj_train, fltr_train, nf_train], batch_size=batch_size, epochs=epochs):
    model_loss += model.train_on_batch(batch, [batch[0], batch[2]])[0]

    # Regularization
    true_batch_size = batch[0].shape[0]
    adv_res_fool = enc_discriminator.train_on_batch(batch, np.ones(true_batch_size))

    # Update stats
    adv_fool += adv_res_fool[1]
    current_batch += 1
    if current_batch % batches_in_epoch == 0:
        model_loss /= batches_in_epoch
        adv_fool /= batches_in_epoch
        model_val_loss = model.evaluate([adj_val, fltr_val, nf_val], [adj_val, nf_val],
                                        batch_size=batch_size, verbose=0)[0]
        log('Epoch {:3d} ({:2.2f}s) - '
            'loss {:.2f} - val_loss {:.2f} - '
            'adv_fool (should be 1.0): {:.2f}'
            ''.format(current_batch // batches_in_epoch, time.time() - t,
                      model_loss, model_val_loss,
                      adv_fool))
        if model_val_loss < best_val_loss:
            best_val_loss = model_val_loss
            patience = es_patience
            log('New best val_loss {:.3f}'.format(best_val_loss))
            model.save_weights(log_dir + 'model_best_val_weights.h5')
        else:
            patience -= 1
            if patience == 0:
                log('Early stopping (best val_loss: {})'.format(best_val_loss))
                break

        t = time.time()
        model_loss = 0
        adv_fool = 0

# Post-training
toc()
log('Loading best weights')
model.load_weights(log_dir + 'model_best_val_weights.h5')
test_loss = model.evaluate([adj_test, fltr_test, nf_test],
                           [adj_test, nf_test],
                           batch_size=batch_size,
                           verbose=0)[0]
log('Test loss: {:.2f}'.format(test_loss))

# Embeddings
print('Computing operational stream.')
embeddings_train = model.encode([adj, fltr, nf])
embeddings_live = model.encode([adj_live, fltr_live, nf_live])

# Save embeddings dataset
print('Saving operational stream.')
joblib.dump([embeddings_train, embeddings_live, y_live], log_dir + 'dataset.pkl')
