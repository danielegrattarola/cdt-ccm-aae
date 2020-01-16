import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, BatchNormalization, Activation, Reshape, \
    Concatenate, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.utils import multi_gpu_model
from spektral.layers import GraphConv, EdgeConditionedConv, GlobalAttentionPool

from src.utils.layers import CCMProjection


class GAE_CCM(object):
    """
    A graph autoencoder with support for combined CCMs.
    """
    def __init__(self, N, F, S=None, latent_space=128, radius=(-1., 0., 1.),
                 dropout_rate=0.0, l2_reg=5e-4, multi_gpu=False):
        self.N = N                        # Number of nodes in a graph
        self.F = F                        # Number of node features
        self.S = S                        # Number of edge features
        self.dropout_rate = dropout_rate  # Dropout rate between convolutions
        self.l2_reg = l2_reg              # L2 regularization for convolutions
        self.latent_space = latent_space  # Dimensionality of a single CCM

        if isinstance(radius, int) or isinstance(radius, float):
            self._radius = [radius]
        elif isinstance(radius, list) or isinstance(radius, tuple):
            self._radius = radius
        else:
            raise TypeError('Radius must be either a single value or a list'
                            'of values.')

        # Model definition
        if self.S is not None:
            self.model, self.encoder, self.decoder, self.clipper = self._model_builder_ecc()
        else:
            self.model, self.encoder, self.decoder, self.clipper = self._model_builder_gcn()

        if multi_gpu:
            try:
                self.model = multi_gpu_model(self.model)
            except ValueError:
                # Only one GPU available
                pass

    def __getattr__(self, item):
        if item == 'radius':
            to_ret = []
            for i_, r_ in enumerate(self._radius):
                if r_ is None or r_ in {'spherical', 'hyperbolic'}:
                    to_ret.append(self.model.get_layer('z_{}'.format(i_)).get_weights()[0])
                else:
                    to_ret.append(r_)
            return to_ret
        else:
            return getattr(self.model, item)

    def compile(self, optimizer, loss=None, **kwargs):
        self.model.compile(optimizer, loss=loss, **kwargs)

    def encode(self, *args, clip=True, **kwargs):
        if clip:
            return self.clipper.predict(*args, **kwargs)
        else:
            return self.encoder.predict(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder.predict(*args, **kwargs)

    def _model_builder_ecc(self):
        gc1_channels = 32
        gc2_channels = 64
        full_latent_space = len(self._radius) * self.latent_space

        # Inputs
        adj_in = Input(shape=(self.N, self.N), name='adj_in')
        nf_in = Input(shape=(self.N, self.F), name='nf_in')
        ef_in = Input(shape=(self.N, self.N, self.S), name='ef_in')
        z_in = Input(shape=(full_latent_space, ), name='z_in')

        # Encoder
        gc1 = EdgeConditionedConv(gc1_channels, kernel_regularizer=l2(self.l2_reg),
                                  name='ecc1')([nf_in, adj_in, ef_in])
        bn1 = BatchNormalization()(gc1)
        relu1 = Activation('relu')(bn1)
        do1 = Dropout(self.dropout_rate)(relu1)

        gc2 = EdgeConditionedConv(gc2_channels, kernel_regularizer=l2(self.l2_reg),
                                  name='ecc2')([do1, adj_in, ef_in])
        bn2 = BatchNormalization()(gc2)
        relu2 = Activation('relu')(bn2)
        do2 = Dropout(self.dropout_rate)(relu2)

        pool = GlobalAttentionPool(128, name='attn_pool')(do2)

        z_enc_list = []
        z_clip_list = []
        for _r in self._radius:
            z_1 = Dense(128, activation='relu')(pool)
            z_2 = Dense(self.latent_space, activation='linear')(z_1)
            z_3 = CCMProjection(_r)(z_2)
            z_enc_list.append(z_2)
            z_clip_list.append(z_3)

        if len(self._radius) > 1:
            z_enc = Concatenate(name='z_enc')(z_enc_list)
            z_clip = Concatenate(name='z_clip')(z_clip_list)
        else:
            z_enc = z_enc_list[0]
            z_clip = z_clip_list[0]

        # Decoder
        dense3 = Dense(128)(z_in)
        bn3 = BatchNormalization()(dense3)
        relu3 = Activation('relu')(bn3)

        dense4 = Dense(256)(relu3)
        bn4 = BatchNormalization()(dense4)
        relu4 = Activation('relu')(bn4)

        dense5 = Dense(512)(relu4)
        bn5 = BatchNormalization()(dense5)
        relu5 = Activation('relu')(bn5)

        # Output
        adj_out_pre = Dense(self.N * self.N, activation='sigmoid')(relu5)
        adj_out = Reshape((self.N, self.N), name='adj_out')(adj_out_pre)

        nf_out_pre = Dense(self.N * self.F, activation='linear')(relu5)
        nf_out = Reshape((self.N, self.F), name='nf_out')(nf_out_pre)

        ef_out_pre = Dense(self.N * self.N * self.S, activation='linear')(relu5)
        ef_out = Reshape((self.N, self.N, self.S), name='ef_out')(ef_out_pre)

        # Build models
        encoder = Model(inputs=[adj_in, nf_in, ef_in], outputs=z_enc)
        clipper = Model(inputs=[adj_in, nf_in, ef_in], outputs=z_clip)
        decoder = Model(inputs=z_in, outputs=[adj_out, nf_out, ef_out])
        model = Model(inputs=[adj_in, nf_in, ef_in], outputs=decoder(clipper.output))
        model.output_names = ['adj', 'nf', 'ef']

        return model, encoder, decoder, clipper

    def _model_builder_gcn(self):
        gc1_channels = 32
        gc2_channels = 64
        full_latent_space = len(self._radius) * self.latent_space

        # Input
        fltr_in = Input(shape=(self.N, self.N), name='fltr_in')
        nf_in = Input(shape=(self.N, self.F), name='nf_in')
        z_in = Input(shape=(full_latent_space, ), name='z_in')

        # Encoder
        gc1 = GraphConv(gc1_channels, kernel_regularizer=l2(self.l2_reg), name='gc1')([nf_in, fltr_in])
        bn1 = BatchNormalization()(gc1)
        relu1 = Activation('relu')(bn1)
        do1 = Dropout(self.dropout_rate)(relu1)

        gc2 = GraphConv(gc2_channels, kernel_regularizer=l2(self.l2_reg), name='gc2')([do1, fltr_in])
        bn2 = BatchNormalization()(gc2)
        relu2 = Activation('relu')(bn2)
        do2 = Dropout(self.dropout_rate)(relu2)

        pool = GlobalAttentionPool(128, name='attn_pool')(do2)

        z_enc_list = []
        z_clip_list = []
        for _r in self._radius:
            z_1 = Dense(128, activation='relu')(pool)
            z_2 = Dense(self.latent_space, activation='linear')(z_1)
            z_3 = CCMProjection(_r)(z_2)
            z_enc_list.append(z_2)
            z_clip_list.append(z_3)

        if len(self._radius) > 1:
            z_enc = Concatenate(name='z_enc')(z_enc_list)
            z_clip = Concatenate(name='z_clip')(z_clip_list)
        else:
            z_enc = z_enc_list[0]
            z_clip = z_clip_list[0]

        # Decoder
        dense3 = Dense(128)(z_in)
        bn3 = BatchNormalization()(dense3)
        relu3 = Activation('relu')(bn3)

        dense4 = Dense(256)(relu3)
        bn4 = BatchNormalization()(dense4)
        relu4 = Activation('relu')(bn4)

        dense5 = Dense(512)(relu4)
        bn5 = BatchNormalization()(dense5)
        relu5 = Activation('relu')(bn5)

        # Output
        adj_out_pre = Dense(self.N * self.N, activation='sigmoid')(relu5)
        adj_out = Reshape((self.N, self.N), name='adj_out')(adj_out_pre)

        nf_out_pre = Dense(self.N * self.F, activation='linear')(relu5)
        nf_out = Reshape((self.N, self.F), name='nf_out')(nf_out_pre)

        # Build models
        encoder = Model(inputs=[fltr_in, nf_in], outputs=z_enc)
        clipper = Model(inputs=[fltr_in, nf_in], outputs=z_clip)
        decoder = Model(inputs=z_in, outputs=[adj_out, nf_out])
        model = Model(inputs=[fltr_in, nf_in], outputs=decoder(clipper.output))
        model.output_names = ['adj', 'nf']

        return model, encoder, decoder, clipper

    @staticmethod
    def geom_reg(r, sigma, l=0.05):
        # To use, set activity_regularizer=self.geom_reg(r_, sigma_) when
        # instantiating z_2 in _model_builder functions
        def geom_regularizer(x):
            sign = np.sign(r)
            free_components = x[..., :-1] ** 2
            bound_component = sign * x[..., -1:] ** 2
            all_components = K.concatenate((free_components, bound_component),
                                           -1)
            ext_product = K.sum(all_components, -1)[..., None]
            output_pre = K.exp(-(ext_product - sign * r ** 2) ** 2 / (2 * sigma ** 2))
            if sign == 0.:
                output_pre = K.zeros_like(output_pre)
            return l * K.sum(1. - output_pre)

        return geom_regularizer
