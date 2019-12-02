from __future__ import absolute_import

import numpy as np
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.constraints import Constraint
from keras.layers import Layer, Average, Concatenate


class CCMProjection(Layer):
    """
    Projects a tensor to a CCM depending on the value of `r`. Optionally,
    `r` can be learned via backpropagation.

    **Input**

    - tensor of shape `(batch_size, input_dim)`.

    **Output**

    - tensor of shape `(batch_size, input_dim)`, where each sample along the
    0th axis is projected to the CCM.

    :param r: radius of the CCM. If r is a number, then use it as fixed
    radius. If `r='spherical'`, use a trainable weight as radius, with a
    positivity constraint. If `r='hyperbolic'`, use a trainable weight
    as radius, with a negativity constraint. If `r=None`, use a trainable
    weight as radius, with no constraints (points will be projected to the
    correct manifold based on the sign of the weight).
    :param kernel_initializer: initializer for the kernel matrix;
    :param kernel_regularizer: regularization applied to the kernel matrix;
    :param kernel_constraint: constraint applied to the kernel matrix.
    """

    def __init__(self,
                 r=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None, **kwargs):
        super(CCMProjection, self).__init__(**kwargs)
        self.radius = r
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        if self.radius == 'spherical':
            self.kernel_constraint = self.Pos()
        elif self.radius == 'hyperbolic':
            self.kernel_constraint = self.Neg()
        else:
            self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        if self.radius in {'spherical', 'hyperbolic'} or self.radius is None:
            self.radius = self.add_weight(shape=(),
                                          initializer=self.kernel_initializer,
                                          name='radius',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        else:
            self.radius = K.constant(self.radius)
        self.built = True

    def call(self, inputs):
        zero = K.constant(0.)

        # Spherical clip
        spherical_clip = self.radius * K.l2_normalize(inputs, -1)
        # Hyperbolic clip
        free_components = inputs[..., :-1]
        bound_component = K.sqrt(K.sum(free_components ** 2, -1)[..., None] + (self.radius ** 2))
        hyperbolic_clip = K.concatenate((free_components, bound_component), -1)

        lt_cond = K.less(self.radius, zero)
        lt_check = K.switch(lt_cond, hyperbolic_clip, inputs)

        gt_cond = K.greater(self.radius, zero)
        output = K.switch(gt_cond, spherical_clip, lt_check)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self, **kwargs):
        config = {}
        base_config = super(CCMProjection, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    class Pos(Constraint):
        """Constrains a scalar weight to be positive.
        """

        def __call__(self, w):
            return K.maximum(w, K.epsilon())

    class Neg(Constraint):
        """Constrains a scalar weight to be negative.
        """

        def __call__(self, w):
            return K.minimum(w, -K.epsilon())


class CCMMembership(Layer):
    """
    Computes the membership of the given points to a constant-curvature
    manifold of radius `r`, as:
    $$
        \\mu(x) = \\mathrm{exp}\\left(\\cfrac{-\\big( \\langle \\vec x, \\vec x \\rangle - r^2 \\big)^2}{2\\sigma^2}\\right).
    $$

    If `r=0`, then \(\\mu(x) = 1\).
    If more than one radius is given, inputs are evenly split across the
    last dimension and membership is computed for each radius-slice pair.
    The output membership is returned according to the `mode` option.

    **Input**

    - tensor of shape `(batch_size, input_dim)`;

    **Output**

    - tensor of shape `(batch_size, output_size)`, where `output_size` is
    computed according to the `mode` option;.

    :param r: int ot list, radia of the CCMs.
    :param mode: 'average' to return the average membership across CCMs, or
    'concat' to return the membership for each CCM concatenated;
    :param sigma: spread of the membership curve;
    """

    def __init__(self, r=1., mode='average', sigma=1., **kwargs):
        super(CCMMembership, self).__init__(**kwargs)
        if isinstance(r, int) or isinstance(r, float):
            self.r = [r]
        elif isinstance(r, list) or isinstance(r, tuple):
            self.r = r
        else:
            raise TypeError('r must be either a single value, or a list/tuple '
                            'of values.')
        possible_modes = {'average', 'concat'}
        if mode not in possible_modes:
            raise ValueError('Possible modes: {}'.format(possible_modes))
        self.mode = mode
        self.sigma = sigma

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        output_part = []
        manifold_size = K.int_shape(inputs)[-1] // len(self.r)

        for idx, r_ in enumerate(self.r):
            start = idx * manifold_size
            stop = start + manifold_size
            part = inputs[..., start:stop]
            sign = np.sign(r_)
            if sign == 0.:
                # This is weird but necessary to make the layer differentiable
                output_pre = K.sum(inputs, -1, keepdims=True) * 0. + 1.
            else:
                free_components = part[..., :-1] ** 2
                bound_component = sign * part[..., -1:] ** 2
                all_components = K.concatenate((free_components, bound_component), -1)
                ext_product = K.sum(all_components, -1, keepdims=True)
                output_pre = K.exp(-(ext_product - sign * r_ ** 2) ** 2 / (2 * self.sigma ** 2))

            output_part.append(output_pre)

        if len(output_part) >= 2:
            if self.mode == 'average':
                output = Average()(output_part)
            elif self.mode == 'concat':
                output = Concatenate()(output_part)
            else:
                raise ValueError()  # Never gets here
        else:
            output = output_part[0]

        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:-1] + (1,)
        return output_shape

    def get_config(self, **kwargs):
        config = {}
        base_config = super(CCMMembership, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
