import tensorflow
import tensorflow_proability

class EBF(tfp.positive_semidefinite_kernels.PositiveSemidefiniteKernel):
    def __init__(
        self,
        var_s=None,
        sigma_u=None,
        sigma_v=None,
        phi=None,
        feature_ndims=1,
        name="ExponentiatedEllipticalQuadratic"):
        
        dtype = tf.float64

        var_s = tf.convert_to_tensor(value=amplitude, name='amplitude', dtype=dtype)
        sigma_u = tf.convert_to_tensor(value=length_scale, name='length_scale', dtype=dtype)
        sigma_v = tf.convert_to_tensor(value=amplitude, name='amplitude', dtype=dtype)
        phi = tf.convert_to_tensor(value=length_scale, name='length_scale', dtype=dtype)
        
    @property
    def var_s(self):
        """Amplitude parameter."""
        return self._var_s

    @property
    def sigma_u(self):
        """Length scale parameter."""
        return self._sigma_u
    
    @property
    def sigma_v(self):
        """Amplitude parameter."""
        return self._sigma_v

    @property
    def phi(self):
        """Length scale parameter."""
        return self._phi

    def _batch_shape(self):
        scalar_shape = tf.TensorShape([])
        return tf.broadcast_static_shape(
            scalar_shape if self.var_s is None else self.var_s.shape,
            scalar_shape if self.sigma_u is None else self.sigma_u.shape,
            scalar_shape if self.sigma_v is None else self.sigma_v.shape,
            scalar_shape if self.phi is None else self.phi.shape)

    def _batch_shape_tensor(self):
        return tf.broadcast_dynamic_shape(
            [] if self.var_s is None else tf.shape(input=self.var_s),
            [] if self.sigma_u is None else tf.shape(input=self.sigma_u),
            [] if self.sigma_v is None else tf.shape(input=self.sigma_v),
            [] if self.phi is None else tf.shape(input=self.phi))

    def _apply(self, x1, x2, example_ndims=0):
        exponent = -0.5 * util.sum_rightmost_ndims_preserving_shape(tf.math.squared_difference(x1, x2), self.feature_ndims)
        if self.length_scale is not None:
            length_scale = util.pad_shape_with_ones(
            self.length_scale, example_ndims)
            exponent /= length_scale**2

        if self.amplitude is not None:
            amplitude = util.pad_shape_with_ones(
            self.amplitude, example_ndims)
            exponent += 2. * tf.math.log(amplitude)

        return tf.exp(exponent)