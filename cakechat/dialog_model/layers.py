import theano.tensor as T
from lasagne.layers.base import MergeLayer, Layer
from six.moves import xrange


class RepeatLayer(Layer):
    """
    Layer that repeats input n times along 1 axis and reshapes.
    The idea is to take some data for each object in the batch and repeat n times along the sequence axis.
    For example for repeating thought vector returned by encoder to feed into decoder in SEQ2SEQ model.

    input: tensor of shape N_1 x N_2 x ... x N_D
    output: tensor of shape N_1 x n x N_2 x ... x N_D
    """

    def __init__(self, incoming, n, **kwargs):
        super(RepeatLayer, self).__init__(incoming, **kwargs)
        self._n = n

    def get_output_shape_for(self, input_shape):
        repeat_times = None if isinstance(self._n, T.TensorVariable) else self._n
        return tuple([input_shape[0], repeat_times] + list(input_shape[1:]))

    def get_output_for(self, input, **kwargs):
        new_shape = [input.shape[0], 1] + [input.shape[k] for k in xrange(1, input.ndim)]

        output = T.reshape(input, new_shape, ndim=input.ndim + 1)  # see the details in pydoc
        output = T.repeat(output, self._n, axis=1)
        return output


class NotEqualMaskLayer(Layer):
    """
    Layer that outputs binary matrix according to elementwise non-equality to a specific element
    """

    def __init__(self, incoming, x, **kwargs):
        super(NotEqualMaskLayer, self).__init__(incoming, **kwargs)
        self._x = x

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        return T.neq(input, self._x)


class SwitchLayer(MergeLayer):
    """
    Layer that performs switching from one input to another according to the condition which is theano.iscalar that contains 0 or 1.
    If condition contains 1 then the output will be the output of the first layer in incomings.
    If condition contains 0 then the output will be the output of the second layer in incomings.
    """

    def __init__(self, incomings, condition, **kwargs):
        super(SwitchLayer, self).__init__(incomings, **kwargs)
        self._condition = condition

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]

    def get_output_for(self, inputs, **kwargs):
        output = T.switch(self._condition, inputs[0], inputs[1])
        return output
