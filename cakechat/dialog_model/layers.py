from keras.layers import K, RepeatVector, Lambda


def repeat_vector(inputs):
    """
    Temporary solution:
    Use this function within a Lambda layer to get a repeated layer with a variable 1-st dimension (seq_len).
    May be useful to further feed it to a Concatenate layer.

    inputs == (layer_for_repeat, layer_for_getting_rep_num):
        layer_for_repeat:           shape == (batch_size, vector_dim)
        layer_for_getting_rep_num:  shape == (batch_size, seq_len, ...)
    :return:
        repeated layer_for_repeat, shape == (batch_size, seq_len, vector_dim)
    """
    layer_for_repeat, layer_for_getting_rep_num = inputs
    repeated_vector = RepeatVector(
        n=K.shape(layer_for_getting_rep_num)[1], name='custom_repeat_vector')(layer_for_repeat)
    # shape == (batch_size, seq_len, vector_dim)
    return repeated_vector


def softmax_with_temperature(logits, temperature):
    """
    :param logits:      shape == (batch_size, seq_len, vocab_size), float32
    :param temperature: shape == (batch_size, 1), float32
    :return:
        transformed tokens probs, shape == (batch_size, seq_len, vocab_size), float32
    """

    def softmax_with_temp(args):
        logits, temperature = args
        repeat_num = K.shape(logits)[1]
        temperature_repeated = RepeatVector(repeat_num)(temperature)
        # shape == (batch_size, seq_len)
        scaled_logits = logits / temperature_repeated
        # shape == (batch_size, seq_len, vocab_size)

        # for numerical stability (e.g. for low temperatures):
        scaled_logits = scaled_logits - K.max(scaled_logits, axis=2, keepdims=True)
        # shape == (batch_size, seq_len, vocab_size)
        transformed_probs = K.softmax(scaled_logits)
        # shape == (batch_size, seq_len, vocab_size)
        return transformed_probs

    # wrap transformation in Lambda to turn the result to Keras layer
    transformed_probs = Lambda(
        function=softmax_with_temp,
        mask=lambda inputs, inputs_masks: inputs_masks[0],  # function to get mask of the first input
        name='softmax_with_temperature')([logits, temperature])
    # output shape == (batch_size, seq_len, vocab_size)

    return transformed_probs
