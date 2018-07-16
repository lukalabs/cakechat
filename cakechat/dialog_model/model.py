import os
from collections import OrderedDict

import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne.init import Normal
from lasagne.layers import InputLayer, DenseLayer, GRULayer, reshape, EmbeddingLayer, SliceLayer, ConcatLayer, \
    DropoutLayer, get_output, get_all_params, get_all_param_values, set_all_param_values, get_all_layers, \
    get_output_shape
from lasagne.objectives import categorical_crossentropy
from six.moves import xrange

from cakechat.config import HIDDEN_LAYER_DIMENSION, GRAD_CLIP, LEARNING_RATE, \
    TRAIN_WORD_EMBEDDINGS_LAYER, WORD_EMBEDDING_DIMENSION, ENCODER_DEPTH, DECODER_DEPTH, DENSE_DROPOUT_RATIO, \
    CONDITION_EMBEDDING_DIMENSION, NN_MODEL_PREFIX, BASE_CORPUS_NAME, INPUT_CONTEXT_SIZE, INPUT_SEQUENCE_LENGTH, \
    OUTPUT_SEQUENCE_LENGTH, NN_MODELS_DIR
from cakechat.dialog_model.layers import RepeatLayer, NotEqualMaskLayer, SwitchLayer
from cakechat.utils.files_utils import DummyFileResolver, ensure_dir, FileNotFoundException
from cakechat.utils.logger import get_logger, laconic_logger
from cakechat.utils.text_processing import SPECIAL_TOKENS

_logger = get_logger(__name__)


class CakeChatModel(object):
    def __init__(self,
                 index_to_token,
                 index_to_condition,
                 model_init_path=None,
                 nn_models_dir=NN_MODELS_DIR,
                 model_prefix=NN_MODEL_PREFIX,
                 corpus_name=BASE_CORPUS_NAME,
                 skip_token=SPECIAL_TOKENS.PAD_TOKEN,
                 learning_rate=LEARNING_RATE,
                 grad_clip=GRAD_CLIP,
                 hidden_layer_dim=HIDDEN_LAYER_DIMENSION,
                 encoder_depth=ENCODER_DEPTH,
                 decoder_depth=DECODER_DEPTH,
                 init_embedding=None,
                 word_embedding_dim=WORD_EMBEDDING_DIMENSION,
                 train_word_embedding=TRAIN_WORD_EMBEDDINGS_LAYER,
                 dense_dropout_ratio=DENSE_DROPOUT_RATIO,
                 condition_embedding_dim=CONDITION_EMBEDDING_DIMENSION,
                 is_reverse_model=False):
        """
        :param index_to_token: Dict with tokens and indices for neural network
        :param model_init_path: Path to weights file to be used for model's intialization
        :param skip_token: Token to skip with masking. Id of this token is inferred from index_to_token dictionary
        :param learning_rate: Learning rate factor for the optimization algorithm
        :param grad_clip: Clipping parameter to prevent gradient explosion
        :param init_embedding: Matrix to initialize word-embedding layer. Default value is random standart-gaussian
            initialization
        """
        self._index_to_token = index_to_token
        self._token_to_index = {v: k for k, v in index_to_token.items()}
        self._vocab_size = len(self._index_to_token)

        self._index_to_condition = index_to_condition
        self._condition_to_index = {v: k for k, v in index_to_condition.items()}
        self._condition_ids_num = len(self._condition_to_index)
        self._condition_embedding_dim = condition_embedding_dim

        self._learning_rate = learning_rate
        self._grad_clip = grad_clip

        self._W_init_embedding = Normal() if init_embedding is None else init_embedding
        self._word_embedding_dim = word_embedding_dim
        self._train_word_embedding = train_word_embedding
        self._skip_token_id = self._token_to_index[skip_token]

        self._hidden_layer_dim = hidden_layer_dim
        self._encoder_depth = encoder_depth
        self._decoder_depth = decoder_depth
        self._dense_dropout_ratio = dense_dropout_ratio

        self._nn_models_dir = nn_models_dir
        self._model_prefix = model_prefix
        self._corpus_name = corpus_name
        self._is_reverse_model = is_reverse_model

        self._model_load_path = model_init_path or self.model_save_path

        self._train_fn = None  # Training functions are compiled as needed
        self._build_model_computational_graph()
        self._compile_theano_functions_for_prediction()

    @property
    def params_str(self):
        params_str = 'gru' \
                     '_hd{hidden_dim}' \
                     '_cdim{condition_dimension}' \
                     '_drop{dropout_ratio}' \
                     '_encd{encoder_depth}' \
                     '_decd{decoder_depth}' \
                     '_il{input_seq_len}' \
                     '_cs{input_cont_size}' \
                     '_ansl{output_seq_len}' \
                     '_lr{learning_rate}' \
                     '_gc{gradient_clip}' \
                     '_{learn_emb}'

        return params_str.format(
            hidden_dim=self._hidden_layer_dim,
            condition_dimension=self._condition_embedding_dim,
            encoder_depth=self._encoder_depth,
            decoder_depth=self._decoder_depth,
            input_seq_len=INPUT_SEQUENCE_LENGTH,
            input_cont_size=INPUT_CONTEXT_SIZE,
            output_seq_len=OUTPUT_SEQUENCE_LENGTH,
            dropout_ratio=self._dense_dropout_ratio,
            learning_rate=self._learning_rate,
            gradient_clip=self._grad_clip,
            learn_emb='learnemb' if self._train_word_embedding else 'fixemb'
        )

    @property
    def model_name(self):
        suffix = ['reverse'] if self._is_reverse_model else []
        params_str = '_'.join([
            self._model_prefix,
            self._corpus_name,
            self.params_str
        ] + suffix)
        return params_str

    @property
    def model_save_path(self):
        return os.path.join(self._nn_models_dir, self.model_name)

    def _build_model_computational_graph(self):
        self._net = OrderedDict()
        self._add_word_embeddings()
        self._add_condition_embeddings()
        self._add_utterance_encoder()
        self._add_context_encoder()
        self._add_decoder()
        self._add_output_dense()

    def _compile_theano_functions_for_prediction(self):
        self._temperature = T.fscalar('temperature')  # theano variable needed for prediction
        self.predict_prob = self._get_predict_fn(logarithm_output_probs=False)
        self.predict_prob_one_step = self._get_predict_one_step_fn(logarithm_output_probs=False)
        self.predict_log_prob = self._get_predict_fn(logarithm_output_probs=True)
        self.predict_log_prob_one_step = self._get_predict_one_step_fn(logarithm_output_probs=True)
        self.predict_sequence_score = self._get_predict_sequence_score_fn()
        self.predict_sequence_score_by_thought_vector = self._get_predict_sequence_score_by_thought_vector_fn()
        self.get_thought_vectors = self._get_thought_vectors_fn()

    def _add_word_embeddings(self):
        self._net['input_x'] = InputLayer(
            shape=(None, None, None), input_var=T.itensor3(name='input_x'), name='input_x')

        self._net['input_y'] = InputLayer(shape=(None, None), input_var=T.imatrix(name='input_y'), name='input_y')

        # Infer these variables from data passed to computation graph since batch shape may differ in training and
        # prediction phases
        self._batch_size = self._net['input_x'].input_var.shape[0]
        self._input_context_size = self._net['input_x'].input_var.shape[1]
        self._input_seq_len = self._net['input_x'].input_var.shape[2]
        self._output_seq_len = self._net['input_y'].input_var.shape[1]

        self._net['input_x_batched'] = \
            reshape(self._net['input_x'], (self._batch_size * self._input_context_size, self._input_seq_len))

        self._net['input_x_mask'] = NotEqualMaskLayer(
            incoming=self._net['input_x_batched'], x=self._skip_token_id, name='mask_x')

        self._net['emb_x'] = EmbeddingLayer(
            incoming=self._net['input_x_batched'],
            input_size=self._vocab_size,
            output_size=self._word_embedding_dim,
            W=self._W_init_embedding,
            name='emb_x')
        # output shape (batch_size, input_context_size, input_seq_len, embedding_dimension)

        self._net['input_y_mask'] = NotEqualMaskLayer(
            incoming=self._net['input_y'], x=self._skip_token_id, name='mask_y')

        self._net['emb_y'] = EmbeddingLayer(
            incoming=self._net['input_y'],
            input_size=self._vocab_size,
            output_size=self._word_embedding_dim,
            W=self._W_init_embedding,
            name='emb_y')
        # output shape (batch_size, output_seq_len, embedding_dimension)

        if not self._train_word_embedding:
            self._net['emb_x'].params[self._net['emb_x'].W].remove('trainable')
            self._net['emb_y'].params[self._net['emb_y'].W].remove('trainable')

    def _add_forward_backward_encoder_layer(self):
        is_single_layer_encoder = self._encoder_depth == 1
        return_only_final_state = is_single_layer_encoder

        # input shape = (batch_size * input_context_size, input_seq_len, embedding_dimension)
        self._net['enc_forward'] = GRULayer(
            incoming=self._net['emb_x'],
            num_units=self._hidden_layer_dim,
            grad_clipping=self._grad_clip,
            only_return_final=return_only_final_state,
            name='encoder_forward',
            mask_input=self._net['input_x_mask'])
        # output shape = (batch_size * input_context_size, input_seq_len, hidden_layer_dimension)
        #             or (batch_size * input_context_size, hidden_layer_dimension)

        # input shape = (batch_size * input_context_size, input_seq_len, embedding_dimension)
        self._net['enc_backward'] = GRULayer(
            incoming=self._net['emb_x'],
            num_units=self._hidden_layer_dim,
            grad_clipping=self._grad_clip,
            only_return_final=return_only_final_state,
            backwards=True,
            name='encoder_backward',
            mask_input=self._net['input_x_mask'])
        # output shape = (batch_size * input_context_size, input_seq_len, hidden_layer_dimension)
        #             or (batch_size * input_context_size, hidden_layer_dimension)

        self._net['enc_0'] = ConcatLayer(
            incomings=[self._net['enc_forward'], self._net['enc_backward']],
            axis=1 if return_only_final_state else 2,
            name='encoder_bidirectional_concat')
        # output shape = (batch_size * input_context_size, input_seq_len, 2 * hidden_layer_dimension)
        #             or (batch_size * input_context_size, 2 * hidden_layer_dimension)

    def _add_condition_embeddings(self):
        self._net['input_condition_id'] = InputLayer(
            shape=(None, ), input_var=T.ivector(name='in_condition_id'), name='input_condition_id')

        self._net['emb_condition_id'] = EmbeddingLayer(
            incoming=self._net['input_condition_id'],
            input_size=self._condition_ids_num,
            output_size=self._condition_embedding_dim,
            name='embedding_condition_id')

    def _add_utterance_encoder(self):
        # input shape = (batch_size * input_context_size, input_seq_len, embedding_dimension)
        self._add_forward_backward_encoder_layer()

        for enc_layer_id in xrange(1, self._encoder_depth):
            is_last_encoder_layer = enc_layer_id == self._encoder_depth - 1
            return_only_final_state = is_last_encoder_layer

            # input shape = (batch_size * input_context_size, input_seq_len, embedding_dimension)
            self._net['enc_' + str(enc_layer_id)] = GRULayer(
                incoming=self._net['enc_' + str(enc_layer_id - 1)],
                num_units=self._hidden_layer_dim,
                grad_clipping=self._grad_clip,
                only_return_final=return_only_final_state,
                name='encoder_' + str(enc_layer_id),
                mask_input=self._net['input_x_mask'])

        self._net['enc'] = self._net['enc_' + str(self._encoder_depth - 1)]

        # output shape = (batch_size * input_context_size, hidden_layer_dim)

    def _add_context_encoder(self):
        self._net['batched_enc'] = reshape(
            self._net['enc'], (self._batch_size, self._input_context_size, get_output_shape(self._net['enc'])[-1]))

        self._net['context_enc'] = GRULayer(
            incoming=self._net['batched_enc'],
            num_units=self._hidden_layer_dim,
            grad_clipping=self._grad_clip,
            only_return_final=True,
            name='context_encoder')

        self._net['switch_enc_to_tv'] = T.iscalar(name='switch_enc_to_tv')

        self._net['thought_vector'] = InputLayer(
            shape=(None, self._hidden_layer_dim), input_var=T.fmatrix(name='thought_vector'), name='thought_vector')

        self._net['enc_result'] = SwitchLayer(
            incomings=[self._net['thought_vector'], self._net['context_enc']], condition=self._net['switch_enc_to_tv'])

        # We need the following to pass as 'givens' argument when compiling theano functions:
        self._default_thoughts_vector = T.zeros((self._batch_size, self._hidden_layer_dim))
        self._default_input_x = T.zeros(shape=(self._net['thought_vector'].input_var.shape[0], 1, 1), dtype=np.int32)

    def _add_decoder(self):
        """
        Decoder returns the batch of sequences of thought vectors, each corresponds to a decoded token
        reshapes this 3d tensor to 2d matrix so that the next Dense layer can convert each thought vector to
        a probability distribution vector
        """

        self._net['hid_states_decoder'] = InputLayer(
            shape=(None, self._decoder_depth, None),
            input_var=T.tensor3('hid_inits_decoder'),
            name='hid_states_decoder')

        # repeat along the sequence axis output_seq_len times, where output_seq_len is inferred from input tensor
        self._net['enc_repeated'] = RepeatLayer(
            incoming=self._net['enc_result'],  # input shape = (batch_size, encoder_output_dimension)
            n=self._output_seq_len,
            name='repeat_layer')

        self._net['emb_condition_id_repeated'] = RepeatLayer(
            incoming=self._net['emb_condition_id'], n=self._output_seq_len, name='embedding_condition_id_repeated')

        self._net['dec_concated_input'] = ConcatLayer(
            incomings=[self._net['emb_y'], self._net['enc_repeated'], self._net['emb_condition_id_repeated']],
            axis=2,
            name='decoder_concated_input')
        # shape = (batch_size, input_seq_len, encoder_output_dimension)

        self._net['dec_0'] = self._net['dec_concated_input']

        for dec_layer_id in xrange(1, self._decoder_depth + 1):
            # input shape = (batch_size, input_seq_len, embedding_dimension + hidden_dimension)
            self._net['dec_' + str(dec_layer_id)] = GRULayer(
                incoming=self._net['dec_' + str(dec_layer_id - 1)],
                num_units=self._hidden_layer_dim,
                grad_clipping=self._grad_clip,
                only_return_final=False,
                name='decoder_' + str(dec_layer_id),
                mask_input=self._net['input_y_mask'],
                hid_init=SliceLayer(self._net['hid_states_decoder'], dec_layer_id - 1, axis=1))

        self._net['dec'] = self._net['dec_' + str(self._decoder_depth)]
        # output shape = (batch_size, output_seq_len, hidden_dimension)

    @staticmethod
    def _remove_all_last_tokens(input_layer, unflatten_sequences_shape, flatten_input=True):
        """
        Helper function that creates a sequence of layers to clean up the tensor from all elements,
        corresponding to the last token in the sequence
        """
        new_flattened_shape = (unflatten_sequences_shape[0] * (unflatten_sequences_shape[1] - 1),) \
                              + unflatten_sequences_shape[2:]

        sliced = SliceLayer(
            incoming=reshape(input_layer, unflatten_sequences_shape) if flatten_input else input_layer,
            indices=slice(0, -1),
            axis=1)  # sequence axis
        return reshape(sliced, new_flattened_shape)

    @staticmethod
    def _remove_all_first_tokens(input_layer, unflatten_sequences_shape, flatten_input=True):
        """
        Helper function that creates a sequence of layers to clean up the tensor from all elements,
        corresponding to the first token in the sequence
        """
        new_flattened_shape = (unflatten_sequences_shape[0] * (unflatten_sequences_shape[1] - 1),) \
                              + unflatten_sequences_shape[2:]

        sliced = SliceLayer(
            incoming=reshape(input_layer, unflatten_sequences_shape) if flatten_input else input_layer,
            indices=slice(1, None),
            axis=1)  # sequence axis
        return reshape(sliced, new_flattened_shape)

    def _add_output_dense(self):
        """
        Adds a dense layer on top of the decoder to convert hidden_state vector to probs distribution over vocabulary.
        For every prob sequence last prob vectors are cut off since they correspond
        to the tokens that go after EOS_TOKEN and we are not interested in them.
        Doesn't need to reshape back the cut tensor since it's convenient to compare
        this "long" output with true one-hot vectors.
        """
        self._net['dec_dropout'] = DropoutLayer(
            incoming=reshape(self._net['dec'], (-1, self._hidden_layer_dim)),
            p=self._dense_dropout_ratio,
            name='decoder_dropout_layer')

        self._net['target'] = self._remove_all_first_tokens(
            self._net['input_y'],
            unflatten_sequences_shape=(self._batch_size, self._output_seq_len),
            flatten_input=False)

        self._net['dec_dropout_nolast'] = self._remove_all_last_tokens(
            self._net['dec_dropout'],
            unflatten_sequences_shape=(self._batch_size, self._output_seq_len, self._hidden_layer_dim))

        self._net['dist_nolast'] = DenseLayer(
            incoming=self._net['dec_dropout_nolast'],
            num_units=self._vocab_size,
            nonlinearity=lasagne.nonlinearities.softmax,
            name='dense_output_probs')

        dist_layer_params = get_all_params(self._net['dist_nolast'])
        param_name_to_param = {p.name: p for p in dist_layer_params}

        self._net['dist'] = DenseLayer(
            incoming=self._net['dec_dropout'],
            num_units=self._vocab_size,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=param_name_to_param['dense_output_probs.W'],
            b=param_name_to_param['dense_output_probs.b'],
            name='dense_output_probs')
        # output tensor has shape (batch_size * (seq_len - 1), vocab_size)

    def _get_train_fn(self):
        output_probs = get_output(self._net['dist_nolast'])

        mask = get_output(self._net['input_y_mask'])[:, 1:].flatten()
        nonpad_ids = mask.nonzero()

        target_ids = get_output(self._net['target'])
        loss_per_object = categorical_crossentropy(predictions=output_probs, targets=target_ids)
        loss = loss_per_object[nonpad_ids].mean()

        all_params = get_all_params(self._net['dist'], trainable=True)

        _logger.info('Computing train updates...')
        updates = lasagne.updates.adadelta(loss_or_grads=loss, params=all_params, learning_rate=self._learning_rate)

        _logger.info('Compiling train function...')

        train_fn = theano.function(
            inputs=[
                self._net['input_x'].input_var, self._net['input_y'].input_var,
                self._net['input_condition_id'].input_var
            ],
            givens={
                self._net['hid_states_decoder'].input_var:
                    T.zeros((self._batch_size, self._decoder_depth, self._hidden_layer_dim)),
                self._net['thought_vector'].input_var:
                    self._default_thoughts_vector,
                self._net['switch_enc_to_tv']:
                    np.cast[np.int32](False)  # Doesn't compile without explicit casting here
            },
            outputs=loss,
            updates=updates)
        return train_fn

    def train(self, *args):
        if not self._train_fn:
            self._train_fn = self._get_train_fn()

        return self._train_fn(*args)

    def _get_nn_output(self, remove_last_output=True):
        """
        :param remove_last_output: If True, prediction for the last token in the sequence is removed.
         If we predict all the outputs for loss calculation and scoring we need to throw away the last prediction
         If we only want to get the distribution to predict the next token, this removing is unnecessary.
        :return:
        """
        if 'output_probs' not in self._net:
            output_probs = get_output(self._net['dist'], deterministic=True)
            output_probs = T.reshape(output_probs, (self._batch_size, -1, self._vocab_size))
            self._net['output_probs'] = output_probs

        # We remove the last probability in the sequence to match the input.
        # output_probs has shape (batch_size * (seq_len - 1), vocab_size)
        if remove_last_output:
            return self._net['output_probs'][:, :-1, :]
            # output_probs has shape (batch_size, seq_len - 1, vocab_size)
        else:
            return self._net['output_probs']

    @staticmethod
    def _perform_temperature_transform(probs, temperature):
        transformed_log_probs = T.log(probs) / temperature
        # For numerical stability (e.g. for low temperatures:
        transformed_log_probs = transformed_log_probs - T.max(transformed_log_probs, axis=2, keepdims=True)
        # Normalization:
        return T.exp(transformed_log_probs) / T.sum(T.exp(transformed_log_probs), axis=2, keepdims=True)

    def _get_predict_fn(self, logarithm_output_probs):
        output_probs = self._get_nn_output()

        _logger.info('Compiling predict function (log_prob=%s)...' % logarithm_output_probs)

        predict_fn = theano.function(
            inputs=[
                self._net['input_x'].input_var, self._net['input_y'].input_var,
                self._net['input_condition_id'].input_var
            ],
            givens={
                self._net['hid_states_decoder'].input_var:
                    T.zeros((self._batch_size, self._decoder_depth, self._hidden_layer_dim)),
                self._net['thought_vector'].input_var:
                    self._default_thoughts_vector,
                self._net['switch_enc_to_tv']:
                    np.cast[np.int32](False)  # Doesn't compile without explicit casting here
            },
            outputs=T.log(output_probs) if logarithm_output_probs else output_probs)
        return predict_fn

    def _get_predict_one_step_fn(self, logarithm_output_probs):
        output_probs = self._get_nn_output(remove_last_output=False)
        new_hiddens = [
            get_output(self._net['dec_{}'.format(layer_id)], deterministic=True)
            for layer_id in xrange(1, self._decoder_depth + 1)
        ]

        tranformed_output_probs = self._perform_temperature_transform(output_probs, self._temperature)

        _logger.info('Compiling one-step predict function (log_prob=%s)...' % logarithm_output_probs)
        predict_one_step_fn = theano.function(
            inputs=[
                self._net['thought_vector'].input_var, self._net['hid_states_decoder'].input_var,
                self._net['input_y'].input_var, self._net['input_condition_id'].input_var, self._temperature
            ],
            givens={
                self._net['input_x'].input_var: self._default_input_x,
                self._net['switch_enc_to_tv']:
                    np.cast[np.int32](True)  # Doesn't compile without explicit casting here
            },
            outputs=[
                T.concatenate(new_hiddens, axis=1),
                T.log(tranformed_output_probs) if logarithm_output_probs else tranformed_output_probs
            ],
            name='predict_probs_one_step')
        return predict_one_step_fn

    def _get_thought_vectors_fn(self):
        thought_vector = get_output(self._net['context_enc'])

        thought_vector_fn = theano.function(inputs=[self._net['input_x'].input_var], outputs=thought_vector)
        return thought_vector_fn

    def _get_sequence_scores(self):
        # Calculate log-likelihood for batch of data on GPU in symbolic operationse
        probs = self._get_nn_output()
        mask = get_output(self._net['input_y_mask'])
        # All shapes are symbolic and are evaluated on run-time only after input tensors are supplied
        batch_size, seq_len, vocab_size = probs.shape
        total_num_tokens = batch_size * seq_len

        # We need reshape here to do effective slicing without any loops or scans
        probs_long = probs.reshape((total_num_tokens, vocab_size))
        output_ids = self._net['input_y'].input_var[:, 1:]
        mask = mask[:, 1:]  # Do not use first tokens for likelihood computation
        # (these are start tokens: we don't even have probabilities for them)
        token_ids_long = output_ids.reshape((total_num_tokens, ))

        # Select probabilities of only observed tokens and reshape back
        observed_tokens_probs = probs_long[T.arange(total_num_tokens), token_ids_long]
        observed_tokens_log_probs = T.log(observed_tokens_probs)
        nonpad_observed_tokens_log_probs = observed_tokens_log_probs.reshape((batch_size, seq_len)) * mask
        batch_scores = nonpad_observed_tokens_log_probs.sum(axis=1)

        return batch_scores

    def _get_predict_sequence_score_fn(self):
        batch_scores = self._get_sequence_scores()
        _logger.info('Compiling sequence scoring function...')
        predict_score_fn = theano.function(
            inputs=[
                self._net['input_x'].input_var, self._net['input_y'].input_var,
                self._net['input_condition_id'].input_var
            ],
            givens={
                self._net['hid_states_decoder'].input_var:
                    T.zeros((self._batch_size, self._decoder_depth, self._hidden_layer_dim)),
                self._net['thought_vector'].input_var:
                    self._default_thoughts_vector,
                self._net['switch_enc_to_tv']:
                    np.cast[np.int32](False)  # Doesn't compile without explicit casting here
            },
            outputs=batch_scores,
            name='predict_sequence_score')

        return predict_score_fn

    def _get_predict_sequence_score_by_thought_vector_fn(self):
        # We need batch_size symbolic variable to be independent of input_x, to avoid loops in the computational graph
        batch_size = self._net['input_y'].input_var.shape[0]
        batch_scores = self._get_sequence_scores()
        _logger.info('Compiling sequence scoring function (with thought vectors as arguments)...')
        predict_score_fn = theano.function(
            inputs=[
                self._net['thought_vector'].input_var,
                self._net['input_y'].input_var,
                self._net['input_condition_id'].input_var,
            ],
            givens={
                self._net['input_x'].input_var:
                    self._default_input_x,
                self._net['hid_states_decoder'].input_var:
                    T.zeros((batch_size, self._decoder_depth, self._hidden_layer_dim)),
                self._net['switch_enc_to_tv']:
                    np.cast[np.int32](True)  # Doesn't compile without explicit casting here
            },
            outputs=batch_scores,
            name='predict_sequence_score_by_thought_vector')

        return predict_score_fn

    @property
    def index_to_token(self):
        return self._index_to_token

    @property
    def condition_to_index(self):
        return self._condition_to_index

    @property
    def index_to_condition(self):
        return self._index_to_condition

    @property
    def token_to_index(self):
        return self._token_to_index

    @property
    def model_load_path(self):
        return self._model_load_path

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def skip_token_id(self):
        return self._skip_token_id

    @property
    def hidden_layer_dim(self):
        return self._hidden_layer_dim

    @property
    def decoder_depth(self):
        return self._decoder_depth

    @property
    def is_reverse_model(self):
        return self._is_reverse_model

    def load_weights(self):
        with open(self.model_load_path, 'rb') as f:
            loaded_file = np.load(f)
            # Just using .values() would't work here because we need to keep the order of elements
            ordered_params = [loaded_file['arr_%d' % i] for i in xrange(len(loaded_file.files))]
        set_all_param_values(self._net['dist'], ordered_params)

    def save_model(self, save_model_path):
        ensure_dir(os.path.dirname(save_model_path))
        ordered_params = get_all_param_values(self._net['dist'])

        with open(save_model_path, 'wb') as f:
            np.savez(f, *ordered_params)

        _logger.info('\nSaved model:\n{}\n'.format(save_model_path))

    @staticmethod
    def delete_model(delete_path):
        if not os.path.isfile(delete_path):
            _logger.warning('Couldn\'t delete model. File not found:\n"{}"'.format(delete_path))
            return

        os.remove(delete_path)
        _logger.info('\nModel is deleted:\n{}'.format(delete_path))

    def print_layer_shapes(self):
        laconic_logger.info('Net shapes:')

        layers = get_all_layers(self._net['dist'])
        for l in layers:
            laconic_logger.info('\t%-20s \t%s' % (l.name, get_output_shape(l)))

    def print_matrices_weights(self):
        laconic_logger.info('\nNet matrices weights:')
        params = get_all_params(self._net['dist'])
        values = get_all_param_values(self._net['dist'])

        total_network_size = 0
        for p, v in zip(params, values):
            param_size = float(v.nbytes) / 1024 / 1024
            # Work around numpy/python 3 regression: 
            # http://www.markhneedham.com/blog/2017/11/19/python-3-typeerror-unsupported-format-string-passed-to-numpy-ndarray-__format__/
            laconic_logger.info('\t{0:<40} dtype: {1:<10} shape: {2:<12} size: {3:<.2f}M'.format(
                p.name, repr(v.dtype), repr(v.shape), param_size))
            total_network_size += param_size
        laconic_logger.info('Total network size: {0:.1f} Mb'.format(total_network_size))


def get_nn_model(index_to_token, index_to_condition, model_init_path=None, w2v_matrix=None, resolver_factory=None,
                 is_reverse_model=False):
    model = CakeChatModel(index_to_token,
                          index_to_condition,
                          model_init_path=model_init_path,
                          init_embedding=w2v_matrix,
                          is_reverse_model=is_reverse_model)

    model.print_layer_shapes()

    # try to initialise model with pre-trained weights
    resolver = resolver_factory(model.model_load_path) if resolver_factory else DummyFileResolver(model.model_load_path)
    model_exists = resolver.resolve()

    if model_exists:
        _logger.info('\nLoading weights from file:\n{}\n'.format(model.model_load_path))
        model.load_weights()
    elif model_init_path:
        raise FileNotFoundException('Can\'t initialize model from file:\n{}\n'.format(model_init_path))
    else:
        _logger.info('\nModel will be built with initial weights.\n')

    model.print_matrices_weights()
    _logger.info('\nModel is built\n')

    return model, model_exists
