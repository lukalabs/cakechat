import math
import os
from functools import partial

import numpy as np
import tensorflow as tf
from keras import Input, Model, optimizers
from keras.layers import K, Bidirectional, Embedding, Concatenate, Dense, Dropout, TimeDistributed, \
    Reshape, Lambda, CuDNNGRU, GRU

from cakechat.config import HIDDEN_LAYER_DIMENSION, GRAD_CLIP, LEARNING_RATE, TRAIN_WORD_EMBEDDINGS_LAYER, \
    WORD_EMBEDDING_DIMENSION, DENSE_DROPOUT_RATIO, CONDITION_EMBEDDING_DIMENSION, MODEL_NAME, BASE_CORPUS_NAME, \
    INPUT_CONTEXT_SIZE, INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH, BATCH_SIZE, LOG_RUN_METADATA, \
    TENSORBOARD_LOG_DIR, EPOCHS_NUM, SHUFFLE_TRAINING_BATCHES, RANDOM_SEED, RESULTS_PATH, USE_CUDNN
from cakechat.dialog_model.callbacks import CakeChatEvaluatorCallback
from cakechat.dialog_model.keras_model import AbstractKerasModel
from cakechat.dialog_model.layers import repeat_vector, softmax_with_temperature
from cakechat.dialog_model.model_utils import get_training_batch
from cakechat.dialog_model.quality.metrics.perplexity import calculate_model_mean_perplexity
from cakechat.dialog_model.quality.metrics.plotters import TensorboardMetricsPlotter
from cakechat.utils.data_structures import create_namedtuple_instance
from cakechat.utils.logger import WithLogger
from cakechat.utils.text_processing import SPECIAL_TOKENS
from cakechat.utils.w2v.utils import get_token_vector


class CakeChatModel(AbstractKerasModel, WithLogger):
    def __init__(self,
                 index_to_token,
                 index_to_condition,
                 training_data_param,
                 validation_data_param,
                 w2v_model_param,
                 model_init_path=None,
                 model_resolver=None,
                 model_name=MODEL_NAME,
                 corpus_name=BASE_CORPUS_NAME,
                 skip_token=SPECIAL_TOKENS.PAD_TOKEN,
                 token_embedding_dim=WORD_EMBEDDING_DIMENSION,
                 train_token_embedding=TRAIN_WORD_EMBEDDINGS_LAYER,
                 condition_embedding_dim=CONDITION_EMBEDDING_DIMENSION,
                 input_seq_len=INPUT_SEQUENCE_LENGTH,
                 input_context_size=INPUT_CONTEXT_SIZE,
                 output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                 hidden_layer_dim=HIDDEN_LAYER_DIMENSION,
                 use_cudnn=USE_CUDNN,
                 dense_dropout_ratio=DENSE_DROPOUT_RATIO,
                 is_reverse_model=False,
                 reverse_model=None,
                 learning_rate=LEARNING_RATE,
                 grad_clip=GRAD_CLIP,
                 batch_size=BATCH_SIZE,
                 epochs_num=EPOCHS_NUM,
                 horovod=None,
                 tensorboard_log_dir=TENSORBOARD_LOG_DIR,
                 log_run_metadata=LOG_RUN_METADATA):
        """
        :param index_to_token: Dict with mapping: tokens indices to tokens
        :param index_to_condition: Dict with mapping: conditions indicies to conditions values
        :param training_data_param: Instance of ModelParam, tuple (value, id) where value is a dataset used for training
        and id is a name this dataset
        :param validation_data_param: Instance of ModelParam, tuple (value, id) where value is a dataset used for
        metrics calculation and id is a concatenation of these datasets' names
        :param w2v_model_param: Instance of ModelParam, tuple (value, id) where value is a word2vec matrix of shape
        (vocab_size, token_embedding_dim) with float values, used for initializing token embedding layers, and id is
        the name of word2vec model
        :param model_init_path: Path to a file with model's saved weights for layers intialization
        :param model_resolver: Factory that takes model path and returns a file resolver object
        :param model_name: String prefix that is prepended to automatically generated model's name. The prefix helps
         distinguish the current experiment from other experiments with similar params.
        :param corpus_name: File name of the training dataset (included into automatically generated model's name)
        :param skip_token: Token to skip with masking, usually _pad_ token. Id of this token is inferred from
        index_to_token dictionary
        :param token_embedding_dim:  Vectors dimensionality of tokens embeddings
        :param train_token_embedding: Bool value indicating whether to train token embeddings along with other model's
        weights or keep them freezed during training
        :param condition_embedding_dim: Vectors dimensionality of conditions embeddings
        :param input_seq_len: Max number of tokens in the context sentences
        :param input_context_size: Max number of sentences in the context
        :param output_seq_len: Max number of tokens in the output sentences
        :param hidden_layer_dim: Vectors dimensionality of hidden layers in GRU and Dense layers
        :param dense_dropout_ratio: Float value between 0 and 1, indicating the ratio of neurons that will be randomly
        deactivated during training to prevent model's overfitting
        :param is_reverse_model: Bool value indicating the type of model:
        False (regular model) - predicts response for the given context
        True (reverse model) - predicts context for the given response (actually, predict the last context sentence for
        given response and the beginning of the context) - used for calculating Maximim Mutual Information metric
        :param reverse_model: Trained reverse model used to generate predictions in *_reranking modes
        :param learning_rate: Learning rate of the optimization algorithm
        :param grad_clip: Clipping parameter of the optimization algorithm, used to prevent gradient explosion
        :param batch_size: Number of samples to be used for gradient estimation on each train step
        :param epochs_num: Number of full dataset passes during train
        :param horovod: Initialized horovod module used for multi-gpu training. Trains on single gpu if horovod=None
        :param tensorboard_log_dir: Path to tensorboard logs directory
        :param log_run_metadata: Set 'True' to profile memory consumption and computation time on tensorboard
        """
        # Calculate batches number in each epoch.
        # The last batch which may be smaller than batch size is included in this number
        batches_num_per_epoch = math.ceil(training_data_param.value.x.shape[0] / batch_size) \
            if training_data_param.value else None

        # Create callbacks
        callbacks = self._create_essential_callbacks(self, horovod)
        callbacks.extend([
            # Custom callback for metrics calculation
            CakeChatEvaluatorCallback(self, index_to_token, batch_size, batches_num_per_epoch)
        ])

        super(CakeChatModel, self).__init__(
            model_resolver_factory=model_resolver,
            metrics_plotter=TensorboardMetricsPlotter(tensorboard_log_dir),
            horovod=horovod,
            training_callbacks=callbacks)
        WithLogger.__init__(self)

        self._model_name = 'reverse_{}'.format(model_name) if is_reverse_model else model_name
        self._rnn_class = CuDNNGRU if use_cudnn else partial(GRU, reset_after=True)

        # tokens params
        self._index_to_token = index_to_token
        self._token_to_index = {v: k for k, v in index_to_token.items()}
        self._vocab_size = len(self._index_to_token)
        self._skip_token_id = self._token_to_index[skip_token]

        self._token_embedding_dim = token_embedding_dim
        self._train_token_embedding = train_token_embedding
        self._W_init_embedding = \
            self._build_embedding_matrix(self._token_to_index, w2v_model_param.value, token_embedding_dim) \
                if w2v_model_param.value else None

        # condition params
        self._index_to_condition = index_to_condition
        self._condition_to_index = {v: k for k, v in index_to_condition.items()}
        self._condition_embedding_dim = condition_embedding_dim

        # data params
        self._training_data = training_data_param.value
        self._validation_data = validation_data_param.value

        # train params
        self._batches_num_per_epoch = batches_num_per_epoch
        self._model_init_path = model_init_path
        self._horovod = horovod

        self._optimizer = optimizers.Adadelta(lr=learning_rate, clipvalue=grad_clip)
        if self._horovod:
            self._optimizer = horovod.DistributedOptimizer(self._optimizer)

        # gather model's params that define the experiment setting
        self._params = create_namedtuple_instance(
            name='Params',
            corpus_name=corpus_name,
            input_context_size=input_context_size,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            token_embedding_dim=token_embedding_dim,
            train_batch_size=batch_size,
            hidden_layer_dim=hidden_layer_dim,
            w2v_model=w2v_model_param.id,
            is_reverse_model=is_reverse_model,
            dense_dropout_ratio=dense_dropout_ratio,
            voc_size=len(self._token_to_index),
            training_data=training_data_param.id,
            validation_data=validation_data_param.id,
            epochs_num=epochs_num,
            optimizer=self._optimizer.get_config())

        # profiling params
        self._run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if log_run_metadata else None
        self._run_metadata = tf.RunMetadata() if log_run_metadata else None

        # parts of computational graph
        self._models = None

        # get trained reverse model used for inference
        self._reverse_model = reverse_model

    @property
    def model_name(self):
        return self._model_name

    @property
    def run_metadata(self):
        return self._run_metadata

    @property
    def token_to_index(self):
        return self._token_to_index

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
    def vocab_size(self):
        return self._vocab_size

    @property
    def skip_token_id(self):
        return self._skip_token_id

    @property
    def hidden_layer_dim(self):
        return self._params.hidden_layer_dim

    @property
    def decoder_depth(self):
        return self._decoder_depth

    @property
    def is_reverse_model(self):
        return self._params.is_reverse_model

    @property
    def reverse_model(self):
        return self._reverse_model

    @property
    def _model_dir(self):
        return os.path.join(RESULTS_PATH, 'nn_models')

    @property
    def _model_params(self):
        return self._params._asdict()

    @property
    def _model_progress_resource_path(self):
        return os.path.join(self.model_path, self._MODEL_PROGRESS_RESOURCE_NAME)

    def _build_model(self):
        # embeddings
        x_tokens_emb_model = self._tokens_embedding_model(name='x_token_embedding')
        y_tokens_emb_model = self._tokens_embedding_model(name='y_token_embedding')
        with tf.variable_scope('condition_embedding_scope', reuse=True):
            condition_emb_model = self._condition_embedding_model()

        # encoding
        with tf.variable_scope('utterance_scope', reuse=True):
            utterance_enc_model = self._utterance_encoder(x_tokens_emb_model)
        with tf.variable_scope('encoder_scope', reuse=True):
            context_enc_model = self._context_encoder(utterance_enc_model)

        # decoding
        with tf.variable_scope('decoder_scope', reuse=True):
            decoder_training_model, decoder_model = self._decoder(y_tokens_emb_model, condition_emb_model)

        # seq2seq
        seq2seq_training_model, seq2seq_model = self._seq2seq(context_enc_model, decoder_training_model, decoder_model)

        self._models = dict(
            utterance_encoder=utterance_enc_model,
            context_encoder=context_enc_model,
            decoder=decoder_model,
            seq2seq=seq2seq_model,
            seq2seq_training=seq2seq_training_model)

        return self._models['seq2seq_training']

    def _get_training_model(self):
        def sparse_categorical_crossentropy_logits(y_true, y_pred):
            return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

        self._logger.info('Compiling seq2seq for train...')

        self._models['seq2seq_training'].compile(
            loss=sparse_categorical_crossentropy_logits,
            optimizer=self._optimizer,
            options=self._run_options,
            run_metadata=self._run_metadata)

        return self._models['seq2seq_training']

    def _tokens_embedding_model(self, name='token_embedding'):
        self._logger.info('Building tokens_embedding_model...')

        tokens_ids = Input(shape=(None, ), dtype='int32', name=name + '_input')
        # output shape == (batch_size, seq_len)

        tokens_embeddings = Embedding(
            input_dim=self._vocab_size,
            output_dim=self._token_embedding_dim,
            trainable=self._train_token_embedding,
            name=name,
            weights=None if self._W_init_embedding is None else [self._W_init_embedding])(tokens_ids)
        # output shape == (batch_size, seq_len, token_emb_size)

        return Model(tokens_ids, tokens_embeddings, name=name + '_model')

    def _condition_embedding_model(self):
        self._logger.info('Building condition_embedding_model...')

        condition_id = Input(shape=(1, ), dtype='int32', name='condition_input')
        # output shape == (batch_size, 1)

        condition_emb = Embedding(
            input_dim=len(self._condition_to_index),
            output_dim=self._condition_embedding_dim,
            name='condition_embedding')(condition_id)
        # output shape == (batch_size, 1, condition_emb_size)

        condition_emb_reshaped = Reshape(
            target_shape=(self._condition_embedding_dim, ), name='condition_embedding_reshaped')(condition_emb)
        # output shape == (batch_size, condition_emb_size)

        return Model(condition_id, condition_emb_reshaped, name='condition_emb_model')

    def _utterance_encoder(self, tokens_emb_model):
        self._logger.info('Building utterance_encoder...')

        tokens_ids = tokens_emb_model.inputs[0]
        # output shape == (batch_size, seq_len)
        tokens_embeddings = tokens_emb_model(tokens_ids)
        # output shape == (batch_size, seq_len, token_emb_size)

        bidir_enc = Bidirectional(
            layer=self._rnn_class(
                units=self._params.hidden_layer_dim, return_sequences=True, name='encoder'),
            name='bidir_utterance_encoder')(tokens_embeddings)
        # output shape == (batch_size, seq_len, 2 * hidden_layer_dim)

        utterance_encoding = self._rnn_class(
            units=self._params.hidden_layer_dim, name='utterance_encoder_final')(bidir_enc)
        # output shape == (batch_size, hidden_layer_dim)

        return Model(tokens_ids, utterance_encoding, name='utterance_encoder_model')

    def _context_encoder(self, utterance_enc_model):
        self._logger.info('Building context_encoder...')
        context_tokens_ids = Input(
            shape=(self._params.input_context_size, self._params.input_seq_len),
            dtype='int32',
            name='context_tokens_ids')
        # output shape == (batch_size, context_size, seq_len)

        context_utterance_embeddings = TimeDistributed(
            layer=utterance_enc_model, input_shape=(self._params.input_context_size,
                                                    self._params.input_seq_len))(context_tokens_ids)
        # output shape == (batch_size, context_size, utterance_encoding_dim)

        context_encoding = self._rnn_class(
            units=self._params.hidden_layer_dim, name='context_encoder')(context_utterance_embeddings)
        # output shape == (batch_size, hidden_layer_dim)

        return Model(context_tokens_ids, context_encoding, name='encoder_model')

    def _decoder(self, tokens_emb_model, condition_emb_model):
        self._logger.info('Building decoder...')

        thought_vector = Input(shape=(self._params.hidden_layer_dim, ), dtype=K.floatx(), name='dec_thought_vector')
        # output shape == (batch_size, hidden_layer_dim)
        response_tokens_ids = tokens_emb_model.inputs[0]
        # output shape == (batch_size, seq_len)
        condition_id = condition_emb_model.inputs[0]
        # output shape == (batch_size, 1)
        temperature = Input(shape=(1, ), dtype='float32', name='dec_temperature')
        # output shape == (batch_size, 1)

        # hardcode decoder's depth here: the general solution for any number of stacked rnn layers hs num is too bulky
        # and we don't need it, so keep it simple, stupid
        self._decoder_depth = 2
        # keep inputs for rnn decoder hidden states globally accessible for all model layers that are using them
        # otherwise you may encounter a keras bug that affects rnn stateful models
        # related discussion: https://github.com/keras-team/keras/issues/9385#issuecomment-365464721
        self._dec_hs_input = Input(
            shape=(self._decoder_depth, self._params.hidden_layer_dim), dtype=K.floatx(), name='dec_hs')
        # shape == (batch_size, dec_depth, hidden_layer_dim)

        response_tokens_embeddings = tokens_emb_model(response_tokens_ids)
        # output shape == (batch_size, seq_len, token_emb_size)
        condition_embedding = condition_emb_model(condition_id)
        # output shape == (batch_size, cond_emb_size)
        conditioned_tv = Concatenate(name='conditioned_tv')([thought_vector, condition_embedding])
        # output shape == (batch_size, hidden_layer_dim + cond_emb_size)

        # Temporary solution:
        # use a custom lambda function for layer repeating and manually set output_shape
        # otherwise the consequent Concatenate layer won't work
        repeated_conditioned_tv = Lambda(
            function=repeat_vector,
            mask=lambda inputs, inputs_masks: inputs_masks[0],  # function to get mask of the first input
            output_shape=(None, self._params.hidden_layer_dim + self._condition_embedding_dim),
            name='repeated_conditioned_tv')([conditioned_tv, response_tokens_ids])
        # output shape == (batch_size, seq_len, hidden_layer_dim + cond_emb_size)

        decoder_input = Concatenate(name='concat_emb_cond_tv')([response_tokens_embeddings, repeated_conditioned_tv])
        # output shape == (batch_size, seq_len, token_emb_size + hidden_layer_dim + cond_emb_size)

        # unpack hidden states to tensors
        dec_hs_0 = Lambda(
            function=lambda x: x[:, 0, :], output_shape=(self._params.hidden_layer_dim, ),
            name='dec_hs_0')(self._dec_hs_input)

        dec_hs_1 = Lambda(
            function=lambda x: x[:, 1, :], output_shape=(self._params.hidden_layer_dim, ),
            name='dec_hs_1')(self._dec_hs_input)

        outputs_seq_0, updated_hs_seq_0 = self._rnn_class(
            units=self._params.hidden_layer_dim, return_sequences=True, return_state=True, name='decoder_0')\
            (decoder_input, initial_state=dec_hs_0)
        # outputs_seq_0 and updated_hs_seq_0 shapes == (batch_size, seq_len, hidden_layer_dim)

        outputs_seq_1, updated_hs_seq_1 = self._rnn_class(
            units=self._params.hidden_layer_dim, return_sequences=True, return_state=True, name='decoder_1')\
            (outputs_seq_0, initial_state=dec_hs_1)
        # outputs_seq_1 and updated_hs_seq_1 shapes == (batch_size, seq_len, hidden_layer_dim)

        outputs_dropout = Dropout(rate=self._params.dense_dropout_ratio)(outputs_seq_1)
        # output shape == (batch_size, seq_len, hidden_layer_dim)
        tokens_logits = Dense(self._vocab_size)(outputs_dropout)
        # output shape == (batch_size, seq_len, vocab_size)
        tokens_probs = softmax_with_temperature(tokens_logits, temperature)
        # output shape == (batch_size, seq_len, vocab_size)

        # pack updated hidden states into one tensor
        updated_hs = Concatenate(
            axis=1, name='updated_hs')([
                Reshape((1, self._params.hidden_layer_dim))(updated_hs_seq_0),
                Reshape((1, self._params.hidden_layer_dim))(updated_hs_seq_1)
            ])

        decoder_training_model = Model(
            inputs=[thought_vector, response_tokens_ids, condition_id, self._dec_hs_input],
            outputs=[tokens_logits],
            name='decoder_training_model')

        decoder_model = Model(
            inputs=[thought_vector, response_tokens_ids, condition_id, self._dec_hs_input, temperature],
            outputs=[tokens_probs, updated_hs],
            name='decoder_model')

        return decoder_training_model, decoder_model

    def _seq2seq(self, context_encoder, decoder_training, decoder):
        self._logger.info('Building seq2seq...')

        context_tokens_ids = context_encoder.inputs[0]
        # output shape == (batch_size, context_size, input_seq_len)
        response_tokens_ids = decoder.inputs[1]
        # output shape == (batch_size, output_seq_len - 1)
        condition_id = decoder.inputs[2]
        # output shape == (batch_size, 1)
        temperature = decoder.inputs[4]
        # output shape == (batch_size, 1)

        context_encoding = context_encoder(inputs=[context_tokens_ids])
        # output shape == (batch_size, hidden_layer_dim)

        tokens_logits = decoder_training(
            inputs=[context_encoding, response_tokens_ids, condition_id, self._dec_hs_input])
        # tokens_probs shape == (batch_size, output_seq_len - 1, vocab_size)

        tokens_probs, _ = decoder(
            inputs=[context_encoding, response_tokens_ids, condition_id, self._dec_hs_input, temperature])
        # tokens_probs shape == (batch_size, output_seq_len - 1, vocab_size)

        training_model = Model(
            inputs=[context_tokens_ids, response_tokens_ids, condition_id, self._dec_hs_input],
            outputs=[tokens_logits],
            name='seq2seq_training_model')

        model = Model(
            inputs=[context_tokens_ids, response_tokens_ids, condition_id, self._dec_hs_input, temperature],
            outputs=[tokens_probs],
            name='seq2seq_model')

        return training_model, model

    def _get_training_batch_generator(self):
        # set unique random seed for different workers to correctly process batches in multi-gpu training
        horovod_seed = self._horovod.rank() if self._horovod else 0
        epoch_id = 0

        while True:  # inifinite batches generator
            epoch_id += 1

            for train_batch in get_training_batch(
                    self._training_data,
                    self._params.train_batch_size,
                    random_permute=SHUFFLE_TRAINING_BATCHES,
                    random_seed=RANDOM_SEED * epoch_id + horovod_seed):

                context_tokens_ids, response_tokens_ids, condition_id = train_batch
                # response tokens are wraped with _start_ and _end_ tokens
                # output shape == (batch_size, seq_len)

                # get input response ids by removing last sequence token (_end_)
                input_response_tokens_ids = response_tokens_ids[:, :-1]
                # output shape == (batch_size, seq_len - 1)

                # get target response ids by removing the first (_start_) token of the sequence
                target_response_tokens_ids = response_tokens_ids[:, 1:]
                # output shape == (batch_size, seq_len - 1)

                # workaround for using sparse_categorical_crossentropy loss
                # see https://github.com/tensorflow/tensorflow/issues/17150#issuecomment-399776510
                target_response_tokens_ids = np.expand_dims(target_response_tokens_ids, axis=-1)
                # output shape == (batch_size, seq_len - 1, 1)

                init_dec_hs = np.zeros(
                    shape=(context_tokens_ids.shape[0], self._decoder_depth, self._params.hidden_layer_dim),
                    dtype=K.floatx())

                yield [context_tokens_ids, input_response_tokens_ids, condition_id,
                       init_dec_hs], target_response_tokens_ids

    def _get_epoch_batches_num(self):
        return self._batches_num_per_epoch

    def get_utterance_encoding(self, utterance_tokens_ids):
        """
        :param utterance_tokens_ids:   shape == (batch_size, seq_len), int32
        :return: utterance_encoding    shape == (batch_size, hidden_layer_dim), float32
        """
        return self._models['utterance_encoder'](utterance_tokens_ids)

    def get_thought_vectors(self, context_tokens_ids):
        """
        :param context_tokens_ids,   shape == (batch_size, context_size, seq_len), int32
        :return: context_encoding,   shape == (batch_size, hidden_layer_dim), float32
        """
        return self._models['context_encoder'].predict(context_tokens_ids)

    def predict_prob(self, context_tokens_ids, response_tokens_ids, condition_id, temperature=1.0):
        """
        :param context_tokens_ids:      shape == (batch_size, context_size, seq_len), int32
        :param response_tokens_ids:     shape == (batch_size, seq_len), int32
        :param condition_id:            shape == (batch_size, 1), int32
        :param temperature:             float32
        :return:
            tokens_probs:               shape == (batch_size, seq_len, vocab_size), float32
        """
        # remove last token, but keep first token to match seq2seq decoder input's shape
        response_tokens_ids = response_tokens_ids[:, :-1]
        # shape == (batch_size, seq_len - 1)

        init_dec_hs = np.zeros(
            shape=(context_tokens_ids.shape[0], self._decoder_depth, self._params.hidden_layer_dim), dtype=K.floatx())
        # shape == (batch_size, decoder_depth, hidden_layer_dim)

        temperature = np.full_like(condition_id, temperature, dtype=np.float32)
        # shape == (batch_size, 1)

        tokens_probs = self._models['seq2seq'].predict(
            [context_tokens_ids, response_tokens_ids, condition_id, init_dec_hs, temperature])
        # shape == (batch_size, seq_len - 1, vocab_size)
        return tokens_probs

    def predict_prob_by_thought_vector(self, thought_vector, response_tokens_ids, condition_id, temperature=1.0):
        """
        :param thought_vector:          shape == (batch_size, hidden_layer_dim), float32
        :param response_tokens_ids:     shape == (batch_size, seq_len), int32
        :param condition_id:            shape == (batch_size, 1), int32
        :param temperature:             float32
        :return:
            tokens_probs:               shape == (batch_size, seq_len, vocab_size), float32
        """
        # remove last token, but keep first token to match seq2seq decoder input's shape
        response_tokens_ids = response_tokens_ids[:, :-1]
        # output shape == (batch_size, seq_len - 1)

        init_dec_hs = \
            np.zeros((thought_vector.shape[0], self._decoder_depth, self._params.hidden_layer_dim), dtype=K.floatx())
        # shape == (batch_size, decoder_depth, hidden_layer_dim)

        temperature = np.full_like(condition_id, temperature, dtype=np.float32)
        # shape == (batch_size, 1)

        tokens_probs, _ = self._models['decoder'].predict(
            [thought_vector, response_tokens_ids, condition_id, init_dec_hs, temperature])
        # shape == (batch_size, seq_len - 1, vocab_size)
        return tokens_probs

    def predict_prob_one_step(self, thought_vector, prev_hidden_states, prev_tokens_id, condition_id, temperature=1.0):
        """
        :param thought_vector:          shape == (batch_size, hidden_layer_dim), float32
        :param prev_hidden_states:      shape == (batch_size, 2, hidden_layer_dim), float32
        :param prev_tokens_id:          shape == (batch_size, 1), int32
        :param condition_id:            shape == (batch_size, 1), int32
        :param temperature:             float32
        :return:
            updated_hidden_states:      shape == (batch_size, 2, hidden_layer_dim), float32
            transformed_token_prob:     shape == (batch_size, vocab_size), float32
        """
        temperature = np.full_like(prev_tokens_id, temperature, dtype=np.float32)
        # shape == (batch_size, 1)

        token_prob, updated_hidden_states = self._models['decoder'].predict(
            [thought_vector, prev_tokens_id, condition_id, prev_hidden_states, temperature])
        return updated_hidden_states, token_prob

    def predict_log_prob(self, context_tokens_ids, response_tokens_ids, condition_id, temperature=1.0):
        """
        :param context_tokens_ids:      shape == (batch_size, context_size, seq_len), int32
        :param response_tokens_ids:     shape == (batch_size, seq_len), int32
        :param condition_id:            shape == (batch_size, 1), int32
        :param temperature:             float32
        :return:
            tokens_probs:               shape == (batch_size, seq_len, vocab_size), float32
        """
        tokens_probs = self.predict_prob(context_tokens_ids, response_tokens_ids, condition_id, temperature)

        tokens_log_probs = np.log(tokens_probs)
        return tokens_log_probs

    def predict_log_prob_one_step(self,
                                  thought_vector,
                                  prev_hidden_states,
                                  prev_tokens_id,
                                  condition_id,
                                  temperature=1.0):
        """
        :param thought_vector:          shape == (batch_size, hidden_layer_dim), float32
        :param prev_hidden_states:      shape == (batch_size, 2 * hidden_layer_dim), float32
        :param prev_tokens_id:          shape == (batch_size, 1), int32
        :param condition_id:            shape == (batch_size, 1), int32
        :param temperature:             float32
        :return:
            updated_hidden_states:      shape == (batch_size, 2 * hidden_layer_dim), float32
            transformed_token_prob:     shape == (batch_size, vocab_size), float32
        """
        updated_hidden_states, token_prob = self.predict_prob_one_step(thought_vector, prev_hidden_states,
                                                                       prev_tokens_id, condition_id, temperature)

        token_log_prob = np.log(token_prob)
        return updated_hidden_states, token_log_prob

    def _compute_sequence_score(self, tokens_ids, tokens_probs):
        """
        :param tokens_ids:      shape == (batch_size, seq_len), int32
        :param tokens_probs:    shape == (batch_size, seq_len - 1, vocab_size), float32
        :return:
        """
        mask = tokens_ids != self._skip_token_id
        # shape == (batch_size, seq_len)

        # All shapes are symbolic and are evaluated on run-time only after input tensors are supplied
        batch_size, truncated_seq_len, vocab_size = tokens_probs.shape
        total_tokens_num = batch_size * truncated_seq_len

        # We need to reshape here for effective slicing without any loops or scans
        probs_long = tokens_probs.reshape((total_tokens_num, vocab_size))
        # shape == (batch_size * (seq_len - 1), vocab_size), float32

        # Do not use first tokens for likelihood computation:
        # these are _start_ tokens, we don't have probabilities for them
        output_ids = tokens_ids[:, 1:]
        # shape == (batch_size, seq_len - 1)
        mask = mask[:, 1:]
        # shape == (batch_size, seq_len - 1)

        token_ids_flattened = output_ids.reshape((total_tokens_num, ))
        # shape == (batch_size * (seq_len - 1))

        # Select probabilities of only observed tokens and reshape back
        observed_tokens_probs = probs_long[np.arange(total_tokens_num), token_ids_flattened]
        # shape == (batch_size * (seq_len - 1), )
        observed_tokens_log_probs = np.log(observed_tokens_probs)
        # shape == (batch_size * (seq_len - 1), )
        nonpad_observed_tokens_log_probs = observed_tokens_log_probs.reshape((batch_size, truncated_seq_len)) * mask
        # shape == (batch_size, seq_len - 1)

        batch_scores = nonpad_observed_tokens_log_probs.sum(axis=1)
        # shape == (batch_size, )
        return batch_scores

    def predict_sequence_score(self, context_tokens_ids, response_tokens_ids, condition_id):
        """
        :param context_tokens_ids:      shape == (batch_size, context_size, seq_len), int32
        :param response_tokens_ids:     shape == (batch_size, seq_len), int32
        :param condition_id:            shape == (batch_size, 1), int32
        :return:
            sequence_score:             shape == (batch_size, 1), float32
        """
        response_tokens_probs = self.predict_prob(context_tokens_ids, response_tokens_ids, condition_id)
        # output shape == (batch_size, seq_len - 1, vocab_size)

        return self._compute_sequence_score(response_tokens_ids, response_tokens_probs)

    def predict_sequence_score_by_thought_vector(self, thought_vector, response_tokens_ids, condition_id):
        """
        :param thought_vector:          shape == (batch_size, hidden_layer_dim), float32
        :param response_tokens_ids:     shape == (batch_size, seq_len), int32
        :param condition_id:            shape == (batch_size, 1), int32
        :return:
            sequence_score:             shape == (batch_size, 1), float32
        """
        response_tokens_probs = self.predict_prob_by_thought_vector(thought_vector, response_tokens_ids, condition_id)
        # output shape == (batch_size, seq_len - 1, vocab_size)

        return self._compute_sequence_score(response_tokens_ids, response_tokens_probs)

    def _evaluate(self):
        self._logger.info('Evaluating model\'s perplexity...')
        metrics = {}

        for dataset_name, dataset in self._validation_data.items():
            perplexity = calculate_model_mean_perplexity(self, dataset)
            metrics[dataset_name] = {'perplexity': perplexity}

        return metrics

    @staticmethod
    def _build_embedding_matrix(token_to_index, w2v_model, embedding_dim):
        embedding_matrix = np.zeros((len(token_to_index), embedding_dim))
        for token, index in token_to_index.items():
            embedding_matrix[index] = get_token_vector(token, w2v_model, embedding_dim)

        return embedding_matrix

    @staticmethod
    def _get_metric_mean(metrics, metric_name):
        return np.mean([metrics[dataset_name][metric_name] for dataset_name in metrics])

    def _is_better_model(self, new_metrics, old_metrics):
        return self._get_metric_mean(new_metrics, metric_name='perplexity') < \
               self._get_metric_mean(old_metrics, metric_name='perplexity')

    def _load_model_if_exists(self):
        if self._model_init_path:
            self._model = self._load_model(self._model, self._model_init_path)
            return

        # proceed with usual process of weights loading if no _model_init_path is passed
        super(CakeChatModel, self)._load_model_if_exists()
