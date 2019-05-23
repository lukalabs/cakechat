from cakechat.dialog_model.keras_model import KerasTFModelIsolator
from cakechat.dialog_model.model import CakeChatModel


class InferenceCakeChatModel(CakeChatModel, KerasTFModelIsolator):
    """
    Inference-aimed extension of CakeChatModel, which supports isolation of underlying Keras (Tensorflow) model
    to fit multi-model multi-threaded run-time environments
    """

    def __init__(self,
                 index_to_token,
                 index_to_condition,
                 training_data_param=None,
                 validation_data_param=None,
                 w2v_model_param=None,
                 model_init_path=None,
                 model_resolver=None,
                 is_reverse_model=False,
                 reverse_model=None):
        KerasTFModelIsolator.__init__(self)

        self.init_model = self._isolate_func(self.init_model)
        self.resolve_model = self._isolate_func(self.resolve_model)
        self.print_weights_summary = self._isolate_func(self.print_weights_summary)
        self.train_model = self._isolate_func(self.train_model)

        self.get_utterance_encoding = self._isolate_func(self.get_utterance_encoding)
        self.get_thought_vectors = self._isolate_func(self.get_thought_vectors)
        self.predict_prob = self._isolate_func(self.predict_prob)
        self.predict_prob_by_thought_vector = self._isolate_func(self.predict_prob_by_thought_vector)
        self.predict_prob_one_step = self._isolate_func(self.predict_prob_one_step)
        self.predict_log_prob = self._isolate_func(self.predict_log_prob)
        self.predict_log_prob_one_step = self._isolate_func(self.predict_log_prob_one_step)
        self.predict_sequence_score = self._isolate_func(self.predict_sequence_score)
        self.predict_sequence_score_by_thought_vector = self._isolate_func(
            self.predict_sequence_score_by_thought_vector)

        super(InferenceCakeChatModel, self).__init__(
            index_to_token=index_to_token,
            index_to_condition=index_to_condition,
            training_data_param=training_data_param,
            validation_data_param=validation_data_param,
            w2v_model_param=w2v_model_param,
            model_init_path=model_init_path,
            model_resolver=model_resolver,
            is_reverse_model=is_reverse_model,
            reverse_model=reverse_model)
