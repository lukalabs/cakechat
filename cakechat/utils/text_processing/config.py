from cakechat.utils.data_structures import create_namedtuple_instance

SPECIAL_TOKENS = create_namedtuple_instance(
    'SPECIAL_TOKENS', PAD_TOKEN='_pad_', UNKNOWN_TOKEN='_unk_', START_TOKEN='_start_', EOS_TOKEN='_end_')

DIALOG_TEXT_FIELD = 'text'
DIALOG_CONDITION_FIELD = 'condition'
