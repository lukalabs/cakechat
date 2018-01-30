from cakechat.utils.data_structures import create_namedtuple_instance

SPECIAL_TOKENS = create_namedtuple_instance(
    'SPECIAL_TOKENS', PAD_TOKEN=u'_pad_', UNKNOWN_TOKEN=u'_unk_', START_TOKEN=u'_start_', EOS_TOKEN=u'_end_')

DIALOG_TEXT_FIELD = 'text'
DIALOG_CONDITION_FIELD = 'condition'
