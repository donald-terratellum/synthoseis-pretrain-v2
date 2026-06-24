from synthoseis_pre_train.pretrain import _default_batch_size_for_model


def test_default_batch_size_for_small_model():
    assert _default_batch_size_for_model(13_999_999) == 2


def test_default_batch_size_for_large_model():
    assert _default_batch_size_for_model(14_000_000) == 1
