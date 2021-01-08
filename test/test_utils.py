import eye_datasets


def test_dataset_split():
    length = 1000000
    data = list(range(length))
    ratio = 0.7

    train_data, eval_data = eye_datasets.utils.dataset_split(data, ratio)

    assert len(train_data) == int(length * ratio)
    assert len(eval_data) + len(train_data) == length
