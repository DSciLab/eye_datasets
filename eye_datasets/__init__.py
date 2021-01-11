from torch.nn.modules import transformer
from .utils import read_meta, dataset_split
from .eye_pacs import Dataset as PACSDataset
from .odir import Dataset as ODIRDataset
from .utils import get_transform


def get_data(opt, train_trans=None, eval_trans=None):
    dataset = opt.dataset.upper()
    if train_trans is None or eval_trans is None:
        train_trans, eval_trans = get_transform(opt)

    data = read_meta(opt.meta_path)
    train_data, eval_data = dataset_split(data,
                                          opt.train_ratio)

    if dataset == 'ODIR':
        return ODIRDataset(train_data,
                           train=True,
                           transform=train_trans), \
               ODIRDataset(eval_data,
                           train=False,
                           transform=eval_trans)
    elif dataset == 'PACS':
        return PACSDataset(train_data,
                           train=True,
                           transform=train_trans), \
               PACSDataset(eval_data,
                           train=False,
                           transform=eval_trans)
    else:
        raise RuntimeError(
            f'Unrecognized dataset ({dataset}).')
