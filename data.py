import h5py
import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


# complex data for evaluation
def load_test(mat_name, num_real):
    real_test = h5py.File(mat_name, 'r')
    key = list(real_test.keys())[0]
    real_test = np.transpose(real_test[key][:])
    real_test = real_test[::int(real_test.shape[0] / num_real), :, :, :]
    complex_data = real_test['real'] + real_test['imag'] * 1j
    return complex_data


# dataset for converting mat into npy array
class MyDataset(Dataset):
    def __init__(self, file_path, aug_ratio=0.5):
        # probability of performing online augmentation
        self.aug_ratio = aug_ratio
        self.mat = self.get_data(file_path)

    # load data & data augmentation
    def get_data(self, file_path):
        data = h5py.File(file_path, 'r')
        key = list(data.keys())[0]
        seq = data.get(key)
        seq = np.array(seq)

        real = np.transpose(seq['real'], [3, 2, 1, 0])[..., np.newaxis]
        imag = np.transpose(seq['imag'], [3, 2, 1, 0])[..., np.newaxis]
        # similarity : a+bj, -a-bj, b-aj, -b+aj
        mat = np.concatenate([
            np.concatenate([real, imag], axis=-1),
            np.concatenate([-1 * real, -1 * imag], axis=-1),
            np.concatenate([imag, -1 * real], axis=-1),
            np.concatenate([-1 * imag, real], axis=-1),
        ], axis=0)
        return mat

    def __len__(self):
        return self.mat.shape[0]

    def __getitem__(self, index):
        ret = self.mat[index]
        # online data augmentation with scale and random noise
        if np.random.uniform(low=0, high=1, size=1) <= self.aug_ratio:
            multi_noise = np.random.uniform(low=0.8, high=1.2, size=(1, 1, 1, 1))
            add_noise = np.random.normal(loc=0, scale=1e-4, size=ret.shape)
            ret = ret * multi_noise + add_noise
        return ret


# data loader for training, no validation, train all data
def data_pipeline(data_path, batch_size, aug_ratio=.5):
    dataset = MyDataset(data_path, aug_ratio)
    data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=8)
    return data_loader


if __name__ == '__main__':
    batch_size = 10
    aug_ratio = 0.5
    data_path = '/opt/tiger/debug_server/VAE/data/H1_32T4R.mat'
    data_path = '/opt/tiger/debug_server/VAE/data/H2_32T4R.mat'
    data_loader = data_pipeline(data_path, batch_size, aug_ratio)
    for x in tqdm.tqdm(data_loader):
        print(x.shape, torch.max(x), torch.min(x))
