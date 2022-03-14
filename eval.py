import torch
import numpy as np
from data import load_test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# parameter settings
NUM_RX = 4
NUM_TX = 32
NUM_DELAY = 32
NUM_REAL = {1: 500, 2: 4000}
NUM_FAKE = NUM_REAL
REAL_DATA_NAME = {1: 'data/H1_32T4R.mat',
                  2: 'data/H2_32T4R.mat'}
SIM = {1: 0.2, 2: 0.1}
MULTI_DIV_SIM = {1: 20, 2: 40}


# online evaluation
def K_nearest(h_true_smp, h_fake_smp, rx_num, tx_num, delay_num, flag):
    h_true = np.reshape(h_true_smp, [h_true_smp.shape[0], rx_num * tx_num * delay_num])
    h_fake = np.reshape(h_fake_smp, [h_fake_smp.shape[0], rx_num * tx_num * delay_num])

    h_true_norm = np.linalg.norm(h_true, axis=1)
    h_fake_norm = np.linalg.norm(h_fake, axis=1)
    h_true_norm = h_true_norm[:, np.newaxis]
    h_fake_norm = h_fake_norm[:, np.newaxis]
    h_true_norm_matrix = np.tile(h_true_norm, (1, rx_num * tx_num * delay_num))
    h_fake_norm_matrix = np.tile(h_fake_norm, (1, rx_num * tx_num * delay_num))
    h_true = h_true / h_true_norm_matrix
    h_fake = h_fake / h_fake_norm_matrix

    r_s = abs(np.dot(h_fake, h_true.conj().T))
    r = r_s * r_s

    r_max = np.max(r, axis=1)
    r_idx = np.argmax(r, axis=1)
    K_sim_abs_mean = np.mean(r_max)

    counts_idx, counts_num = np.unique(r_idx, return_counts=True)
    K_multi = np.zeros((1, h_fake_smp.shape[0]))
    K_multi[:, counts_idx] = counts_num
    K_multi_std = float(np.sqrt(np.var(K_multi, axis=1) * h_fake_smp.shape[0] / (h_fake_smp.shape[0] - 1)))

    return K_sim_abs_mean, K_multi_std, K_multi_std / K_sim_abs_mean

# evaluation pipeline
def online_eval(g, type=1):
    real_test = load_test(REAL_DATA_NAME[type], NUM_REAL[type])

    if type == 1:
        fake_data = generator_1(NUM_FAKE[type], g, None)
    else:
        fake_data = generator_2(NUM_FAKE[type], g, None)

    # Data checking
    if (np.shape(fake_data) == np.array([NUM_FAKE[type], NUM_RX, NUM_TX, NUM_DELAY])).all():
        sim, multi, multi_div_sim = K_nearest(real_test, fake_data, NUM_RX, NUM_TX, NUM_DELAY, 1)
        score = (MULTI_DIV_SIM[type] - multi_div_sim) / MULTI_DIV_SIM[type]
    else:
        return 0, 0, 0, 0
    return sim, multi, multi_div_sim, score


def generator_1(num_fake_1, g, file_real_1=None):
    # g is the torch model on device
    generator_C = g.eval()
    size_packet = 100
    with torch.no_grad():
        for idx in range(int(num_fake_1 / size_packet)):
            fake_data = generator_C.sample(size_packet)
            fake_real_part, fake_imag_part = fake_data[..., 0], fake_data[..., 1]
            fake_real_part, fake_imag_part = fake_real_part.detach().cpu().numpy(), fake_imag_part.detach().cpu().numpy()
            fake_data_reshape = fake_real_part + fake_imag_part * 1j
            if idx == 0:
                data_fake_all = fake_data_reshape
            else:
                data_fake_all = np.concatenate((data_fake_all, fake_data_reshape), axis=0)
    return data_fake_all


def generator_2(num_fake_2, g, file_real_2=None):
    generator_U = g.eval()
    size_packet = 100
    with torch.no_grad():
        for idx in range(int(num_fake_2 / size_packet)):
            fake_data = generator_U.sample(size_packet)
            fake_real_part, fake_imag_part = fake_data[..., 0], fake_data[..., 1]
            fake_real_part, fake_imag_part = fake_real_part.detach().cpu().numpy(), fake_imag_part.detach().cpu().numpy()
            fake_data_reshape = fake_real_part + fake_imag_part * 1j
            if idx == 0:
                data_fake_all = fake_data_reshape
            else:
                data_fake_all = np.concatenate((data_fake_all, fake_data_reshape), axis=0)
    return data_fake_all

