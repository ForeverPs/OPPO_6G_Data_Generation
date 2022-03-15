import tqdm
import torch
from model import ResVAE
from eval import online_eval
from torch.optim import Adam
from data import data_pipeline
from torch.optim.lr_scheduler import StepLR


torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(epochs, batch_size, data_path, lr, aug_ratio):
    data_loader = data_pipeline(data_path, batch_size, aug_ratio)

    model = ResVAE()
    try:
        model.load_state_dict(torch.load(pretrain_path, map_location='cpu'))
    except:
        print('Training from scratch...')
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=300, gamma=.5)

    best_score = score_thresh
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for x in tqdm.tqdm(data_loader):
            x = x.float().to(device)
            model.zero_grad()

            recon = model(x)
            loss = model.loss(x, recon)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() / len(data_loader)

        scheduler.step()
        
        model.eval()
        sim, multi, multi_div_sim, score = online_eval(model, type=data_type)

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), 'saved_models/%d/att_sim_%.3f_multi_%.3f_score_%.3f.pth' % (data_type, sim, multi, score))
        print('Epoch : %05d | loss : %.3f | Sim : %.3f | Multi : %.3f | Multi_div_Sim : %.3f | Score : %.3f' %
              (epoch, epoch_loss, sim, multi, multi_div_sim, score))


if __name__ == '__main__':
    # please only change data_type for training
    data_type = 1

    batch_size = 16
    thresh = {1: 0.8, 2: 0.76}
    data_paths = {1: 'data/H1_32T4R.mat', 2: 'data/H2_32T4R.mat'}
    keys = {2: 'saved_models/2/att_sim_0.212_multi_1.918_score_0.774.pth',
            1: 'saved_models/1/att_sim_0.320_multi_1.076_score_0.832.pth'}

    lr = 1e-3
    epochs = 10000
    aug_ratio = .3
    pretrain_path = keys[data_type]
    score_thresh = thresh[data_type]
    data_path = data_paths[data_type]
    train(epochs, batch_size, data_path, lr, aug_ratio)
