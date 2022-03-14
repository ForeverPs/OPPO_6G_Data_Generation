import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def encoder_3d():
    class ConvNet(nn.Module):
        def __init__(self, input_channel, output_channel):
            super(ConvNet, self).__init__()
            self.conv = nn.Conv3d(input_channel, output_channel, kernel_size=3, stride=2, padding=1)
            self.bn = nn.BatchNorm3d(output_channel)
            self.relu = nn.LeakyReLU(0.1)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            return x

    class ConvTransposeNet(nn.Module):
        def __init__(self, input_channel, output_channel):
            super(ConvTransposeNet, self).__init__()
            self.conv = nn.ConvTranspose3d(input_channel, output_channel, kernel_size=3, stride=2, padding=1)
            self.bn = nn.BatchNorm3d(output_channel)
            self.relu = nn.LeakyReLU(0.1)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            return x

    class Encoder(nn.Module):
        def __init__(self, input_dim, latent_dim=128):
            super(Encoder, self).__init__()
            self.avg_pooling = torch.nn.AdaptiveAvgPool3d((32, 32, 32))
            self.conv_block = nn.Sequential(
                ConvNet(input_dim, 32),
                ConvNet(32, 64),
                ConvNet(64, 128),
                ConvNet(128, 256),
                ConvNet(256, 512),
            )

            self.log_var = nn.Linear(512, latent_dim)
            self.mean = nn.Linear(512, latent_dim)

        def forward(self, x):
            x = self.avg_pooling(x)
            x = self.conv_block(x).reshape(x.shape[0], 512)
            mean = self.mean(x)
            log_var = self.log_var(x)
            return mean, log_var

    class BaseLineDecoder(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(BaseLineDecoder, self).__init__()
            linear_block = nn.Sequential(
                nn.Linear(hidden_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(),
            )
            conv_block = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 2, stride=2),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 64, 2, stride=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(64, 32, 2, stride=2),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(32, 16, 2, stride=2),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(16, 32, 2, stride=2),
                nn.Tanh()
            )

            self.reshape = nn.AdaptiveAvgPool2d((16, 16))
            self.linear_block = linear_block
            self.conv_block = conv_block

        def forward(self, input):
            output = self.linear_block(input)
            output = output.view(-1, 256, 2, 2)
            output = self.conv_block(output) * 5.0  # [32, 16, 16]
            output = self.reshape(output)
            output = output.permute(0, 2, 3, 1).reshape(-1, 4, 2, 32, 32).permute(0, 1, 3, 4, 2)
            return output

    class Encoder3D_Decoder2D_VAE(nn.Module):
        def __init__(self, input_dim, latent_dim=128):
            super(Encoder3D_Decoder2D_VAE, self).__init__()
            self.latent_dim = latent_dim
            self.encoder = Encoder(input_dim, latent_dim)
            self.decoder = BaseLineDecoder(input_dim, latent_dim)

        def forward(self, x):
            tx = x.permute(0, 4, 1, 2, 3)
            mean, log_var = self.encoder(tx)
            noise = torch.randn_like(mean)
            feat = noise * torch.exp(log_var * 0.5) + mean
            recon = self.decoder(feat)
            return recon, mean, log_var

        def sample(self, size):
            noise = torch.randn((size, self.latent_dim)).to(device)
            recon = self.decoder(noise)
            return recon

        def loss(self, recon, x, mean, log_var, kld_weight):
            recons_loss = F.mse_loss(x, recon)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim=1), dim=0)
            loss = recons_loss + kld_weight * kld_loss
            return {'loss': loss, 'recon_loss': recons_loss.item(), 'kld_loss': -kld_loss.item()}
    return Encoder3D_Decoder2D_VAE(2, 128)


def encoder_2d():
    class VAE(nn.Module):
        def __init__(self, in_channels=8, latent_dim=128):
            super(VAE, self).__init__()
            self.latent_dim = latent_dim
            channels = [1, 2, 4, 8, 16]

            self.encoder = nn.ModuleList([nn.Sequential(
                nn.Conv2d(in_channels * channels[i], in_channels * channels[i + 1], 3, 2, 1),
                nn.BatchNorm2d(in_channels * channels[i + 1]),
                nn.LeakyReLU(),
            ) for i in range(len(channels) - 1)])
            self.encoder.append(nn.AdaptiveAvgPool2d(1))
            self.mu = nn.Linear(128, latent_dim)
            self.log_var = nn.Linear(128, latent_dim)

            channels.reverse()
            self.decode_in = nn.Linear(latent_dim, 4 * 4 * channels[0] * in_channels)

            self.decoder = nn.ModuleList([nn.Sequential(
                nn.ConvTranspose2d(channels[i] * in_channels, channels[i + 1] * in_channels, 3, 2, 1, 1),
                nn.BatchNorm2d(in_channels * channels[i + 1]),
                nn.LeakyReLU(),
            ) for i in range(len(channels) - 1)])

            self.decode_conv = nn.Conv2d(8, 8, 3, 2, 1)
            self.tanh = nn.Tanh()

        def encode(self, x):
            # x : batch, 4, 32, 32, 2 -> batch, 8, 32, 32
            x = x.view(x.shape[0], 8, 32, 32)
            for encoder in self.encoder:
                x = encoder(x)
            x = x.view(x.shape[0], -1)
            mu = self.mu(x)
            log_var = self.log_var(x)
            return mu, log_var

        def decode(self, z):
            # z : batch, latent_dim
            z = self.decode_in(z)
            z = z.view(z.shape[0], -1, 4, 4)
            for decoder in self.decoder:
                z = decoder(z)
            z = self.decode_conv(z)  # z : batch, 8, 32, 32
            z = z.view(z.shape[0], 4, 2, 32, 32).permute(0, 1, 3, 4, 2)
            z = 5.0 * self.tanh(z)  # z : batch, 4, 32, 32, 2
            return z

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu

        def forward(self, x):
            # x : batch, 4, 32, 32, 2
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            recon = self.decode(z)
            recon = recon.view(x.shape)
            return recon, mu, log_var

        def loss(self, recon, x, mu, log_var, kld_weight):
            recons_loss = F.mse_loss(recon, x) + 0.1 * nn.L1Loss()(recon, x)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            loss = recons_loss + kld_weight * kld_loss
            return {'loss': loss, 'recon_loss': recons_loss.item(), 'kld_loss': -kld_loss.item()}

        def sample(self, num_samples):
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decode(z)
            return samples
    return VAE()


class PAM_Module(nn.Module):
    """ Position attention module"""
    # Ref from SAGAN
    def __init__(self, in_dim=8):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim=8):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out
        return out


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                            stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                            planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, input_channel=32, hidden_dim=128):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.input_channel = input_channel

        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3,
                            stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.feat = nn.Sequential(nn.Linear(512*block.expansion, hidden_dim // 2),
                                  nn.BatchNorm1d(hidden_dim // 2, eps=1e-6))

        self.sigmoid = nn.Sigmoid()
        self.bit_layer = BitLayer(2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # modified by jiaxi to return feature
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.feat(out)
        out = self.sigmoid(out)
        out = self.bit_layer(out)
        return out


def ResNet18(input_channel=32, hidden_dim=128):
    return ResNet(BasicBlock, [2, 2, 2, 2], input_channel, hidden_dim)


def convert2bit(input_n, B):
    num_ = input_n.long().to(device)
    exp_bts = torch.arange(0, B)
    exp_bts = exp_bts.repeat(input_n.shape + (1,)).to(device)
    bits = torch.div(num_.unsqueeze(-1), 2 ** exp_bts, rounding_mode='trunc')
    bits = bits % 2
    bits = bits.reshape(bits.shape[0], -1).float().to(device)
    return bits


class Bitflow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, b_):
        ctx.constant = b_
        scale = 2 ** b_
        out = torch.round(x * scale - 0.5)
        out = convert2bit(out, b_)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2) / ctx.constant
        return grad_num, None

class BitLayer(nn.Module):
    def __init__(self, B):
        super(BitLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Bitflow.apply(x, self.B)
        return out

    
class ConvNet(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv3d(input_channel, output_channel, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm3d(output_channel)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ConvTransposeNet(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ConvTransposeNet, self).__init__()
        self.conv = nn.ConvTranspose3d(input_channel, output_channel, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm3d(output_channel)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super(Encoder, self).__init__()
        self.avg_pooling = torch.nn.AdaptiveAvgPool3d((32, 32, 32))
        self.conv_block = nn.Sequential(
            ConvNet(input_dim, 32),
            ConvNet(32, 64),
            ConvNet(64, 128),
            ConvNet(128, 256),
            ConvNet(256, 512),
        )
        self.feat = nn.Linear(512, latent_dim // 2)
        self.sigmoid = nn.Sigmoid()
        self.bit_layer = BitLayer(2)

    def forward(self, x):
        x = self.avg_pooling(x)
        x = self.conv_block(x).reshape(x.shape[0], 512)
        feat = self.feat(x)
        feat = self.sigmoid(feat)
        feat = self.bit_layer(feat)
        return feat

class BaseLineDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BaseLineDecoder, self).__init__()
        linear_block = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.BatchNorm1d(2048),
            # nn.Dropout(0.2),
            nn.LeakyReLU(),
        )
        conv_block = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
        )

        self.linear_block = linear_block
        self.conv_block = conv_block

        self.head = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.pam = PAM_Module()
        self.cam = CAM_Module()
        self.layer_norm = nn.LayerNorm((8, 32, 32))

    def forward(self, input):
        output = self.linear_block(input)
        output = output.view(-1, 512, 2, 2)
        output = self.conv_block(output) # [batch, 16, 64, 64]
        output = self.head(output)  # [batch, 8, 32, 32]
        output = output * self.sigmoid(self.pam(output) + self.cam(output))
        output = self.layer_norm(output)
        output = self.sigmoid(output) - 0.5
        output = output.permute(0, 2, 3, 1).reshape(-1, 4, 2, 32, 32).permute(0, 1, 3, 4, 2)
        return output

class ResVAE(nn.Module):
    def __init__(self, input_dim=2, latent_dim=48):
        super(ResVAE, self).__init__()
        self.latent_dim = latent_dim
        self.reshape_input = nn.AdaptiveAvgPool2d(32)
        self.encoder = ResNet18(32, latent_dim)
        self.decoder = BaseLineDecoder(input_dim, latent_dim)

    def forward(self, x):
        tx = x.permute(0, 1, 4, 2, 3).reshape(-1, 16, 16, 32).permute(0, 3, 1, 2)
        tx = self.reshape_input(tx)
        feat = self.encoder(tx)
        recon = self.decoder(feat)
        return recon

    def sample(self, size):
        noise = torch.randint(2, (size, self.latent_dim)).float().to(device)
        recon = self.decoder(noise)
        return recon


def f(score):
    p = 0.5 * np.log(score / (1 - score))
    return p


# def generator_1(num_fake_1, file_generator_1, file_real_1=None):
#     params = torch.load(file_generator_1, map_location=device)
#     generators, weights = dict(), list()
#
#     for key in params.keys():
#         if 'resnet_vae' not in key:
#             generators[key] = encoder_2d().to(device) if '2d' in key else encoder_3d().to(device)
#             generators[key].load_state_dict(params[key])
#             generators[key].eval()
#         else:
#             generators[key] = ResVAE().to(device)
#             generators[key].load_state_dict(params[key])
#             generators[key].eval()
#         score = float(key.split('_')[-1].replace('.pth', ''))
#         weights.append(f(score))
#     weights = np.array(weights) / (np.sum(np.array(weights)))
#     generators = [generators[key] for key in generators]
#     print(weights)
#
#     size_packet = 100
#     with torch.no_grad():
#         for idx in range(int(num_fake_1 / size_packet)):
#             if idx == int(num_fake_1 / size_packet) - 1:
#                 size_packet = size_packet + num_fake_1 % size_packet
#
#             generator_C = np.random.choice(generators, 1, p=weights)[0]
#             fake_data = generator_C.sample(size_packet)
#             fake_real_part, fake_imag_part = fake_data[..., 0], fake_data[..., 1]
#             fake_real_part, fake_imag_part = fake_real_part.detach().cpu().numpy(), fake_imag_part.detach().cpu().numpy()
#             fake_data_reshape = fake_real_part + fake_imag_part * 1j
#             if idx == 0:
#                 data_fake_all = fake_data_reshape
#             else:
#                 data_fake_all = np.concatenate((data_fake_all, fake_data_reshape), axis=0)
#     return data_fake_all


# single model
def generator_1(num_fake_1, file_generator_1, file_real_1=None):
    generator_C = ResVAE()
    generator_C.load_state_dict(torch.load(file_generator_1, map_location='cpu'))
    generator_C = generator_C.to(device).eval()

    size_packet = 500
    with torch.no_grad():
        for idx in range(int(num_fake_1 / size_packet)):
            if idx == int(num_fake_1 / size_packet) - 1:
                size_packet = size_packet + num_fake_1 % size_packet
            fake_data = generator_C.sample(size_packet)
            fake_real_part, fake_imag_part = fake_data[..., 0], fake_data[..., 1]
            fake_real_part, fake_imag_part = fake_real_part.detach().cpu().numpy(), fake_imag_part.detach().cpu().numpy()
            fake_data_reshape = fake_real_part + fake_imag_part * 1j
            if idx == 0:
                data_fake_all = fake_data_reshape
            else:
                data_fake_all = np.concatenate((data_fake_all, fake_data_reshape), axis=0)
    return data_fake_all


# single model
def generator_2(num_fake_1, file_generator_1, file_real_1=None):
    generator_C = ResVAE()
    generator_C.load_state_dict(torch.load(file_generator_1, map_location=device))
    generator_C = generator_C.to(device).eval()

    size_packet = 500
    with torch.no_grad():
        for idx in range(int(num_fake_1 / size_packet)):
            if idx == int(num_fake_1 / size_packet) - 1:
                size_packet = size_packet + num_fake_1 % size_packet
            fake_data = generator_C.sample(size_packet)
            fake_real_part, fake_imag_part = fake_data[..., 0], fake_data[..., 1]
            fake_real_part, fake_imag_part = fake_real_part.detach().cpu().numpy(), fake_imag_part.detach().cpu().numpy()
            fake_data_reshape = fake_real_part + fake_imag_part * 1j
            if idx == 0:
                data_fake_all = fake_data_reshape
            else:
                data_fake_all = np.concatenate((data_fake_all, fake_data_reshape), axis=0)
    return data_fake_all


# Boost model
# def generator_2(num_fake_2, file_generator_2, file_real_2=None):
#     params = torch.load(file_generator_2, map_location=device)
#     generators, weights = dict(), list()
#
#     for key in params.keys():
#         generators[key] = ResVAE().to(device)
#         generators[key].load_state_dict(params[key])
#         generators[key].eval()
#         score = float(key.split('_')[-1].replace('.pth', ''))
#         weights.append(f(score))
#     weights = np.array(weights) / (np.sum(np.array(weights)))
#     generators = [generators[key] for key in generators]
#     print(weights)
#
#     size_packet = 100
#     with torch.no_grad():
#         for idx in range(int(num_fake_2 / size_packet)):
#             if idx == int(num_fake_2 / size_packet) - 1:
#                 size_packet = size_packet + num_fake_2 % size_packet
#             generator_C = np.random.choice(generators, 1, p=weights)[0]
#             fake_data = generator_C.sample(size_packet)
#             fake_real_part, fake_imag_part = fake_data[..., 0], fake_data[..., 1]
#             fake_real_part, fake_imag_part = fake_real_part.detach().cpu().numpy(), fake_imag_part.detach().cpu().numpy()
#             fake_data_reshape = fake_real_part + fake_imag_part * 1j
#             if idx == 0:
#                 data_fake_all = fake_data_reshape
#             else:
#                 data_fake_all = np.concatenate((data_fake_all, fake_data_reshape), axis=0)
#     return data_fake_all


if __name__ == '__main__':
    NUM_FAKE_1 = 500
    NUM_FAKE_2 = 4000
    fake_1 = generator_1(NUM_FAKE_1, 'generator_1.pth.tar', None)
    fake_2 = generator_2(NUM_FAKE_2, 'generator_2.pth.tar', None)
    print(fake_1.shape, fake_2.shape)

    # from eval import load_test, K_nearest
    # real1 = load_test(1)
    # real2 = load_test(2)
    # sim1, multi1, _ = K_nearest(real1, fake_1, 4, 32, 32, 1)
    # sim2, multi2, _ = K_nearest(real2, fake_2, 4, 32, 32, 1)

    # print('Data1 | Sim : %.3f | Multi : %.3f | Score : %.3f' % (sim1, multi1, 1-multi1/sim1/20))
    # print('Data2 | Sim : %.3f | Multi : %.3f | Score : %.3f' % (sim2, multi2, 1-multi2/sim2/40))