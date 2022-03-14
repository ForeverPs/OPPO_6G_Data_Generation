import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Position Attention Module
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


# Channel Attention Module
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


# ResNet Block
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

# ResNet BottleNeck
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

# ResNet Backbone As Encoder
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
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.feat(out)
        out = self.sigmoid(out)
        out = self.bit_layer(out)
        return out


def ResNet18(input_channel=32, hidden_dim=128):
    return ResNet(BasicBlock, [2, 2, 2, 2], input_channel, hidden_dim)


def ResNet34(input_channel=32, hidden_dim=128):
    return ResNet(BasicBlock, [3, 4, 6, 3], input_channel, hidden_dim)


def ResNet50(input_channel=32, hidden_dim=128):
    return ResNet(Bottleneck, [3, 4, 6, 3], input_channel, hidden_dim)


def ResNet101(input_channel=32, hidden_dim=128):
    return ResNet(Bottleneck, [3, 4, 23, 3], input_channel, hidden_dim)


def ResNet152(input_channel=32, hidden_dim=128):
    return ResNet(Bottleneck, [3, 8, 36, 3], input_channel, hidden_dim)


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

# ConvBlock for VAE3D Encoder    
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

# Upsampling for VAE3D Decoder
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

# VAE3D Encoder
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

# VAE3D Decoder
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
    
    def loss(self, target, predict):
        predict = predict.reshape((-1, 4 * 32 * 32, 2))
        predict_complex = torch.complex(predict[..., 0], predict[..., 1])
        predict = F.normalize(predict_complex, p=2, dim=1)

        target = target.reshape((-1, 4 * 32 * 32, 2))
        target_complex = torch.complex(target[..., 0], target[..., 1])
        target = F.normalize(target_complex, p=2, dim=1)
        recon_loss = F.mse_loss(predict.real, target.real) + F.mse_loss(predict.imag, target.imag)
        return 1000 * recon_loss


if __name__ == '__main__':
    model = ResVAE().eval().to(device)
    model.load_state_dict(torch.load('saved_models/2/att_sim_0.212_multi_1.918_score_0.774.pth', map_location=device))
    recon = model.sample(10)
    x = torch.randn(10, 128)
    print(recon.shape)
