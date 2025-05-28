import torch
import torch.nn as nn
import torch.nn.functional as F
from mesh2tex import common
from mesh2tex.layers import (
    ResnetBlockPointwise,
    EqualizedLR
)
from performer_pytorch import SelfAttention


class DecoderEachLayerC(nn.Module):
    def __init__(self, c_dim=128, z_dim=128, dim=3,
                 hidden_size=128, leaky=True, 
                 resnet_leaky=True, eq_lr=False):
        super().__init__()
        self.c_dim = c_dim
        self.eq_lr = eq_lr

        # Submodules
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        if not resnet_leaky:
            self.resnet_actvn = F.relu
        else:
            self.resnet_actvn = lambda x: F.leaky_relu(x, 0.2)

        self.conv_p = nn.Conv1d(dim, hidden_size, 1)

        self.block0 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block1 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block2 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block3 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block4 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)

        self.fc_cz_0 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_1 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_2 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_3 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_4 = nn.Linear(c_dim + z_dim, hidden_size)

        self.conv_out = nn.Conv1d(hidden_size, 3, 1)

        if self.eq_lr:
            self.conv_p = EqualizedLR(self.conv_p)
            self.conv_out = EqualizedLR(self.conv_out)
            self.fc_cz_0 = EqualizedLR(self.fc_cz_0)
            self.fc_cz_1 = EqualizedLR(self.fc_cz_1)
            self.fc_cz_2 = EqualizedLR(self.fc_cz_2)
            self.fc_cz_3 = EqualizedLR(self.fc_cz_3)
            self.fc_cz_4 = EqualizedLR(self.fc_cz_4)

        # Initialization
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, p, geom_descr, z, **kwargs):
        c = geom_descr['global']
        batch_size, D, T = p.size()

        cz = torch.cat([c, z], dim=1)
        net = self.conv_p(p)
        net = net + self.fc_cz_0(cz).unsqueeze(2)
        net = self.block0(net)
        net = net + self.fc_cz_1(cz).unsqueeze(2)
        net = self.block1(net)
        net = net + self.fc_cz_2(cz).unsqueeze(2)
        net = self.block2(net)
        net = net + self.fc_cz_3(cz).unsqueeze(2)
        net = self.block3(net)
        net = net + self.fc_cz_4(cz).unsqueeze(2)
        net = self.block4(net)

        out = self.conv_out(self.actvn(net))
        out = torch.sigmoid(out)

        return out


class DecoderEachLayerCLarger(nn.Module):
    def __init__(self, c_dim=128, z_dim=128, dim=3,
                 hidden_size=128, leaky=True, 
                 resnet_leaky=True, eq_lr=False,
                 use_attention=False, attention_heads=1,
                 num_blocks=7,**kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.eq_lr = eq_lr
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
        
        if not resnet_leaky:
            self.resnet_actvn = F.relu
        else:
            self.resnet_actvn = lambda x: F.leaky_relu(x, 0.2)

        # Submodules
        self.conv_p = nn.Conv1d(dim, hidden_size, 1)

        self.block0 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block1 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block2 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block3 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block4 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block5 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)
        self.block6 = ResnetBlockPointwise(
            hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr)

        self.fc_cz_0 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_1 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_2 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_3 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_4 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_5 = nn.Linear(c_dim + z_dim, hidden_size)
        self.fc_cz_6 = nn.Linear(c_dim + z_dim, hidden_size)

        self.conv_out = nn.Conv1d(hidden_size, 3, 1)

        if self.eq_lr:
            self.conv_p = EqualizedLR(self.conv_p)
            self.conv_out = EqualizedLR(self.conv_out)
            self.fc_cz_0 = EqualizedLR(self.fc_cz_0)
            self.fc_cz_1 = EqualizedLR(self.fc_cz_1)
            self.fc_cz_2 = EqualizedLR(self.fc_cz_2)
            self.fc_cz_3 = EqualizedLR(self.fc_cz_3)
            self.fc_cz_4 = EqualizedLR(self.fc_cz_4)
            self.fc_cz_5 = EqualizedLR(self.fc_cz_5)
            self.fc_cz_6 = EqualizedLR(self.fc_cz_6)

        # Initialization
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, p, geom_descr, z, **kwargs):
        c = geom_descr['global']
        batch_size, D, T = p.size()

        cz = torch.cat([c, z], dim=1)

        net = self.conv_p(p)
        net = net + self.fc_cz_0(cz).unsqueeze(2)
        net = self.block0(net)
        net = net + self.fc_cz_1(cz).unsqueeze(2)
        net = self.block1(net)
        net = net + self.fc_cz_2(cz).unsqueeze(2)
        net = self.block2(net)
        net = net + self.fc_cz_3(cz).unsqueeze(2)
        net = self.block3(net)
        net = net + self.fc_cz_4(cz).unsqueeze(2)
        net = self.block4(net)
        net = net + self.fc_cz_5(cz).unsqueeze(2)
        net = self.block5(net)
        net = net + self.fc_cz_6(cz).unsqueeze(2)
        net = self.block6(net)

        out = self.conv_out(self.actvn(net))
        out = torch.sigmoid(out)

        return out



class DecoderEachLayerCLarger_Attention(nn.Module):
    def __init__(self, c_dim=128, z_dim=128, dim=3,
                 hidden_size=64, leaky=True, 
                 resnet_leaky=True, eq_lr=False,
                 use_attention=True, attention_heads=4, num_blocks=7, **kwargs): 
        super().__init__()
        self.c_dim = c_dim
        self.eq_lr = eq_lr
        self.use_attention = use_attention  

        self.actvn = F.relu if not leaky else lambda x: F.leaky_relu(x, 0.2)
        self.resnet_actvn = F.relu if not resnet_leaky else lambda x: F.leaky_relu(x, 0.2)

        self.conv_p = nn.Conv1d(dim, hidden_size, 1)

        self.blocks = nn.ModuleList([
            ResnetBlockPointwise(hidden_size, actvn=self.resnet_actvn, eq_lr=eq_lr) for _ in range(num_blocks)
        ])

        self.fc_cz = nn.ModuleList([
            nn.Linear(c_dim + z_dim, hidden_size) for _ in range(num_blocks)
        ])

        self.conv_out = nn.Conv1d(hidden_size, 3, 1)

        if self.eq_lr:
            self.conv_p = EqualizedLR(self.conv_p)
            self.conv_out = EqualizedLR(self.conv_out)
            self.fc_cz = nn.ModuleList([EqualizedLR(fc) for fc in self.fc_cz])

        if self.use_attention:
            self.attn = SelfAttention(dim=hidden_size, heads=attention_heads)

        nn.init.zeros_(self.conv_out.weight)

    def forward(self, p, geom_descr, z, **kwargs):
        c = geom_descr['global']
        cz = torch.cat([c, z], dim=1)

        net = self.conv_p(p)
        for i, (block, fc) in enumerate(zip(self.blocks, self.fc_cz)):
            net = net + fc(cz).unsqueeze(2)
            net = block(net)

            if self.use_attention and i == 3:
                net_perm = net.permute(0, 2, 1)
                net_attn = self.attn(net_perm)
                net = net + net_attn.permute(0, 2, 1)

        out = self.conv_out(self.actvn(net))
        out = torch.sigmoid(out)
        return out
