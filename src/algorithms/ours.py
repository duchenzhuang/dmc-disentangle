import algorithms.modules as m
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.sac import SAC


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class OURS(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.aux_update_freq = args.aux_update_freq
        self.aux_lr = args.aux_lr
        self.aux_beta = args.aux_beta

        shared_cnn = self.critic.encoder.shared_cnn
        aux_cnn = m.HeadCNN(shared_cnn.out_shape, args.num_head_layers, args.num_filters).cuda()
        aux_encoder = m.Encoder(
            shared_cnn,
            aux_cnn,
            m.RLProjection(aux_cnn.out_shape, args.projection_dim, args.disentangle, args.ccm_dim, args.task_dim_rate)
        ).cuda()


        self.disentangle = args.disentangle
        self.ccm_lambda = args.ccm_lambda
        if self.disentangle:
            self.ccm_head = aux_encoder
        else:
            self.ccm_head = m.CCMHead(aux_encoder, args.ccm_dim).cuda()

        self.bn = nn.BatchNorm1d(args.ccm_dim, affine=False).cuda()

        self.init_optimizer()
        self.train()

    def train(self, training=True):
        super().train(training)
        if hasattr(self, 'ccm_head'):
            self.ccm_head.train(training)

    def init_optimizer(self):
        self.ccm_optimizer = torch.optim.Adam(
            self.ccm_head.parameters(), lr=self.aux_lr, betas=(self.aux_beta, 0.999)
        )
    def update_ccm(self, x, x_pos, L=None, step=None):
        assert x.size(-1) == 84 and x_pos.size(-1) == 84

        self.ccm_optimizer.zero_grad()
        # identical encoders
        z_1 = self.ccm_head.ccm_forward(x)
        z_2 = self.ccm_head.ccm_forward(x_pos)

        # empirical cross-correlation matrix
        c = self.bn(z_1).T @ self.bn(z_2)

        # cross-correlation matrix
        c.div_(z_1.size(0))  # DxD

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(1/32)
        off_diag = off_diagonal(c).pow_(2).sum().mul(1/32)
        ccm_loss = self.ccm_lambda * (on_diag + 3.9e-3 * off_diag)

        ccm_loss.backward()
        self.ccm_optimizer.step()
        if L is not None:
            L.log('train/ccm_loss', ccm_loss, step)

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done, pos = replay_buffer.sample_curl()

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

        if step % self.aux_update_freq == 0:
            self.update_ccm(obs, obs, L, step)