import algorithms.modules as m
import torch
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
            m.RLProjection(aux_cnn.out_shape, args.projection_dim)
        ).cuda()

        self.ccm_target = torch.eye(args.projection_dim).cuda()
        self.ccm_mask = ~torch.eye(args.projection_dim, dtype=bool).cuda()
        self.ccm_lambda = 0.005
        self.ccm_head = m.CURLHead(shared_cnn).cuda()
        self.ccm_optimizer = torch.optim.Adam(
            self.ccm_head.parameters(), lr=args.aux_lr, betas=(args.aux_beta, 0.999)
        )


        self.pad_head = m.InverseDynamics(aux_encoder, action_shape, args.hidden_dim).cuda()
        self.init_pad_optimizer()
        self.train()

    def train(self, training=True):
        super().train(training)
        if hasattr(self, 'pad_head'):
            self.pad_head.train(training)

    def init_pad_optimizer(self):
        self.pad_optimizer = torch.optim.Adam(
            self.pad_head.parameters(), lr=self.aux_lr, betas=(self.aux_beta, 0.999)
        )

    def update_inverse_dynamics(self, obs, obs_next, action, L=None, step=None):
        assert obs.shape[-1] == 84 and obs_next.shape[-1] == 84

        pred_action = self.pad_head(obs, obs_next)
        pad_loss = F.mse_loss(pred_action, action)

        self.ccm_head.zero_grad()
        pad_loss.backward()
        self.ccm_optimizer.step()
        if L is not None:
            L.log('train/aux_loss', pad_loss, step)

    def update_ccm(self, x, x_pos, L=None, step=None):
        assert x.size(-1) == 84 and x_pos.size(-1) == 84

        # identical encoders
        z_a = self.pad_head.encoder(x)
        z_pos = self.pad_head.encoder(x_pos)

        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # NxD
        z_pos_norm = (z_pos - z_pos.mean(0)) / z_pos.std(0)  # NxD

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_pos_norm) / z_a.size(0)  # DxD
        c_diff = (c - self.ccm_target).pow(2)  # DxD
        c_diff[self.ccm_mask] *= self.ccm_lambda  # non-diag elements multiply with lambda
        ccm_loss = c_diff.sum()

        self.pad_optimizer.zero_grad()
        ccm_loss.backward()
        self.pad_optimizer.step()
        if L is not None:
            L.log('train/ccm_loss', ccm_loss, step)


    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

        if step % self.aux_update_freq == 0:
            new_obs, new_action, new_next_obs = replay_buffer.sample_multi_views()

            self.update_inverse_dynamics(torch.cat((obs, new_obs), 0),
                                         torch.cat((next_obs, new_next_obs), 0),
                                         torch.cat((action, new_action), 0),
                                         L, step)

            self.update_ccm(torch.cat((obs, next_obs), 0),
                            torch.cat((new_obs, new_next_obs), 0),
                            L,
                            step)