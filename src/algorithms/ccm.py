import algorithms.modules as m
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.sac import SAC
import augmentations
import rad_augmentation as rad
from copy import deepcopy
import utils

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class CCM(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.aux_update_freq = args.aux_update_freq
        self.aux_lr = args.aux_lr
        self.aux_beta = args.aux_beta
        self.soda_batch_size = args.soda_batch_size

        shared_cnn = self.critic.encoder.shared_cnn
        aux_cnn = m.HeadCNN(shared_cnn.out_shape, args.num_head_layers, args.num_filters).cuda()
        aux_encoder = m.Encoder(
            shared_cnn,
            aux_cnn,
            m.RLProjection(aux_cnn.out_shape, args.projection_dim)
        ).cuda()

        self.autoEncoder = m.Decoder(aux_encoder, obs_shape=obs_shape).cuda()

        self.ccm_lambda = args.ccm_lambda
        self.ccm_head = m.CCMHead(aux_encoder, args.hidden_dim).cuda()
        self.bn = nn.BatchNorm1d(args.hidden_dim, affine=False).cuda()

        self.pad_head = m.InverseDynamics(aux_encoder, action_shape, args.hidden_dim).cuda()


        # NEG
        self.soda_tau = args.soda_tau
        self.predictor = m.SODAPredictor(aux_encoder, args.projection_dim).cuda()
        self.predictor_target = deepcopy(self.predictor)
        self.neg_optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=args.aux_lr, betas=(args.aux_beta, 0.999)
        )


        # rad
        # self.data_augs = data_augs
        # self.augs_funcs = {}
        self.augs_funcs = {
                'crop':rad.random_crop,
                'grayscale':rad.random_grayscale,
                'cutout':rad.random_cutout,
                'cutout_color':rad.random_cutout_color,
                # 'flip':rad.random_flip,
                # 'rotate':rad.random_rotation,
                'rand_conv':rad.random_convolution,
                'color_jitter':rad.random_color_jitter,
                'translate':rad.random_translate,
                'no_aug':rad.no_aug,
            }

        # for aug_name in self.data_augs.split('-'):
        #     assert aug_name in aug_to_func, 'invalid data aug string'
        #     self.augs_funcs[aug_name] = aug_to_func[aug_name]


        self.init_optimizer()
        self.train()

    def train(self, training=True):
        super().train(training)
        if hasattr(self, 'pad_head'):
            self.pad_head.train(training)
        if hasattr(self, 'ccm_head'):
            self.ccm_head.train(training)
        if hasattr(self, 'autoEncoder'):
            self.autoEncoder.train(training)

    def init_optimizer(self):
        self.pad_optimizer = torch.optim.Adam(
            self.pad_head.parameters(), lr=self.aux_lr, betas=(self.aux_beta, 0.999)
        )
        self.ccm_optimizer = torch.optim.Adam(
            self.ccm_head.parameters(), lr=self.aux_lr, betas=(self.aux_beta, 0.999)
        )
        self.ae_optimizer = torch.optim.Adam(
            self.autoEncoder.parameters(), lr=self.aux_lr, betas=(self.aux_beta, 0.999)
        )

    def update_inverse_dynamics(self, obs, obs_next, action, L=None, step=None):

        assert obs.shape[-1] == 84 and obs_next.shape[-1] == 84

        pred_action = self.pad_head(obs, obs_next)
        pad_loss = F.mse_loss(pred_action, action)

        self.pad_optimizer.zero_grad()
        pad_loss.backward()
        self.pad_optimizer.step()
        if L is not None:
            L.log('train/aux_loss', pad_loss, step)

    def update_ccm(self, x, x_pos, L=None, step=None):
        assert x.size(-1) == 84 and x_pos.size(-1) == 84

        self.ccm_optimizer.zero_grad()
        # identical encoders
        z_1 = self.ccm_head(x)
        z_2 = self.ccm_head(x_pos)

        # empirical cross-correlation matrix
        c = self.bn(z_1).T @ self.bn(z_2)

        # cross-correlation matrix
        c.div_(z_1.size(0))  # DxD

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(1/32)
        off_diag = off_diagonal(c).pow_(2).sum().mul(1/32)
        ccm_loss = 0.01 * (on_diag + 3.9e-3 * off_diag)

        ccm_loss.backward()
        self.ccm_optimizer.step()
        if L is not None:
            L.log('train/ccm_loss', ccm_loss, step)

    def update_soda_ccm(self, x, size, L=None, step=None):
        assert x.size(-1) == size

        aug_x = x.clone()

        x = augmentations.random_crop(x)
        aug_x = augmentations.random_crop(aug_x)
        x_pos = augmentations.random_overlay(aug_x)

        self.ccm_optimizer.zero_grad()
        # identical encoders
        z_1 = self.ccm_head(x)
        z_2 = self.ccm_head(x_pos)

        # empirical cross-correlation matrix
        c = self.bn(z_1).T @ self.bn(z_2)

        # cross-correlation matrix
        c.div_(z_1.size(0))  # DxD

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(1/32)
        off_diag = off_diagonal(c).pow_(2).sum().mul(1/32)
        ccm_loss = 0.01 * (on_diag + 3.9e-3 * off_diag)

        ccm_loss.backward()
        self.ccm_optimizer.step()
        if L is not None:
            L.log('train/ccm_loss', ccm_loss, step)

    def compute_neg_loss(self, x0, x1, margin):
        h0 = self.predictor(x0)
        with torch.no_grad():
            h1 = self.predictor_target.encoder(x1)
        # h0 = F.normalize(h0, p=2, dim=1)
        # h1 = F.normalize(h1, p=2, dim=1)
        # dist = F.mse_loss(h0, h1)
        dist = F.cosine_similarity(h0, h1)
        return (margin - dist).pow(2).sum()

    def update_neg_rad(self, x, neg, anchor, L=None, step=None):
        neg_loss = self.compute_neg_loss(x, neg, anchor)

        self.neg_optimizer.zero_grad()
        neg_loss.backward()
        self.neg_optimizer.step()

        utils.soft_update_params(
            self.predictor, self.predictor_target,
            self.soda_tau
        )
        if L is not None:
            L.log('train/neg_loss', neg_loss, step)

    def update_ac(self, obs, L, step):
        h, rec_obs = self.autoEncoder(obs)
        rec_loss = F.mse_loss(obs, rec_obs)

        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + 1e-6 * latent_loss

        self.ae_optimizer.zero_grad()
        loss.backward()
        self.ae_optimizer.step()
        L.log('train/ae_loss', loss, step)


    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done, pos = replay_buffer.sample_curl()
        # obs, action, reward, next_obs, not_done = replay_buffer.sample()
        # ori, obs, action, reward, next_obs, not_done = replay_buffer.sample_rad(self.augs_funcs)
        # obs, action, reward, next_obs, not_done, pos = replay_buffer.sample_rad(self.augs_funcs)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

        if step % self.aux_update_freq == 0:
            # new_obs, new_action, new_next_obs = replay_buffer.sample_multi_views()

            # self.update_inverse_dynamics(torch.cat((obs, new_obs), 0),
            #                              torch.cat((next_obs, new_next_obs), 0),
            #                              torch.cat((action, new_action), 0),
            #                              L, step)

            # self.update_ccm(torch.cat((obs, next_obs), 0),
            #                 torch.cat((new_obs, new_next_obs), 0),
            #                 L,
            #                 step)
            # self.update_inverse_dynamics(obs, next_obs, action, L, step)
            # self.update_ccm(obs, pos, L, step)
            # self.update_ac(obs, L, step)


            # # soda augmentation + ccm + same obs
            # self.update_soda_ccm(obs, 84, L, step)


            # soda augmentation + ccm + diff
            # obs = replay_buffer.sample_soda(self.soda_batch_size)
            # self.update_soda_ccm(obs, 100, L, step)



            # # rad augmentation + ccm
            # self.update_ccm(obs, pos, L, step)



            # # rad augmentation + ccm + diff
            obs, action, reward, next_obs, not_done, pos = replay_buffer.sample_rad(self.augs_funcs)
            self.update_ccm(obs, pos, L, step)


            # NEG SAMPLE
            obs, neg, anchor = replay_buffer.sample_neg(self.soda_batch_size)
            self.update_neg_rad(obs, neg, anchor, L, step)