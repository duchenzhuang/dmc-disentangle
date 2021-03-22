import algorithms.modules as m
import torch
import torch.nn.functional as F
from algorithms.sac import SAC


class CCM(SAC):
	"""
	Naive CCM for redundancy reduction, ref to LeCun.
	Based on CURL Samples for now.
	"""
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.aux_update_freq = args.aux_update_freq

		self.ccm_head = m.CURLHead(self.critic.encoder).cuda()

		self.ccm_optimizer = torch.optim.Adam(
			self.ccm_head.parameters(), lr=args.aux_lr, betas=(args.aux_beta, 0.999)
		)

		# Target Matrix: using Identity for now.
		self.ccm_target = torch.eye(args.projection_dim).cuda()
		self.ccm_mask = ~torch.eye(args.projection_dim, dtype=bool).cuda()
		self.ccm_lambda = 0.005

		# # define mask matrix
		# self.half_latent_size = args.hidden_dim // 2
		# M1 = torch.zeros((self.half_latent_size, self.half_latent_size))
		# M2 = torch.ones((self.half_latent_size, self.half_latent_size))
		# self.ccm_mask = torch.cat((torch.cat((M1, M2)), torch.cat((M2, M1))), dim=1).bool().cuda()

		self.train()

	def train(self, training=True):
		super().train(training)
		if hasattr(self, 'ccm_head'):
			self.ccm_head.train(training)

	def update_ccm(self, x, x_pos, L=None, step=None):
		assert x.size(-1) == 84 and x_pos.size(-1) == 84

        # identical encoders
		z_a = self.ccm_head.encoder(x)
		z_pos = self.ccm_head.encoder(x_pos)

		z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # NxD
		z_pos_norm = (z_pos - z_pos.mean(0)) / z_pos.std(0)  # NxD

		# cross-correlation matrix
		c = torch.mm(z_a_norm.T, z_pos_norm) / z_a.size(0)  # DxD
		c_diff = (c - self.ccm_target).pow(2)  # DxD
		c_diff[self.ccm_mask] *= self.ccm_lambda  # non-diag elements multiply with lambda
		ccm_loss = 0.01 * c_diff.sum()

		self.ccm_optimizer.zero_grad()
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
			self.update_ccm(obs, pos, L, step)
