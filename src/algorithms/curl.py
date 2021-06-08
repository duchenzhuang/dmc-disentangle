import algorithms.modules as m
import torch
import torch.nn.functional as F
from algorithms.sac import SAC
import augmentations


class CURL(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.aux_update_freq = args.aux_update_freq
		self.soda_batch_size = args.soda_batch_size

		self.curl_head = m.CURLHead(self.critic.encoder).cuda()

		self.curl_optimizer = torch.optim.Adam(
			self.curl_head.parameters(), lr=args.aux_lr, betas=(args.aux_beta, 0.999)
		)
		self.use_intrinsic = args.use_intrinsic
		self.in_gamma = args.in_gamma
		self.in_decay = args.in_decay

		self.train()

	def train(self, training=True):
		super().train(training)
		if hasattr(self, 'curl_head'):
			self.curl_head.train(training)

	def update_curl(self, x, x_pos, L=None, step=None):
		assert x.size(-1) == 84 and x_pos.size(-1) == 84

		z_a = self.curl_head.encoder(x)
		with torch.no_grad():
			z_pos = self.critic_target.encoder(x_pos)
		
		logits = self.curl_head.compute_logits(z_a, z_pos)
		labels = torch.arange(logits.shape[0]).long().cuda()
		curl_loss = F.cross_entropy(logits, labels)
		
		self.curl_optimizer.zero_grad()
		curl_loss.backward()
		self.curl_optimizer.step()
		if L is not None:
			L.log('train/aux_loss', curl_loss, step)

	def update_soda_curl(self, x, size, L=None, step=None):
		assert x.size(-1) == size

		aug_x = x.clone()

		x = augmentations.random_crop(x)
		aug_x = augmentations.random_crop(aug_x)
		x_pos = augmentations.random_overlay(aug_x)

		z_a = self.curl_head.encoder(x)
		with torch.no_grad():
			z_pos = self.critic_target.encoder(x_pos)

		logits = self.curl_head.compute_logits(z_a, z_pos)
		labels = torch.arange(logits.shape[0]).long().cuda()
		curl_loss = F.cross_entropy(logits, labels)

		self.curl_optimizer.zero_grad()
		curl_loss.backward()
		self.curl_optimizer.step()
		if L is not None:
			L.log('train/aux_loss', curl_loss, step)

	def compute_intri_reward(self, next_state, next_state_pos, L, step):
		with torch.no_grad():
			z_a = self.curl_head.encoder(next_state)
			z_pos = self.critic_target.encoder(next_state_pos)

			logits = self.curl_head.compute_logits(z_a, z_pos)
			labels = torch.arange(logits.shape[0]).long().cuda()

			exp = torch.exp(logits)
			tmp1 = exp.gather(1, labels.unsqueeze(-1)).squeeze()
			tmp2 = exp.sum(1)
			softmax = tmp1 / tmp2
			intri_reward = - torch.log(softmax)
			norm_intri_reward = (intri_reward - intri_reward.mean()) / intri_reward.std()
		if L is not None:
			L.log('train/reward_mean', intri_reward.mean(), step)
			L.log('train/reward_std', intri_reward.std(), step)
			L.log('train/norm_reward_max', norm_intri_reward.max(), step)
			L.log('train/norm_reward_min', norm_intri_reward.min(), step)
		return norm_intri_reward.unsqueeze(1)

	# def intrinsic_reward(self, x, x_pos):
	# 	assert x.size(-1) == 84 and x_pos.size(-1) == 84
	#
	# 	z_a = self.curl_head.encoder(x)
	# 	with torch.no_grad():
	# 		z_pos = self.critic_target.encoder(x_pos)
	#
	# 	logits = self.curl_head.compute_logits(z_a, z_pos)
	# 	labels = torch.arange(logits.shape[0]).long().cuda()
	# 	curl_loss = F.cross_entropy(logits, labels)
	#
	# 	return torch.log(curl_loss + 1)


	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done, pos, next_pos = replay_buffer.sample_curl_intri()

		# reward += 0.01 * self.intrinsic_reward(obs, pos)
		if step % 100000 == 0 and step != 0:
			print("==decaying==", self.in_gamma)
			self.in_gamma = self.in_gamma * self.in_decay
			print(self.in_gamma)

		if self.use_intrinsic:
			reward += self.in_gamma * self.compute_intri_reward(next_obs, next_pos, L, step)

		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()

		if step % self.aux_update_freq == 0:
			# ori curl
			self.update_curl(obs, pos, L, step)
			#
			# # diff curl
			# obs, action, reward, next_obs, not_done, pos = replay_buffer.sample_curl()
			# self.update_curl(obs, pos, L, step)

			# soda + curl
			# self.update_soda_curl(pos, 84, L, step)

			# # soda + curl + diff
			# obs = replay_buffer.sample_soda(self.soda_batch_size)
			# self.update_soda_curl(obs, 100, L, step)
