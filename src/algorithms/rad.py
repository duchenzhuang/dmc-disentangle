import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import algorithms.modules as m
from algorithms.sac import SAC
import rad_augmentation as rad

class RAD(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.augs_funcs = {
			'crop': rad.random_crop,
			'grayscale': rad.random_grayscale,
			'cutout': rad.random_cutout,
			'cutout_color': rad.random_cutout_color,
			'flip': rad.random_flip,
			'rotate': rad.random_rotation,
			'rand_conv': rad.random_convolution,
			'color_jitter': rad.random_color_jitter,
			'translate': rad.random_translate,
			'no_aug': rad.no_aug,
		}

	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_rad_norm(self.augs_funcs)
		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()


