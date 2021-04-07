import json
import os
import random
from datetime import datetime

import numpy as np
import torch

import augmentations
import rad_augmentation as rad
import torchvision.transforms as TF

class eval_mode(object):
	def __init__(self, *models):
		self.models = models

	def __enter__(self):
		self.prev_states = []
		for model in self.models:
			self.prev_states.append(model.training)
			model.train(False)

	def __exit__(self, *args):
		for model, state in zip(self.models, self.prev_states):
			model.train(state)
		return False

def preprocess_obs(obs, bits=5):
	"""Preprocessing image, see https://arxiv.org/abs/1807.03039."""
	bins = 2**bits
	assert obs.dtype == torch.float32
	if bits < 8:
		obs = torch.floor(obs / 2**(8 - bits))
	obs = obs / bins
	obs = obs + torch.rand_like(obs) / bins
	obs = obs - 0.5
	return obs


def soft_update_params(net, target_net, tau):
	for param, target_param in zip(net.parameters(), target_net.parameters()):
		target_param.data.copy_(
			tau * param.data + (1 - tau) * target_param.data
		)


def set_seed_everywhere(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


def write_info(args, fp):
	data = {
		'timestamp': str(datetime.now()),
		'args': str(args)
	}
	with open(fp, 'w') as f:
		json.dump(data, f)


def make_dir(dir_path):
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path

def center_crop_images(image, output_size):
	h, w = image.shape[2:]
	new_h, new_w = output_size, output_size

	top = (h - new_h)//2
	left = (w - new_w)//2

	image = image[:, :, top:top + new_h, left:left + new_w]
	return image


def array_init(capacity, dims, dtype):
	"""Preallocate array in memory"""
	chunks = 20
	zero_dim_size = int(capacity / chunks)
	array = np.zeros((capacity, *dims), dtype=dtype)
	temp = np.ones((zero_dim_size, *dims), dtype=dtype)

	for i in range(chunks):
		array[i*zero_dim_size:(i+1)*zero_dim_size] = temp

	return array


class ReplayBuffer(object):
	"""Buffer to store environment transitions"""
	def __init__(self, obs_shape, action_shape, capacity, batch_size):
		self.capacity = capacity
		self.batch_size = batch_size

		# self.obs = array_init(capacity, obs_shape, dtype=np.uint8)
		# self.next_obs = array_init(capacity, obs_shape, dtype=np.uint8)
		self.obs = np.empty((capacity, *obs_shape), dtype=np.uint8)
		self.next_obs = np.empty((capacity, *obs_shape), dtype=np.uint8)

		self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
		self.rewards = np.empty((capacity, 1), dtype=np.float32)
		self.not_dones = np.empty((capacity, 1), dtype=np.float32)

		self.idx = 0
		self.full = False

		self.pre_image_size = 84 # if crop in image size ; else 100
		self.image_size = 84


	def add(self, obs, action, reward, next_obs, done):
		np.copyto(self.obs[self.idx], obs)
		np.copyto(self.actions[self.idx], action)
		np.copyto(self.rewards[self.idx], reward)
		np.copyto(self.next_obs[self.idx], next_obs)
		np.copyto(self.not_dones[self.idx], not done)

		self.idx = (self.idx + 1) % self.capacity
		self.full = self.full or self.idx == 0

	def _get_idxs(self, n=None):
		if n is None:
			n = self.batch_size
		return np.random.randint(
			0, self.capacity if self.full else self.idx, size=n
		)

	def sample_soda(self, n=None):
		return torch.as_tensor(self.obs[self._get_idxs(n)]).cuda().float()

	def sample_curl(self, n=None):
		idxs = self._get_idxs(n)

		obs = torch.as_tensor(self.obs[idxs]).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		next_obs = torch.as_tensor(self.next_obs[idxs]).cuda().float()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		pos = augmentations.random_crop(obs.clone())
		obs = augmentations.random_crop(obs)
		next_obs = augmentations.random_crop(next_obs)

		return obs, actions, rewards, next_obs, not_dones, pos

	def sample_rad(self, aug_funcs, n=None):

		# augs specified as flags
		# curl_sac organizes flags into aug funcs
		# passes aug funcs into sampler

		idxs = self._get_idxs(n)

		obs = torch.as_tensor(self.obs[idxs]).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		next_obs = torch.as_tensor(self.next_obs[idxs]).cuda().float()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		obses = np.empty(self.obs[idxs].shape, dtype=np.uint8)
		np.copyto(self.obs[idxs], obses)

		obs = augmentations.random_crop(obs)
		next_obs = augmentations.random_crop(next_obs)

		if aug_funcs:
			for aug, func in aug_funcs.items():
				# apply crop and cutout first
				if 'crop' in aug or 'cutout' in aug:
					obses = func(obses)
				elif 'translate' in aug:
					og_obses = center_crop_images(obses, self.pre_image_size)
					obses, rndm_idxs = func(og_obses, self.image_size, return_random_idxs=True)

		obses = torch.as_tensor(obses).cuda().float()
		obses = obses / 255.
		# augmentations go here
		if aug_funcs:
			for aug, func in aug_funcs.items():
				# skip crop and cutout augs
				if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
					continue
				obses = func(obses)

		pos = obses * 255.

		return obs, actions, rewards, next_obs, not_dones, pos

	def sample_neg(self, n=None):
		anchor = (torch.rand((1,)) + 0.25) / 1.25   # 0.2 - 1.0

		def prob_random(p=0.5):
			if random.random() > p:
				return -1.
			else:
				return 1.

		left_right = prob_random(0.5)
		rotate_angle = torch.acos(anchor) * left_right * 180 / 3.14

		aug_funcs = {
                # 'crop':rad.random_crop,
                # 'grayscale':rad.random_grayscale,
                # 'cutout':rad.random_cutout,
                # 'cutout_color':rad.random_cutout_color,
                # 'flip':rad.random_flip,
                # 'rotate':rad.random_rotation,
                'rotate':TF.RandomRotation((rotate_angle, rotate_angle), resample=False, expand=False, fill=0),
                # 'rand_conv':rad.random_convolution,
                # 'color_jitter':rad.random_color_jitter,
                # 'translate':rad.random_translate,
                # 'no_aug':rad.no_aug,
            }

		idxs = self._get_idxs(n)

		obs = torch.as_tensor(self.obs[idxs]).cuda().float()

		obses = np.empty(self.obs[idxs].shape, dtype=np.uint8)
		np.copyto(self.obs[idxs], obses)

		obs = augmentations.random_crop(obs)

		if aug_funcs:
			for aug, func in aug_funcs.items():
				# apply crop and cutout first
				if 'crop' in aug or 'cutout' in aug:
					obses = func(obses)
				elif 'translate' in aug:
					og_obses = center_crop_images(obses, self.pre_image_size)
					obses, rndm_idxs = func(og_obses, self.image_size, return_random_idxs=True)

		obses = torch.as_tensor(obses).cuda().float()
		obses = obses / 255.
		# augmentations go here
		if aug_funcs:
			for aug, func in aug_funcs.items():
				# skip crop and cutout augs
				if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
					continue
				obses = func(obses)

		pos = obses * 255.
		pos = augmentations.random_crop(pos)
		pos = augmentations.random_overlay(pos)

		return obs, pos, anchor.cuda()

	def sample_rad_norm(self, aug_funcs, n=None):
		# augs specified as flags
		# curl_sac organizes flags into aug funcs
		# passes aug funcs into sampler
		idxs = self._get_idxs(n)

		obses = self.obs[idxs]
		next_obses = self.next_obs[idxs]

		if aug_funcs:
			for aug, func in aug_funcs.items():
				# apply crop and cutout first
				if 'crop' in aug or 'cutout' in aug:
					obses = func(obses)
					next_obses = func(next_obses)
				elif 'translate' in aug:
					og_obses = center_crop_images(obses, self.pre_image_size)
					og_next_obses = center_crop_images(next_obses, self.pre_image_size)
					obses, rndm_idxs = func(og_obses, self.image_size, return_random_idxs=True)
					next_obses = func(og_next_obses, self.image_size, **rndm_idxs)

		obses = torch.as_tensor(obses).cuda().float()
		next_obses = torch.as_tensor(next_obses).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		obses = obses / 255.
		next_obses = next_obses / 255.

		# augmentations go here
		if aug_funcs:
			for aug, func in aug_funcs.items():
				# skip crop and cutout augs
				if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
					continue
				obses = func(obses)
				next_obses = func(next_obses)

		return obses * 255., actions, rewards, next_obses * 255., not_dones

	def sample_rad_pair(self, aug_funcs, n=None):
		# augs specified as flags
		# curl_sac organizes flags into aug funcs
		# passes aug funcs into sampler
		idxs = self._get_idxs(n)
		obses = self.obs[idxs]
		poses = np.empty(self.obs[idxs].shape, dtype=np.uint8)
		np.copyto(self.obs[idxs], poses)

		if aug_funcs:
			for aug, func in aug_funcs.items():
				# apply crop and cutout first
				if 'crop' in aug or 'cutout' in aug:
					obses = func(obses)
					poses = func(poses)
				elif 'translate' in aug:
					og_obses = center_crop_images(obses, self.pre_image_size)
					og_poses = center_crop_images(poses, self.pre_image_size)
					obses, rndm_idxs = func(og_obses, self.image_size, return_random_idxs=True)
					poses = func(og_poses, self.image_size, **rndm_idxs)

		obses = torch.as_tensor(obses).cuda().float() / 255.
		poses = torch.as_tensor(poses).cuda().float() / 255.

		# augmentations go here
		if aug_funcs:
			for aug, func in aug_funcs.items():
				# skip crop and cutout augs
				if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
					continue
				obses = func(obses)
				poses = func(poses)

		return obses * 255., poses * 255.


	def sample(self, n=None):
		idxs = self._get_idxs(n)

		obs = torch.as_tensor(self.obs[idxs]).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		next_obs = torch.as_tensor(self.next_obs[idxs]).cuda().float()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()
		obs = augmentations.random_crop(obs)

		next_obs = augmentations.random_crop(next_obs)

		return obs, actions, rewards, next_obs, not_dones
