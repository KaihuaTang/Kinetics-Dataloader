#!/usr/bin/env python3
# Modified from https://github.com/facebookresearch/SlowFast

import os
import csv
import random
import torch
import torch.utils.data
from torchvision import transforms

import utils_decoder as decoder
import utils_general as utils


class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, logger, num_retries=3):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in ["train", "val", "test"], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.logger = logger

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = cfg['train']['num_ensemble_views'] * cfg['train']['num_spatial_crops']
        elif self.mode in ["test"]:
            self._num_clips = cfg['test']['num_ensemble_views'] * cfg['test']['num_spatial_crops']

        self.logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()


    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(self.cfg['data']['path_to_anno_dir'], "output_{}.csv".format(self.mode))
        assert os.path.isfile(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(list(csv.reader(f))):
                assert (len(path_label) == 2)

                path, label = path_label

                # sample _num_clips for each video
                for idx in range(self._num_clips):
                    self._path_to_videos.append(path)
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}

        assert (len(self._path_to_videos) > 0), "Failed to load Kinetics split {} from {}".format(self.mode, path_to_file)
        self.logger.info("Constructing kinetics dataloader (size: {}) from {}".format(len(self._path_to_videos), path_to_file))


    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg['data']['train_scales'][0]
            max_scale = self.cfg['data']['train_scales'][1]
            crop_size = self.cfg['data']['train_crop_size']

        elif self.mode in ["test"]:
            # temporal_sample_index
            temporal_sample_index = self._spatial_temporal_idx[index] // self.cfg['test']['num_spatial_crops']

            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            if self.cfg['test']['num_spatial_crops'] > 1:
                spatial_sample_index = self._spatial_temporal_idx[index] % self.cfg['test']['num_spatial_crops']
            else:
                spatial_sample_index = 1
            
            # frame scales
            if self.cfg['test']['num_spatial_crops'] > 1:
                min_scale, max_scale, crop_size = [self.cfg['data']['test_crop_size']] * 3    
            else:
                min_scale, max_scale, crop_size = [self.cfg['data']['train_scales'][0]] * 2 + [self.cfg['data']['test_crop_size']]

            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        sampling_rate = self.cfg['data']['sampling_rate']

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            video_container = None
            try:
                video_container = utils.get_video_container(
                    self._path_to_videos[index],
                    backend = self.cfg['data']['decoding_backend'],
                )
            except Exception as e:
                self.logger.info("Failed to load video from {} with error {}".format(self._path_to_videos[index], e))

            # Select a random video if the current video was not able to access.
            if video_container is None:
                self.logger.warning("Failed to meta load video idx {} from {}; trial {}".format(index, self._path_to_videos[index], i_try))
                
                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            frames = decoder.decode(
                video_container,
                sampling_rate,
                self.cfg['data']['num_frames'],
                temporal_sample_index,
                self.cfg['test']['num_ensemble_views'],
                video_meta=self._video_meta[index],
                target_fps=self.cfg['data']['target_fps'],
                backend=self.cfg['data']['decoding_backend'],
                max_spatial_scale=min_scale,
                use_offset=self.cfg['data']['use_offset_sampling'],
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                self.logger.warning("Failed to decode video idx {} from {}; trial {}".format(index, self._path_to_videos[index], i_try))

                if self.mode not in ["test"] and i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue


            frames = utils.tensor_normalize(frames, self.cfg['data']['mean'], self.cfg['data']['std'])
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation.
            frames = utils.spatial_sampling(
                    frames,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg['data']['random_flip'],
                )

            label = self._labels[index]
            frames = [frames]
            return frames, label, index, {}
        else:
            raise RuntimeError("Failed to fetch video after {} retries.".format(self._num_retries))


    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos


    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)