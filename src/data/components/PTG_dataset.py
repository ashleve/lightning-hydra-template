
import torch

import numpy as np

from typing import Optional, Callable, Dict, List

class PTG_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        videos: List[str], 
        num_classes: int,
        actions_dict: Dict[str, int],
        gt_path: str, 
        features_path: str,
        sample_rate: int,
        window_size: int,
        transform: Optional[Callable]=None,
        target_transform: Optional[Callable]=None
    ):
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_size = -1
        self.norm_stats = dict()

        input_frames_list = []
        target_frames_list = []
        mask_frames_list = []
        for vid in videos:
            features = np.load(self.features_path + vid.split(".")[0] + ".npy")
            
            file_ptr = open(self.gt_path + vid, "r")
            content = file_ptr.read().split("\n")[:-1]

            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]

            # mask out the end of the window size of the end of the sequence to prevent overlap between videos.
            mask = np.ones_like(classes)
            mask[-window_size:] = 0


            input_frames_list.append(features[:, :: self.sample_rate])
            target_frames_list.append(classes[:: self.sample_rate])
            mask_frames_list.append(mask[:: self.sample_rate])


        self.feature_frames = np.concatenate(input_frames_list, axis=1, dtype=np.single).transpose()
        self.target_frames = np.concatenate(target_frames_list, axis=0, dtype=int, casting='unsafe')
        self.mask_frames = np.concatenate(mask_frames_list, axis=0, dtype=int, casting='unsafe')

        self.norm_stats['mean'] = self.feature_frames.mean(axis=0)
        self.norm_stats['std'] = self.feature_frames.std(axis=0)
        self.norm_stats['max'] = self.feature_frames.max(axis=0)
        self.norm_stats['min'] = self.feature_frames.min(axis=0)

        self.dataset_size = self.target_frames.shape[0] - self.window_size

        # Get weights for sampler by inverse count.  
        # Weights represent the GT of the final frame of a window starting from idx
        class_name, counts = np.unique(self.target_frames, return_counts=True)
        class_weights =  1. / counts
        class_lookup = dict()
        for i, cn in enumerate(class_name):
            class_lookup[cn] = class_weights[i]
        self.weights = np.zeros((self.dataset_size))
        for i in range(self.dataset_size):
            self.weights[i] = class_lookup[self.target_frames[i+self.window_size]]
        # Set weights to 0 for frames before the window length
        # So they don't get picked
        self.weights[:self.window_size] = 0

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """Grab a window of frames starting at ``idx``

        :param idx: The first index of the time window

        :return: features, targets, and mask of the windw
        """
        features = self.feature_frames[idx:idx+self.window_size,:]
        target = self.target_frames[idx:idx+self.window_size]
        mask = self.mask_frames[idx:idx+self.window_size]

        # Transforms/Augmentations
        if self.transform is not None:
            features = self.transform(features)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return features, target, mask
