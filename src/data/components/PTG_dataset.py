
import torch
import random

import numpy as np


def move_feat_points(
        window_features, feat_version,
        w, h, num_obj_classes,
        hand_dist_delta, obj_dist_delta
    ):
    """Simulate moving the center points of the bounding boxes by
    adjusting the distances

    :param window_features: Array of size (number of frames in window x features )
    :param feat_version: Algorithm version used to generate the input features
    :param w: Width of the frames
    :param h: Height of the frames
    :param num_obj_classes: Number of object classes from the detections
        used to generate the features
    :param hand_dist_delta: Decimal percentage to calculate the +-offset in 
        pixels for the hands
    :param obj_dist_delta: Decimal percentage to calculate the +-offset in 
        pixels for the objects

    :return: ``window_features`` with the feature's distances updated
    """
    hand_delta_x = w * hand_dist_delta
    hand_delta_y = h * hand_dist_delta

    rhand_delta_x = random.uniform(-hand_delta_x, hand_delta_x)
    rhand_delta_y = random.uniform(-hand_delta_y, hand_delta_y)

    lhand_delta_x = random.uniform(-hand_delta_x, hand_delta_x)
    lhand_delta_y = random.uniform(-hand_delta_y, hand_delta_y)

    obj_ddelta_x = w * obj_dist_delta
    obj_ddelta_y = h * obj_dist_delta

    obj_delta_x = random.uniform(-obj_ddelta_x, obj_ddelta_x)
    obj_delta_y = random.uniform(-obj_ddelta_y, obj_ddelta_y)
    
    for window_i in range(window_features.shape[0]):
        features = window_features[window_i]

        if feat_version == 1:
            pass
        elif feat_version == 2:
            num_obj_feats = num_obj_classes - 2 # not including hands in count
            num_obj_points = num_obj_feats * 2

            # Distance from hand to object
            right_dist_idx2 = num_obj_points+1
            left_dist_idx1 = num_obj_points+2; left_dist_idx2 = left_dist_idx1+num_obj_points

            for delta_x, delta_y, start_idx, end_idx in zip(
                [rhand_delta_x, lhand_delta_x],
                [rhand_delta_y, lhand_delta_y],
                [1, left_dist_idx1],
                [right_dist_idx2, left_dist_idx2]
            ): 
                x_vals = features[start_idx:end_idx:2]
                features[start_idx:end_idx:2] = np.where(
                    x_vals != 0,
                    x_vals + delta_x + obj_delta_x,
                    0
                )

                y_vals = features[start_idx+1:end_idx:2]
                features[start_idx+1:end_idx:2] = np.where(
                    y_vals != 0,
                    y_vals + delta_y + obj_delta_y,
                    0
                )

            # Distance between hands
            hands_dist_idx = left_dist_idx2 + 1

            x_feat = features[hands_dist_idx]
            features[hands_dist_idx] = np.where(
                x_feat != 0,
                x_feat + rhand_delta_x + lhand_delta_x,
                0
            )

            y_feat = features[hands_dist_idx+1]
            features[hands_dist_idx+1] = np.where(
                y_feat != 0,
                y_feat + rhand_delta_y + lhand_delta_y,
                0
            )

        else:
            NotImplementedError(f"Unhandled version '{feat_version}'")
        
        window_features[window_i] = features
    return window_features

def update_activation_feats(
        window_features, feat_version,
        num_obj_classes, conf_delta
    ):
    """Update the activation feature of each class by +-``conf_delta``

    :param window_features: Array of size (number of frames in window x features )
    :param feat_version: Algorithm version used to generate the input features
    :param num_obj_classes: Number of object classes from the detections
        used to generate the features

    :return: ``window_features`` with the activation features updated
    """
    min_conf_delta = 1 - conf_delta
    max_conf_delta = 1 + conf_delta

    delta = random.uniform(min_conf_delta, max_conf_delta)

    for window_i in range(window_features.shape[0]):
        features = window_features[window_i]
        if feat_version == 1:
            features = np.where(
                features != 0,
                np.clip(features * delta, 0, 1),
                0
            )
        elif feat_version == 2:
            num_obj_feats = num_obj_classes - 2 # not including hands in count
            num_obj_points = num_obj_feats * 2

            obj_acts_idx = num_obj_points + 1 + num_obj_points + 2 + 1
            activation_idxs = [0, num_obj_points+1] + list(range(obj_acts_idx, len(features)))

            features[activation_idxs] = np.where(
                features[activation_idxs] != 0,
                np.clip(features[activation_idxs] * delta, 0, 1),
                0
            ) 

        else:
            NotImplementedError(f"Unhandled version '{feat_version}'")
        
        window_features[window_i] = features
    return window_features

def normalize_feat_pts(
        window_features, feat_version,
        w, h, num_obj_classes
    ):
    """Normalize the distances from -1 to 1 with respect to the image size

    :param window_features: Array of size (number of frames in window x features )
    :param feat_version: Algorithm version used to generate the input features
    :param w: Width of the frames
    :param h: Height of the frames
    :param num_obj_classes: Number of object classes from the detections
        used to generate the features

    :return: ``window_features`` with the pixel values normalized from 0-1 
    """
    for window_i in range(window_features.shape[0]):
        features = window_features[window_i]

        if feat_version == 1:
            pass
        elif feat_version == 2:
            num_obj_feats = num_obj_classes - 2 # not including hands in count
            num_obj_points = num_obj_feats * 2

            # Distance from hand to object
            right_dist_idx2 = num_obj_points+1
            left_dist_idx1 = num_obj_points+2; left_dist_idx2 = left_dist_idx1+num_obj_points

            for start_idx, end_idx in zip(
                [1, left_dist_idx1],
                [right_dist_idx2, left_dist_idx2]
            ):    
                x_vals = features[start_idx:end_idx:2]
                features[start_idx:end_idx:2] = x_vals / w

                y_vals = features[start_idx+1:end_idx:2]
                features[start_idx+1:end_idx:2] = y_vals / h

            # Distance between hands
            hands_dist_idx = left_dist_idx2 + 1

            x_feat = features[hands_dist_idx]
            features[hands_dist_idx] = x_feat / w

            y_feat = features[hands_dist_idx+1]
            features[hands_dist_idx+1] = y_feat / h

        else:
            NotImplementedError(f"Unhandled version '{feat_version}'")
        
        window_features[window_i] = features
    return window_features


class PTG_Dataset(torch.utils.data.Dataset):
    def __init__(self, videos, num_classes, actions_dict, gt_path, features_path, sample_rate, window_size,
                 im_w, im_h, num_obj_classes, feat_version, hand_dist_delta, obj_dist_delta, conf_delta,
                 augmentation=False, transform=None, target_transform=None):
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.dataset_size = -1
        self.augmentation = augmentation

        self.im_w = im_w
        self.im_h = im_h
        self.feat_version = feat_version
        self.num_obj_classes = num_obj_classes
        self.hand_dist_delta = hand_dist_delta
        self.obj_dist_delta = obj_dist_delta
        self.conf_delta = conf_delta

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
        if self.augmentation:
            features = move_feat_points(
                features, self.feat_version,
                self.im_w, self.im_h, self.num_obj_classes,
                self.hand_dist_delta, self.obj_dist_delta
            )
            #features = update_activation_feats(
            #    features, self.feat_version, 
            #    self.num_obj_classes, 
            #    self.conf_delta
            #)
            #features = normalize_feat_pts(
            #    features, self.feat_version,
            #    self.im_w, self.im_h, self.num_obj_classes
            #)
        
        return features, target, mask
