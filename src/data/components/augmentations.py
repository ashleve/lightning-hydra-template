import random
import torch

import numpy as np


class MoveCenterPts(torch.nn.Module):
    """Simulate moving the center points of the bounding boxes by
    adjusting the distances
    """
    def __init__(
        self, hand_dist_delta, obj_dist_delta,
        im_w, im_h, num_obj_classes, feat_version
    ):
        """
        :param hand_dist_delta: Decimal percentage to calculate the +-offset in 
            pixels for the hands
        :param obj_dist_delta: Decimal percentage to calculate the +-offset in 
            pixels for the objects
        :param w: Width of the frames
        :param h: Height of the frames
        :param num_obj_classes: Number of object classes from the detections
            used to generate the features
        :param feat_version: Algorithm version used to generate the input features
        """
        super().__init__()

        self.hand_dist_delta = hand_dist_delta
        self.obj_dist_delta = obj_dist_delta

        self.im_w = im_w
        self.im_h = im_h
        self.num_obj_classes = num_obj_classes

        self.feat_version = feat_version

        # Deltas
        self.hand_delta_x = self.im_w * self.hand_dist_delta
        self.hand_delta_y = self.im_h * self.hand_dist_delta

        self.obj_ddelta_x = self.im_w * self.obj_dist_delta
        self.obj_ddelta_y = self.im_h * self.obj_dist_delta

    def init_deltas(self):
        rhand_delta_x = random.uniform(-self.hand_delta_x, self.hand_delta_x)
        rhand_delta_y = random.uniform(-self.hand_delta_y, self.hand_delta_y)

        lhand_delta_x = random.uniform(-self.hand_delta_x, self.hand_delta_x)
        lhand_delta_y = random.uniform(-self.hand_delta_y, self.hand_delta_y)

        obj_delta_x = random.uniform(-self.obj_ddelta_x, self.obj_ddelta_x)
        obj_delta_y = random.uniform(-self.obj_ddelta_y, self.obj_ddelta_y)

        return [rhand_delta_x, rhand_delta_y], [lhand_delta_x, lhand_delta_y], [obj_delta_x, obj_delta_y]

    def forward(self, window):
        ( [rhand_delta_x, rhand_delta_y],
          [lhand_delta_x, lhand_delta_y],
          [obj_delta_x, obj_delta_y] ) = self.init_deltas()

        if self.feat_version == 1:
            pass
        
        elif self.feat_version == 2:
            num_obj_feats = self.num_obj_classes - 2 # not including hands in count
            num_obj_points = num_obj_feats * 2

            # Distance from hand to object
            right_dist_idx2 = num_obj_points+1
            left_dist_idx1 = num_obj_points+2; left_dist_idx2 = left_dist_idx1+num_obj_points

            for hand_delta_x, hand_delta_y, start_idx, end_idx in zip(
                [rhand_delta_x, lhand_delta_x],
                [rhand_delta_y, lhand_delta_y],
                [1, left_dist_idx1],
                [right_dist_idx2, left_dist_idx2]
            ):
                window[:, start_idx:end_idx:2] = np.where(
                    window[:, start_idx:end_idx:2] != 0,
                    window[:, start_idx:end_idx:2] + hand_delta_x + obj_delta_x,
                    window[:, start_idx:end_idx:2]
                )

                window[:, start_idx+1:end_idx:2] = np.where(
                    window[:, start_idx+1:end_idx:2] != 0,
                    window[:, start_idx+1:end_idx:2] + hand_delta_y + obj_delta_y,
                    window[:, start_idx+1:end_idx:2]
                )

            # Distance between hands
            hands_dist_idx = left_dist_idx2 + 1

            window[:, hands_dist_idx] = np.where(
                window[:, hands_dist_idx] != 0,
                window[:, hands_dist_idx] + rhand_delta_x + lhand_delta_x,
                window[:, hands_dist_idx]
            )

            window[:, hands_dist_idx+1] = np.where(
                window[:, hands_dist_idx+1] != 0,
                window[:, hands_dist_idx+1] + rhand_delta_y + lhand_delta_y,
                window[:, hands_dist_idx+1]
            )

        else:
            NotImplementedError(f"Unhandled version '{self.feat_version}'")

        return window

    def __repr__(self) -> str:
        detail = f"(hand_dist_delta={self.hand_dist_delta}, obj_dist_delta={self.obj_dist_delta}, im_w={self.im_w}, im_h={self.im_h}, num_obj_classes={self.num_obj_classes}, feat_version={self.feat_version})"
        return f"{self.__class__.__name__}{detail}"

class ActivationDelta(torch.nn.Module):
    """Update the activation feature of each class by +-``conf_delta``
    """
    def __init__(self, conf_delta, num_obj_classes, feat_version):
        """
        :param conf delta:
        :param num_obj_classes: Number of object classes from the detections
            used to generate the features
        :param feat_version: Algorithm version used to generate the input features
        """
        super().__init__()

        self.conf_delta = conf_delta

        self.num_obj_classes = num_obj_classes

        self.feat_version = feat_version

    def init_delta(self):
        min_conf_delta = 1 - self.conf_delta
        max_conf_delta = 1 + self.conf_delta

        delta = random.uniform(min_conf_delta, max_conf_delta)

        return delta

    def forward(self, window):
        delta = self.init_delta()
        
        if self.feat_version == 1:
            window[:] = np.where(
                window[:] != 0,
                np.clip(window[:] + delta, 0, 1),
                window[:]
            )

        elif self.feat_version == 2:
            num_obj_feats = self.num_obj_classes - 2 # not including hands in count
            num_obj_points = num_obj_feats * 2

            obj_acts_idx = num_obj_points + 1 + num_obj_points + 2 + 1
            activation_idxs = [0, num_obj_points+1] + list(range(obj_acts_idx, len(window)))

            window[:, activation_idxs] = np.where(
                window[:, activation_idxs] != 0,
                np.clip(window[:, activation_idxs] * delta, 0, 1),
                window[:, activation_idxs]
            ) 

        else:
            NotImplementedError(f"Unhandled version '{self.feat_version}'")

        return window

    def __repr__(self) -> str:
        detail = f"(conf_delta={self.conf_delta}, num_obj_classes={self.num_obj_classes}, feat_version={self.feat_version})"
        return f"{self.__class__.__name__}{detail}"

class NormalizePixelPts(torch.nn.Module):
    """Normalize the distances from -1 to 1 with respect to the image size
    """
    def __init__(self, im_w, im_h, num_obj_classes, feat_version):
        """
        :param w: Width of the frames
        :param h: Height of the frames
        :param num_obj_classes: Number of object classes from the detections
            used to generate the features
        :param feat_version: Algorithm version used to generate the input features
        """
        super().__init__()

        self.im_w = im_w
        self.im_h = im_h
        self.num_obj_classes = num_obj_classes

        self.feat_version = feat_version

    def forward(self, window):
        if self.feat_version == 1:
            pass 
        
        elif self.feat_version == 2:
            num_obj_feats = self.num_obj_classes - 2 # not including hands in count
            num_obj_points = num_obj_feats * 2

            # Distance from hand to object
            right_dist_idx2 = num_obj_points+1
            left_dist_idx1 = num_obj_points+2; left_dist_idx2 = left_dist_idx1+num_obj_points

            for start_idx, end_idx in zip(
                [1, left_dist_idx1],
                [right_dist_idx2, left_dist_idx2]
            ):    
                window[:, start_idx:end_idx:2] = window[:, start_idx:end_idx:2] / self.im_w
                window[:, start_idx+1:end_idx:2] = window[:, start_idx+1:end_idx:2] / self.im_h

            # Distance between hands
            hands_dist_idx = left_dist_idx2 + 1

            window[:, hands_dist_idx] = window[:, hands_dist_idx] / self.im_w
            window[:, hands_dist_idx+1] = window[:, hands_dist_idx+1] / self.im_h

        else:
            NotImplementedError(f"Unhandled version '{self.feat_version}'")

        return window

    def __repr__(self) -> str:
        detail = f"(im_w={self.im_w}, im_h={self.im_h}, num_obj_classes={self.num_obj_classes}, feat_version={self.feat_version})"
        return f"{self.__class__.__name__}{detail}"
