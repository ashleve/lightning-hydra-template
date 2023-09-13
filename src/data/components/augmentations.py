import random
import torch

import numpy as np


class MoveCenterPts(torch.nn.Module):
    """Simulate moving the center points of the bounding boxes by
    adjusting the distances for each frame
    """

    def __init__(
        self, hand_dist_delta, obj_dist_delta, window_size, im_w, im_h, num_obj_classes, feat_version
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
        self.window_size = window_size
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

        return (
            [rhand_delta_x, rhand_delta_y],
            [lhand_delta_x, lhand_delta_y],
            [obj_delta_x, obj_delta_y],
        )

    def forward(self, features):
        for i in range(features.shape[0]):
            frame = features[i]

            (
                [rhand_delta_x, rhand_delta_y],
                [lhand_delta_x, lhand_delta_y],
                [obj_delta_x, obj_delta_y],
            ) = self.init_deltas()

            if self.feat_version == 1:
                pass

            elif self.feat_version == 2:
                num_obj_feats = self.num_obj_classes - 2  # not including hands in count
                num_obj_points = num_obj_feats * 2

                # Distance from hand to object
                right_dist_idx2 = num_obj_points + 1
                left_dist_idx1 = num_obj_points + 2
                left_dist_idx2 = left_dist_idx1 + num_obj_points

                for hand_delta_x, hand_delta_y, start_idx, end_idx in zip(
                    [rhand_delta_x, lhand_delta_x],
                    [rhand_delta_y, lhand_delta_y],
                    [1, left_dist_idx1],
                    [right_dist_idx2, left_dist_idx2],
                ):
                    frame[start_idx:end_idx:2] = np.where(
                        frame[start_idx:end_idx:2] != 0,
                        frame[start_idx:end_idx:2] + hand_delta_x + obj_delta_x,
                        frame[start_idx:end_idx:2],
                    )

                    frame[start_idx + 1 : end_idx : 2] = np.where(
                        frame[start_idx + 1 : end_idx : 2] != 0,
                        frame[start_idx + 1 : end_idx : 2] + hand_delta_y + obj_delta_y,
                        frame[start_idx + 1 : end_idx : 2],
                    )

                # Distance between hands
                hands_dist_idx = left_dist_idx2 + 1

                frame[hands_dist_idx] = np.where(
                    frame[hands_dist_idx] != 0,
                    frame[hands_dist_idx] + rhand_delta_x + lhand_delta_x,
                    frame[hands_dist_idx],
                )

                frame[hands_dist_idx + 1] = np.where(
                    frame[hands_dist_idx + 1] != 0,
                    frame[hands_dist_idx + 1] + rhand_delta_y + lhand_delta_y,
                    frame[hands_dist_idx + 1],
                )

            elif self.feat_version == 3:
                # Right and left hand distances
                right_idx1 = 3; right_idx2 = 5; 
                left_idx1 = 5; left_idx2 = 7
                for hand_delta_x, hand_delta_y, start_idx, end_idx in zip(
                    [rhand_delta_x, lhand_delta_x],
                    [rhand_delta_y, lhand_delta_y],
                    [right_idx1, left_idx1],
                    [right_idx2, left_idx2],
                ):
                    frame[start_idx:end_idx:2] = (
                        frame[start_idx:end_idx:2] + hand_delta_x
                    )
                    
                    frame[start_idx + 1 : end_idx : 2] = (
                        frame[start_idx + 1 : end_idx : 2] + hand_delta_y
                    )

                # Object distances
                start_idx = 10
                while start_idx < len(frame):
                    frame[start_idx] = frame[start_idx] + obj_delta_x
                    frame[start_idx + 1] = frame[start_idx + 1] + obj_delta_y
                    start_idx += 5

            else:
                NotImplementedError(f"Unhandled version '{self.feat_version}'")

            features[i] = frame
        return features

    def __repr__(self) -> str:
        detail = f"(hand_dist_delta={self.hand_dist_delta}, obj_dist_delta={self.obj_dist_delta}, im_w={self.im_w}, im_h={self.im_h}, num_obj_classes={self.num_obj_classes}, feat_version={self.feat_version})"
        return f"{self.__class__.__name__}{detail}"


class ActivationDelta(torch.nn.Module):
    """Update the activation feature of each class by +-``conf_delta``"""

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
        delta = random.uniform(-self.conf_delta, self.conf_delta)

        return delta

    def forward(self, features):
        delta = self.init_delta()

        if self.feat_version == 1:
            features[:] = np.where(
                features[:] != 0, np.clip(features[:] + delta, 0, 1), features[:]
            )

        elif self.feat_version == 2:
            num_obj_feats = self.num_obj_classes - 2  # not including hands in count
            num_obj_points = num_obj_feats * 2

            obj_acts_idx = num_obj_points + 1 + num_obj_points + 2 + 1
            activation_idxs = [0, num_obj_points + 1] + list(
                range(obj_acts_idx, len(features))
            )

            features[:, activation_idxs] = np.where(
                features[:, activation_idxs] != 0,
                np.clip(features[:, activation_idxs] + delta, 0, 1),
                features[:, activation_idxs],
            )

        else:
            NotImplementedError(f"Unhandled version '{self.feat_version}'")

        return features

    def __repr__(self) -> str:
        detail = f"(conf_delta={self.conf_delta}, num_obj_classes={self.num_obj_classes}, feat_version={self.feat_version})"
        return f"{self.__class__.__name__}{detail}"


class NormalizePixelPts(torch.nn.Module):
    """Normalize the distances from -1 to 1 with respect to the image size"""

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

    def forward(self, features):
        if self.feat_version == 1:
            pass

        elif self.feat_version == 2:
            num_obj_feats = self.num_obj_classes - 2  # not including hands in count
            num_obj_points = num_obj_feats * 2

            # Distance from hand to object
            right_dist_idx2 = num_obj_points + 1
            left_dist_idx1 = num_obj_points + 2
            left_dist_idx2 = left_dist_idx1 + num_obj_points

            for start_idx, end_idx in zip(
                [1, left_dist_idx1], [right_dist_idx2, left_dist_idx2]
            ):
                features[:, start_idx:end_idx:2] = (
                    features[:, start_idx:end_idx:2] / self.im_w
                )
                features[:, start_idx + 1 : end_idx : 2] = (
                    features[:, start_idx + 1 : end_idx : 2] / self.im_h
                )

            # Distance between hands
            hands_dist_idx = left_dist_idx2 + 1

            features[:, hands_dist_idx] = features[:, hands_dist_idx] / self.im_w
            features[:, hands_dist_idx + 1] = features[:, hands_dist_idx + 1] / self.im_h

        elif self.feat_version == 3:
            pass

        else:
            NotImplementedError(f"Unhandled version '{self.feat_version}'")

        
        return features

    def __repr__(self) -> str:
        detail = f"(im_w={self.im_w}, im_h={self.im_h}, num_obj_classes={self.num_obj_classes}, feat_version={self.feat_version})"
        return f"{self.__class__.__name__}{detail}"

class NormalizeFromCenter(torch.nn.Module):
    """Normalize the distances from -1 to 1 with respect to the image center
    
    Missing objects will be set to (2, 2)
    """

    def __init__(self, im_w, im_h, feat_version):
        """
        :param w: Width of the frames
        :param h: Height of the frames
        :param feat_version: Algorithm version used to generate the input features
        """
        super().__init__()

        self.im_w = im_w
        self.half_w = im_w / 2
        self.im_h = im_h
        self.half_h = im_h / 2

        self.feat_version = feat_version

    def forward(self, features):
        if self.feat_version == 1:
            pass

        elif self.feat_version == 2:
            pass

        elif self.feat_version == 3:
            # Right and left hand distances
            start_idx = 3; end_idx = 7
            features[:, start_idx:end_idx:2] = (
                features[:, start_idx:end_idx:2] / self.half_w
            )
            
            features[:, start_idx + 1 : end_idx : 2] = (
                features[:, start_idx + 1 : end_idx : 2] / self.half_h
            )

            # Object distances
            start_idx = 10
            while start_idx < features.shape[1]:
                features[:, start_idx] = features[:, start_idx] / self.half_w
                features[:, start_idx + 1] = features[:, start_idx + 1] / self.half_h
                start_idx += 5

        else:
            NotImplementedError(f"Unhandled version '{self.feat_version}'")

        return features

    def __repr__(self) -> str:
        detail = f"(im_w={self.im_w}, im_h={self.im_h}, feat_version={self.feat_version})"
        return f"{self.__class__.__name__}{detail}"
   