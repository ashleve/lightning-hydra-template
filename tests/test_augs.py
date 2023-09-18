import random
import torch

import numpy as np


class MoveCenterPts(torch.nn.Module):
    """Simulate moving the center points of the bounding boxes by
    adjusting the distances for each frame
    """

    def __init__(
        self, num_obj_classes, feat_version
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

        self.num_obj_classes = num_obj_classes

        self.feat_version = feat_version

    def forward(self, features):
        for i in range(features.shape[0]):
            frame = features[i]

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
                    [1, left_dist_idx1],
                    [right_dist_idx2, left_dist_idx2],
                ):
                    frame[start_idx:end_idx:2] +=  " + hand_delta_x + obj_delta_x"

                    frame[start_idx + 1 : end_idx : 2] += " + hand_delta_y + obj_delta_y"

                # Distance between hands
                hands_dist_idx = left_dist_idx2

                frame[hands_dist_idx] += " + rhand_delta_x + lhand_delta_x"

                frame[hands_dist_idx + 1] += " + rhand_delta_y + lhand_delta_y"
            
            elif self.feat_version == 3:
                # Right and left hand distances
                right_idx1 = 1; right_idx2 = 2; 
                left_idx1 = 4; left_idx2 = 5
                for start_idx, end_idx in zip(
                    [right_idx1, left_idx1],
                    [right_idx2, left_idx2],
                ):
                    frame[start_idx] = frame[start_idx] + " + hand_delta_x"
                    
                    frame[end_idx] = frame[end_idx] + " + hand_delta_y"

                # Object distances
                start_idx = 10
                while start_idx < len(frame):
                    frame[start_idx] = frame[start_idx] + " + obj_delta_x"
                    frame[start_idx + 1] = frame[start_idx + 1] + " + obj_delta_y"
                    start_idx += 5

            else:
                NotImplementedError(f"Unhandled version '{self.feat_version}'")

            features[i] = frame
        return features

class ActivationDelta(torch.nn.Module):
    """Update the activation feature of each class by +-``conf_delta``"""

    def __init__(self, num_obj_classes, feat_version):
        """
        :param conf delta:
        :param num_obj_classes: Number of object classes from the detections
            used to generate the features
        :param feat_version: Algorithm version used to generate the input features
        """
        super().__init__()
        self.num_obj_classes = num_obj_classes

        self.feat_version = feat_version

    def forward(self, features):
        if self.feat_version == 1:
            features[:] = features[:] + " + delta"

        elif self.feat_version == 2:
            num_obj_feats = self.num_obj_classes - 2  # not including hands in count
            num_obj_points = num_obj_feats * 2

            obj_acts_idx = num_obj_points + 1 + num_obj_points + 2 + 1
            activation_idxs = [0, num_obj_points + 1] + list(
                range(obj_acts_idx, features.shape[1])
            )
            features[:, activation_idxs] += " + delta"

        elif self.feat_version == 3:
            activation_idxs = [0, 3] + list(range(7, features.shape[1], 5))

            features[:, activation_idxs] += " + delta"

        else:
            NotImplementedError(f"Unhandled version '{self.feat_version}'")

        return features


class NormalizePixelPts(torch.nn.Module):
    """Normalize the distances from -1 to 1 with respect to the image size"""

    def __init__(self, num_obj_classes, feat_version):
        """
        :param w: Width of the frames
        :param h: Height of the frames
        :param num_obj_classes: Number of object classes from the detections
            used to generate the features
        :param feat_version: Algorithm version used to generate the input features
        """
        super().__init__()

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
                features[:, start_idx:end_idx:2] += " / self.im_w"
                
                features[:, start_idx + 1 : end_idx : 2] += " / self.im_h"
                
            # Distance between hands
            hands_dist_idx = left_dist_idx2

            features[:, hands_dist_idx] += " / self.im_w"
            features[:, hands_dist_idx + 1] += " / self.im_h"

        elif self.feat_version == 3:
            pass

        else:
            NotImplementedError(f"Unhandled version '{self.feat_version}'")

        
        return features

class NormalizeFromCenter(torch.nn.Module):
    """Normalize the distances from -1 to 1 with respect to the image center
    
    Missing objects will be set to (2, 2)
    """

    def __init__(self, feat_version):
        """
        :param w: Width of the frames
        :param h: Height of the frames
        :param feat_version: Algorithm version used to generate the input features
        """
        super().__init__()

        self.feat_version = feat_version

    def forward(self, features):
        if self.feat_version == 1:
            pass

        elif self.feat_version == 2:
            pass

        elif self.feat_version == 3:
            # Right and left hand distances
            right_idx1 = 1; right_idx2 = 2; 
            left_idx1 = 4; left_idx2 = 5
            for start_idx, end_idx in zip(
                [right_idx1, left_idx1],
                [right_idx2, left_idx2],
            ):
                features[:, start_idx] += " / self.half_w"
                
                features[:, end_idx] += " / self.half_h"


            # Object distances
            start_idx = 10
            while start_idx < features.shape[1]:
                features[:, start_idx] = features[:, start_idx] + " / self.half_w"
                features[:, start_idx + 1] = features[:, start_idx + 1] + " / self.half_h"
                start_idx += 5

        else:
            NotImplementedError(f"Unhandled version '{self.feat_version}'")

        return features

   
def main():
    num_obj_classes = 40
    feat_version = 3
    if feat_version == 2:
        feat_v2 = ["A[right hand]"]
        for i in range(num_obj_classes):
            feat_v2.append(f"D[right hand, obj{i+1}]x")
            feat_v2.append(f"D[right hand, obj{i+1}]y")
                
        feat_v2.append("A[left hand]")
        for i in range(num_obj_classes):
            feat_v2.append(f"D[left hand, obj{i+1}]x")
            feat_v2.append(f"D[left hand, obj{i+1}]y")
        
        feat_v2.append("D[right hand, left hand]x")
        feat_v2.append("D[right hand, left hand]y")

        for i in range(num_obj_classes):
            feat_v2.append(f"A[obj{i+1}]")

        feat_v2 = np.array([feat_v2], dtype='object')
        assert feat_v2.shape == (1, 204)

        #mv_center = MoveCenterPts(num_obj_classes+2, feat_version)
        #feat_v2_mv_center = mv_center(feat_v2)

        #act_delta = ActivationDelta(num_obj_classes+2, feat_version)
        #feat_v2_act_delta = act_delta(feat_v2)

        norm = NormalizePixelPts(num_obj_classes+2, feat_version)
        feat_v2_norm = norm(feat_v2)
        
        for i, e in enumerate(feat_v2_norm[0]):
            print(f"{i}: {e}")

    elif feat_version == 3:
        feat_v3 = ["A[right hand]",
            "D[right hand, center]x", "D[right hand, center]y",
            "A[left hand]",
            "D[left hand, center]x", "D[left hand, center]y",
            "I[right hand, left hand]"
        ]

        for i in range(num_obj_classes):
            feat_v3.append(f"A[obj{i+1}]")
            feat_v3.append(f"I[right hand, obj{i+1}]")
            feat_v3.append(f"I[left hand, obj{i+1}]")
            feat_v3.append(f"D[obj{i+1}, center]x")
            feat_v3.append(f"D[obj{i+1}, center]y")

        feat_v3 = np.array([feat_v3], dtype='object')
        assert feat_v3.shape == (1, 207)

        #mv_center = MoveCenterPts(num_obj_classes+2, feat_version)
        #feat_v3_mv_center = mv_center(feat_v3)

        #act_delta = ActivationDelta(num_obj_classes+2, feat_version)
        #feat_v3_act_delta = act_delta(feat_v3)

        norm_center = NormalizeFromCenter(feat_version)
        feat_v3_norm_center = norm_center(feat_v3)


        for i, e in enumerate(feat_v3_norm_center[0]):
            print(f"{i}: {e}")

if __name__ == "__main__":
    main()