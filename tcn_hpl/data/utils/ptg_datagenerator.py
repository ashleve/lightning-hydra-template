import os
import yaml
import glob
import warnings
import kwcoco

import numpy as np
import ubelt as ub

from angel_system.data.common.load_data import (
    activities_from_dive_csv,
    objs_as_dataframe,
    time_from_name,
    sanitize_str,
)
from angel_system.activity_hmm.train_activity_classifier import (
    data_loader,
    compute_feats,
)

#####################
# Inputs
#####################
ptg_root = "/home/local/KHQ/hannah.defazio/angel_system/"
activity_config_path = f"{ptg_root}/config/activity_labels"
recipe = "coffee"
activity_config_fn = f"{activity_config_path}/recipe_{recipe}.yaml"

data_dir = "/data/users/hannah.defazio/ptg_nas/data_copy/"
extracted_data_dir = f"{data_dir}/coffee_extracted"
activity_gt_dir = f"{data_dir}/coffee_labels/Labels"

obj_exp_name = "coffee_base"
obj_dets_dir = f"/data/PTG/cooking/annotations/coffee/results/{obj_exp_name}"

training_split = {
    "train_activity": [
        f"all_activities_{x}"
        for x in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 25, 26, 27,
            28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 40, 47, 48, 49]
    ],
    "val": [f"all_activities_{x}" for x in [23, 24, 42, 46]],
    "test": [f"all_activities_{x}" for x in [20, 33, 39, 50, 51, 52, 53, 54]],
}  # Coffee specific

feat_version = 4

#####################
# Output
#####################
exp_name = f"coffee_conf_10_all_hands_feat_v{str(feat_version)}"
output_data_dir = f"{data_dir}/TCN_data/{exp_name}"
if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir)

gt_dir = f"{output_data_dir}/groundTruth"
if not os.path.exists(gt_dir):
    os.makedirs(gt_dir)

bundle_dir = f"{output_data_dir}/splits"
if not os.path.exists(bundle_dir):
    os.makedirs(bundle_dir)
# Clear out the bundles
filelist = [f for f in os.listdir(bundle_dir)]
for f in filelist:
    os.remove(os.path.join(bundle_dir, f))

features_dir = f"{output_data_dir}/features"
if not os.path.exists(features_dir):
    os.makedirs(features_dir)

#####################
# Mapping
#####################
with open(activity_config_fn, "r") as stream:
    activity_config = yaml.safe_load(stream)
activity_labels = activity_config["labels"]

with open(f"{output_data_dir}/mapping.txt", "w") as mapping:
    for label in activity_labels:
        i = label["id"]
        label_str = label["label"]
        if label_str == "done":
            continue
        mapping.write(f"{i} {label_str}\n")

#####################
# Features,
# groundtruth and
# bundles
#####################
for split in training_split.keys():
    kwcoco_file = f"{obj_dets_dir}/{obj_exp_name}_results_{split}_conf_0.1_plus_hl_hands.mscoco.json"
    dset = kwcoco.CocoDataset(kwcoco_file)

    num_classes = len(dset.cats)

    for video_id in ub.ProgIter(
        dset.index.videos.keys(), desc=f"Creating features for videos in {split}"
    ):
        video = dset.index.videos[video_id]
        video_name = video["name"]

        activity_gt_fn = f"{activity_gt_dir}/{video_name}.csv"
        gt = activities_from_dive_csv(activity_gt_fn)
        gt = objs_as_dataframe(gt)

        image_ids = dset.index.vidid_to_gids[video_id]
        num_images = len(image_ids)

        video_dset = dset.subset(gids=image_ids, copy=True)

        # features
        (
            act_map,
            inv_act_map,
            image_activity_gt,
            image_id_to_dataset,
            label_to_ind,
            act_id_to_str,
            ann_by_image,
        ) = data_loader(video_dset, activity_config)
        X, y = compute_feats(
            act_map,
            image_activity_gt,
            image_id_to_dataset,
            label_to_ind,
            act_id_to_str,
            ann_by_image,
            feat_version=feat_version
        )

        X = X.T

        np.save(f"{features_dir}/{video_name}.npy", X)

        # groundtruth
        with open(f"{gt_dir}/{video_name}.txt", "w") as gt_f:
            for image_id in image_ids:
                image = dset.imgs[image_id]
                image_n = image["file_name"]

                frame_idx, time = time_from_name(image_n)
                matching_gt = gt.loc[(gt["start"] <= time) & (gt["end"] >= time)]

                if matching_gt.empty:
                    label = "background"
                    activity_label = label
                else:
                    label = matching_gt.iloc[0]["class_label"]
                    activity = [
                        x
                        for x in activity_labels[1:-1]
                        if sanitize_str(x["full_str"]) == label
                    ]
                    if not activity:
                        warnings.warn(
                            f"Label: {label} is not in the activity labels config, ignoring"
                        )
                        activity_label = "background"
                    else:
                        activity = activity[0]
                        activity_label = activity["label"]

                gt_f.write(f"{activity_label}\n")

        # bundles
        with open(f"{bundle_dir}/{split}.split1.bundle", "a+") as bundle:
            bundle.write(f"{video_name}.txt\n")

print("Done!")
print(f"Saved training data to {output_data_dir}")
