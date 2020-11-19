import wandb
import os


def make_artifact():
    DATASET_PATH = "../data/skyhacks"

    # Set project which will own the artifact
    os.mkdir("../logs") if not os.path.exists("../logs") else None
    run = wandb.init(
        project="hackathon_template_test",
        entity="kino",
        dir="../logs",
        name="artifact_upload",
        job_type="artifact"
    )

    # Recursively add all files from a directory
    artifact = wandb.Artifact(
        "skyhacks-dataset",     # dataset name
        type="dataset"          # type can be anything but we recommend sticking to "dataset" / "model" / "result"
    )
    artifact.add_dir(DATASET_PATH)
    # artifact.add_file(PATH, name="some_model.h5")

    # Upload artifact
    run.log_artifact(artifact, aliases=["latest", "cleaned"])
    print("Files are being uploaded now...")


def download_artifact():
    SAVE_DIR = "../data/skyhacks"

    # Choose what artifact you want to download
    api = wandb.Api()
    artifact = api.artifact('kino/hackathon_template_test/skyhacks-dataset:v0')

    # Download artifact
    artifact.download(root=SAVE_DIR)
    print("Files are being downloaded now...")


if __name__ == "__main__":
    make_artifact()
    # download_artifact()
