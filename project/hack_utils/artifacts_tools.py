import wandb
import os


def upload_artifact():
    FOLDER_TO_UPLOAD_PATH = "../data/skyhacks"

    ARTIFACT_NAME = "skyhacks-dataset"
    TYPE = "dataset"  # type can be anything but we recommend sticking to "dataset" / "model" / "result"

    WANDB_PROJECT = "hackathon_template_test"
    WANDB_TEAM = "kino"

    # Set project which will own the artifact
    os.mkdir("../logs") if not os.path.exists("../logs") else None
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_TEAM,
        dir="../logs",
        name="artifact_upload",
        job_type="artifact"
    )

    # Recursively add all files from a directory
    artifact = wandb.Artifact(
        name=ARTIFACT_NAME,
        type=TYPE
    )
    artifact.add_dir(FOLDER_TO_UPLOAD_PATH)
    # artifact.add_file(MODEL_PATH)

    # Upload artifact
    run.log_artifact(artifact, aliases=["latest"])
    print("Files are being uploaded now...")


def download_artifact():
    SAVE_DIR = "../data/skyhacks"
    ARTIFACT = "kino/hackathon_template_test/skyhacks-dataset:v0"

    # Choose what artifact you want to download
    api = wandb.Api()
    artifact = api.artifact(ARTIFACT)

    # Download artifact
    artifact.download(root=SAVE_DIR)
    print("Files are being downloaded now...")


if __name__ == "__main__":
    upload_artifact()
    # download_artifact()
