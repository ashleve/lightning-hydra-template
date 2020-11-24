## Setup

#### 1. Install anaconda
https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html

#### 2. Create anaconda env
```
    conda create --name hack_env
    conda activate hack_env
```

#### 3. Make sure proper python PATH is loaded
Unix
```
    which python
```
Windows
```
    for %i in (python.exe) do @echo. %~$PATH:i
```
Expected result: `PATH_TO_CONDA/envs/ENV_NAME/bin/python`

#### 4. Install pytorch with conda
Installation command generator: https://pytorch.org/get-started/locally/
```
    conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
```

#### 5. Clone repo
```
    git clone https://github.com/kinoai/hackathon-template
```

#### 6. Install requirements with pip
```
    cd hackathon-template
    pip install -r requirements.txt
```

#### 7. Log into your wandb account
```
    wandb login
```

#### 8. PyCharm setup
- open this repository as PyCharm project
- set "hack_env" as project interpreter:<br> 
`Ctrl + Shift + A -> type "Project Interpreter"`
- mark folder "project" as sources root:<br>
`right click on directory -> "Mark Directory as" -> "Sources Root"`
- set terminal emulation:<br> 
`Ctrl + Shift + A -> type "Edit Configurations..." -> select "Emulate terminal in output console"`
- run training:<br>
`right click on train.py file -> "Run 'train'"`



#### Important notes!
- If you are not using GPU (CUDA incompatible GPU) you may need to specify the number of GPUs manually instead of leaving the default `-1` in `config.yaml`:
```
    num_of_gpus: 0
```