## Setup
1. Install anaconda if you don't already have it

    https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html

2. Clone repo
    ```
        git clone https://github.com/kinoai/hackathon-template
        cd hackathon-template
    ```
3. Create conda environment (change 'hack_env' to any name you want)
    ```
        conda update conda
        conda env create -f conda_env.yml -n hack_env
        conda activate hack_env
    ```
4. Make sure proper python PATH is loaded<br>
    - Unix
        ```
            which python
        ```
    - Windows
        ```
            for %i in (python.exe) do @echo. %~$PATH:i
        ```
    Expected result: `PATH_TO_CONDA/envs/ENV_NAME/bin/python`
5. Install requirements
    ```
        pip install -r requirements.txt
    ```
6. Log to your Weights&Biases account
    ```
        wandb login
    ```
7. Run training
    ```
        cd project
        python train.py
    ```
<br>


#### Important notes!
- If you are not using GPU (CUDA incompatible GPU) you may need to specify the number of GPUs manually instead of leaving the default `-1` in `project_config.yaml`:
    ```
        num_of_gpus: 0
    ```
<br>


## PyCharm setup
- open this repository as PyCharm project
- set "hack_env" as project interpreter:<br> 
`Ctrl + Shift + A -> type "Project Interpreter"`
- mark folder "project" as sources root:<br>
`right click on directory -> "Mark Directory as" -> "Sources Root"`
- set terminal emulation:<br> 
`Ctrl + Shift + A -> type "Edit Configurations..." -> select "Emulate terminal in output console"`
- run training:<br>
`right click on train.py file -> "Run 'train'"`