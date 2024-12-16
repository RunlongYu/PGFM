
# Physics-Guided Foundation Model for Scientific Discovery: An Application to Aquatic Science
This repository is the official implementation of **Physics-Guided Foundation Model for Scientific Discovery: An Application to Aquatic Science** submitted to **AAAI-25 aisi track**


![''](img/PGFM-v5.png)

## Usage

<!-- ### Example -->
#### 1. Pre-training stage
```
cd stage_main
python3 train.py --strategy n+1
```

| Argument         | Type   | Default | Description                                                  | Choices                                                         |
| ---------------- | ------ | ------- | ------------------------------------------------------------ | --------------------------------------------------------------- |
| `--model`        | `str`  | `FM`    | The model to be used.                                        | `FM`, `FM_TF`                                                   |
| `--stage1_label` | `str`  | `sim`   | The label of the prediction column for stage 1.              | `sim`      |
| `--dataset`      | `str`  | `lake`  | The dataset to be used.                                      |                                                                 |
| `--gpu`          | `int`  | `0`     | The GPU device ID to be used.                                |                                                                 |
| `--mutation`     | `int`  | `1`     | Use mutation during training (1 to use, 0 to not use).       |                                                                 |
| `--strategy`     | `str`  | `n+1`   | The strategy to be used during training.                     | `1,1`, `1+1`, `n,1`, `n+1`                                      |
| `--seed`         | `int`  | `42`    | The random seed for reproducibility.                         |                                                                 |

#### 2. Fine-tuning stage for predicting lake water temperature
```
cd stage2_scripts

python3 train.py --model_type fm-pg --strategy n+1 --seed 40 --pt_datetime 'fm-model_datetime' --current_time
```


| Argument         | Type   | Default              | Description                                         | Choices                                                               |
| ---------------- | ------ | ------------------- | --------------------------------------------------- | --------------------------------------------------------------------- |
| `--model_type`   | `str`  | `fm-pg`             | Get the model type from input arguments.            | `lstm`, `ealstm`, `transformer`, `fm-lstm`, `fm-transformer`, `fm-pg`, `PB` |
| `--strategy`     | `str`  | `n+1`               | The strategy to be used during training.            | `1,1`, `1+1`, `n,1`, `n+1`                                            |
| `--pt_datetime`  | `str`  | `2024-08-08-14-42`  | The datetime for the pre-trained model.             |                                                                       |
| `--gpu`          | `int`  | `0`                 | The GPU device ID to be used.                       |                                                                       |
| `--label_name`   | `str`  | `obs_temp`          | The label name to be used.                          |                                                                       |
| `--seed`         | `int`  | `40`                | The random seed for reproducibility.                |                                                                       |
| `--current_time` | `str`  | (Required)          | The start time for the current run.                 |                                                                       |


create data for [step3](#3-fine-tuning-stage-for-predicting-dissolved-oxygen-do-concentrations)

```
python3 create_data_for_do.py --stage2_datetime stage2_datetime
```

#### 3. Fine-tuning stage for predicting dissolved oxygen (DO) concentrations

```
cd stage3_scripts

python3 train.py --model_type fm-pg+ --strategy n+1 --seed 40 --pt_datetime 'fm-model_datetime' --current_time current_time
```
| Argument         | Type   | Default              | Description                                         | Choices                                                                         |
| ---------------- | ------ | ------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------- |
| `--model_type`   | `str`  | `fm-pg`             | Get the model type from input arguments.            | `lstm`, `ealstm`, `transformer`, `fm-lstm`, `fm-ealstm`, `fm-transformer`, `fm-pg`, `fm-pg+`, `PB` |
| `--strategy`     | `str`  | `n+1`               | The strategy to be used during training.            | `1,1`, `1+1`, `n,1`, `n+1`                                                      |
| `--pt_datetime`  | `str`  | `2024-08-08-14-42`  | The datetime for the pre-trained model.             |                                                                                 |
| `--gpu`          | `int`  | `0`                 | The GPU device ID to be used.                       |                                                                                 |
| `--label_name`   | `str`  | `obs_do`            | The label name to be used.                          |                                                                                 |
| `--seed`         | `int`  | `40`                | The random seed for reproducibility.                |                                                                                 |
| `--current_time` | `str`  | (Required)          | The time when training starts.                      |                                                                                 |



## Directory Descriptions
```text
Project Directory/
│
├── config/                # Stores configuration files, such as hyperparameters, environment variables, and settings
│
├── data/                  # Stores datasets
│
├── draw/                  # Contains scripts for generating visualizations or plots
│
├── layer/                 # Includes implementations of various layers used in our models
│
├── model/                 # Contains the PGFM model and all other baseline architectures, along with related files
│
├── optimizer/             # Contains implementations of optimizers used in training
│
├── run/                   # Includes scripts for running the training of the foundation model
│
├── stage1_main/           # Holds the main scripts for training of the foundation model
│
├── stage1_trainer/        # Contains training scripts specific to the foundation model
│
├── stage2_scripts/        # Includes scripts specific to fine-tuning models for predicting water temperature
│
├── stage3_scripts/        # Includes scripts specific to fine-tuning models for predicting DO concentrations
│
├── utils/                 # Utility functions and helper scripts that are commonly used across different scripts
│
└── README.md              # Project documentation
```

## Contact

Should you have any questions regarding our paper or codes, please don't hesitate to reach out via email at chq29@pitt.edu or ruy59@pitt.edu.


## Acknowledgment 
Our code is developed based on [GitHub - jdwillard19/MTL_lakes-Torch: Meta Transfer Learning for Lake Temperature Prediction](https://github.com/jdwillard19/MTL_lakes) and [GitHub - RunlongYu/CELS](https://github.com/RunlongYu/CELS).