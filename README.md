# Abstractive Text Summarization

This repository contains a PyTorch Lightning-based project for abstractive text summarization. It leverages the Hugging Face Transformers library and supports parameter-efficient fine-tuning (PEFT) techniques like LoRA and Prefix-Tuning.

## Features

- **Abstractive Summarization:** The core task is to generate a concise and coherent summary of a given text.
- **Multilingual Model:** The project is configured to use `facebook/mbart-large-50`, a multilingual model, making it adaptable to various languages. The default configuration is for Arabic (`ar_AR`).
- **Parameter-Efficient Fine-Tuning (PEFT):**
    - **LoRA (Low-Rank Adaptation):** Efficiently fine-tune large models with a small number of trainable parameters.
    - **Prefix-Tuning:** Another PEFT technique that adds a small number of trainable parameters to the model's prefix.
    - **Full Fine-Tuning:** The project also supports traditional full fine-tuning of the model.
- **PyTorch Lightning:** The training and evaluation pipelines are built using PyTorch Lightning, which provides a clean and organized structure for training deep learning models.
- **Hydra for Configuration:** All aspects of the project, including the model, trainer, and data, are configured using YAML files managed by Hydra. This allows for easy experimentation and reproducibility.
- **ROUGE for Evaluation:** The model's performance is evaluated using the ROUGE metric.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Muhammad-Abdelsattar/Abstractive-summarizer abstractive_summarizer
   cd abstractive_summarizer
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The main configuration file for this project is `config.yaml`. This file is the central place to control all aspects of the project.

### Data Configuration

Update the `data.files` section in `config.yaml` with the correct paths to your training, validation, and test data files. The data should be in CSV format.

```yaml
data:
  files:
    train_data_source: /path/to/your/train_data.csv
    valid_data_source: /path/to/your/valid_data.csv
    test_data_source: /path/to/your/test_data.csv
```

### Model and Training Configuration

You can modify various parameters in `config.yaml` to change the behavior of the model and the training process. Here are some of the key parameters:

- `model.model_id`: The Hugging Face model to use (e.g., `t5-base`, `facebook/bart-large-cnn`).
- `model.use_peft_config`: The PEFT technique to use. Set to `lora` for LoRA, `prefix` for Prefix-Tuning, or `null` for full fine-tuning.
- `training.trainer.max_epochs`: The number of epochs to train for.
- `training.trainer.accelerator`: The hardware accelerator to use (`cpu`, `gpu`).
- `training.optim.optimizers.AdamW.lr`: The learning rate for the AdamW optimizer.
- `data.params.train_batch_size`: The batch size for training.

### Overriding Configuration from the Command Line

You can override any parameter from `config.yaml` directly from the command line using Hydra's syntax. For example:

- To change the model and use LoRA:
  ```bash
  python main.py model.model_id=t5-base model.use_peft_config=lora
  ```
- To change the learning rate and number of epochs:
  ```bash
  python main.py training.optim.optimizers.AdamW.lr=1e-5 training.trainer.max_epochs=5
  ```

## Training

To start the training process, run the `main.py` script:

```bash
python main.py
```

The training progress will be logged to the console, and the logs will be saved in the `./logs` directory by default.

### Checkpoints

PyTorch Lightning will automatically save checkpoints of your model during training. The checkpoints will be saved in the `./checkpoints` directory. The best model (based on the `Rouge-l` score on the validation set) will be saved.

You can resume training from a checkpoint by setting the `training.checkpoint` parameter in `config.yaml` or from the command line:

```bash
python main.py training.checkpoint=/path/to/your/checkpoint.ckpt
```

## Evaluation

The model is evaluated on the validation set at the end of each epoch during training. The ROUGE scores are logged.

To run a standalone evaluation on the test set, you can use the `evaluate.py` script. You need to provide the path to a trained model checkpoint.

```bash
python evaluate.py /path/to/your/checkpoint.ckpt
```

This will load the model from the checkpoint, run it on the test set defined in `config.yaml`, and print the ROUGE scores.

## Using a Different Model

To use a different summarization model from the Hugging Face Hub, you need to update the `model.model_id` in `config.yaml`. For example, to use the `t5-base` model, you would change the configuration to:

```yaml
model:
  model_id: t5-base
```

When using a different model, you might need to adjust other parameters as well, such as the `tokenizer_args` if the new model requires different language settings. For most sequence-to-sequence models on the Hub, changing the `model_id` should be sufficient.

## Dependencies

The main dependencies of this project are:
- `torch`
- `pytorch-lightning`
- `transformers`
- `peft`
- `hydra-core`
- `omegaconf`
- `pandas`
- `rouge`
- `sentencepiece`
- `protobuf==3.19.0`
