# ConceptBottleneck-GUI-Experiment

## Purpose
This project explores Concept Bottleneck Models (CBMs) through a simple CNN architecture using the MNIST dataset. The goal is to deeply understand CBMs, which will help in developing a GUI interface that builds on Lily’s paper, automating the integration of CBMs into large language models (LLMs).

## Project Structure
- `data/`: Placeholder for raw data files.
- `src/`: Contains the main source code.
  - `model.py`: Defines the `ConceptBottleneckCNN` class.
  - `train.py`: Implements the training loop with concept and classification losses.
  - `test.py`: Implements the testing loop with evaluation metrics.
  - `utils.py`: Contains helper functions for data loading and concept label generation.
  - `visualization.py`: Functions to plot concept activations and contributions.
  - `hooks.py`: Hook functions to capture concept activations and contributions.
- `config/`: Contains configuration files.
  - `config.json`: Stores hyperparameters and other settings.
- `scripts/`: Contains scripts for running the training and testing processes.
  - `run.py`: Main script to execute the pipeline.
- `outputs/`: Stores activations, contributions, and generated visualizations.
- `requirements.txt`: Lists dependencies for the project.
- `.gitignore`: Specifies files and directories to be ignored by Git.

## How to Run the Project

### 1. Set Up the Environment
#### Create and activate a virtual environment:
```bash
python -m venv cbm_env  # Create virtual environment
source cbm_env/bin/activate  # Activate it on macOS/Linux
cbm_env\Scripts\activate  # Activate it on Windows
