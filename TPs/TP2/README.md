
# TP2: Machine Learning Project

## Description
This project, **TP2**, is designed for implementing and evaluating machine learning models. It includes components for data preprocessing, model training, and evaluation, accompanied by example scripts and configuration templates.

The repository consists of two main sections:
- `code`: A corrected or finalized version of the project, including additional features and configurations.

## Features
- **Data Processing**: Scripts for data handling and preprocessing.
- **Model Training**: Python modules for training machine learning models.
- **Configuration Templates**: Easily configurable YAML templates for parameter management.
- **Script Automation**: Using `Justfile` for streamlined command execution.
- **Organized Directory**: Separation of source code, configuration files, and project documentation.

## Directory Structure
```
TP2/
├── code/
│   └── sujet/
│       ├── src/                # Source files for the main project
│       │   ├── dataset.py      # Handles dataset loading and preprocessing
│       │   ├── model.py        # Defines machine learning model structure
│       │   └── trainer.py      # Implements model training and evaluation
│       ├── .gitignore          # Specifies ignored files for version control
│       ├── Justfile            # Task automation file
│       ├── main.py             # Entry point for the project
│       └── requirements.txt    # Dependencies for the project
├── correction/
│   ├── src/                    # Source files for the corrected version
│   │   ├── dataset.py          # Dataset management
│   │   ├── model.py            # Machine learning model
│   │   └── trainer.py          # Training process
│   ├── configs/                # Configuration templates
│   │   ├── default.yaml        # Default configuration
│   │   └── template.py         # Python template for configurations
│   ├── .gitignore              # Specifies ignored files
│   ├── Justfile                # Task automation file
│   ├── main.py                 # Entry point for corrected implementation
│   └── requirements.txt        # Dependencies for the corrected version
├── subject.pdf                 # Project assignment details
├── dataset.md                  # Kaggle link for the dataset
└── README.md                   # Project documentation
```

## Usage
### Running the Main Script
Navigate to the desired directory (`code/sujet` or `correction`) and execute:
```bash
python main.py
```

### Automating Tasks
The `Justfile` simplifies repetitive tasks:
```bash
just <command>
```
Refer to the `Justfile` for available commands.

