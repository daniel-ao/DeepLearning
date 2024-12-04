
# TP2: Machine Learning Project

## Description
This project, **TP2**, is designed for implementing and evaluating machine learning models. It includes components for data preprocessing, model training, and evaluation, accompanied by example scripts and configuration templates.

The repository consists of two main sections:
- `code/sujet`: Contains the base implementation for the project, designed as a starting point for development.
- `correction`: A corrected or finalized version of the project, including additional features and configurations.

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
├── dataset.md                  # Documentation for the dataset
└── README.md                   # Project documentation
```

## Requirements
To set up the project, ensure you have the following installed:
- Python 3.8 or higher
- Required libraries specified in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

## Setup and Installation
1. Clone the repository or extract the downloaded zip file:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the `code/sujet` or `correction` directory depending on your usage.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
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

### Configuration
Use the provided YAML configuration file (`default.yaml`) to customize parameters.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
[License Name] - Placeholder for license details.

---

### Notes
- Ensure all dependencies are installed to avoid runtime errors.
- Use the `subject.pdf` file as a reference for project requirements and guidelines.
- Customize the `default.yaml` configuration to fit your specific use case.
