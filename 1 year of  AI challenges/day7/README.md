# Telco Customer Churn Prediction

This project predicts customer churn using machine learning with scikit-learn.

## Setup

1. **Clone the repository** and navigate to the `day7` directory.
2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On Mac/Linux
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script:
```bash
python -m app
```

## Data
- `Telco-Customer-Churn.csv` and `churn.csv` are the datasets used for training and evaluation.

## Troubleshooting
- If you encounter `ModuleNotFoundError`, ensure your virtual environment is activated and dependencies are installed.
- If you see version conflicts, try reinstalling dependencies with `pip install --force-reinstall -r requirements.txt`.
