# Setup and run

This guide sets up the Movie Recommendation System on Windows with PowerShell. Run commands from the project root, the folder containing `requirements.txt`.

## Prerequisites

- Python 3.10
- The project dependencies listed in `requirements.txt`
- The MovieLens 1M dataset
- Generated model files in `models/` and processed CSV files in `data/processed/`

The dataset, processed CSV files, and trained models are intentionally excluded from GitHub. A fresh clone must generate them before the app can start.

## 1. Create a virtual environment

Open PowerShell in the project folder and run:

```powershell
py -3.10 -m venv .venv
```

If PowerShell blocks activation, allow scripts for the current terminal session and activate the environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

The prompt should show `(.venv)` when activation succeeds.

## 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Download the dataset and generate artifacts

Download and extract the [MovieLens 1M dataset](https://files.grouplens.org/datasets/movielens/ml-1m.zip). Copy `movies.dat`, `ratings.dat`, and `users.dat` into `data/raw/`, creating that directory if needed.

Run the notebooks in this order to create the processed CSV files and model files:

1. `02_preprocessing.ipynb`
2. `03_collaborative_filtering.ipynb`
3. `04_content_based.ipynb`
4. `05_hybrid_recommendation.ipynb`

For notebook-relative paths to work, start Jupyter with the `notebooks/` directory as its working root. From the project root, run:

```powershell
Set-Location notebooks
python -m jupyter notebook
```

Open and run the notebooks in the listed order. `01_eda.ipynb` and `06_evaluation.ipynb` are optional.

## 4. Start the app

```powershell
python -m streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit, usually <http://localhost:8501>.

To stop the app, focus the PowerShell terminal running Streamlit and press `Ctrl+C`.

## Optional: run on another port

If port 8501 is already in use:

```powershell
python -m streamlit run app/streamlit_app.py --server.port 8502
```

Then open <http://localhost:8502>.

## Troubleshooting

- **`py` is not recognized or Python 3.10 is unavailable:** Install Python 3.10, then reopen PowerShell. Use `py -0p` to list installed Python versions.
- **A required project file is missing:** Confirm these paths exist relative to the project root:

  ```text
  models/svd_model.pkl
  models/movie_similarity.pkl
  data/processed/movies_processed.csv
  data/processed/train_ratings.csv
  ```

- **`streamlit` is not recognized:** Activate `.venv` and run the app with `python -m streamlit run app/streamlit_app.py`.
- **Dependency installation fails:** Confirm the virtual environment is active, then rerun `python -m pip install -r requirements.txt` and review the first package error printed by pip.
- **The app is still running after closing the browser:** The server runs in the terminal; stop it with `Ctrl+C`.

## macOS or Linux

From the project root, create and activate a virtual environment, install dependencies, and start the app:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```
