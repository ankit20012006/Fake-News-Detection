# Fake News Detection

A Streamlit application that classifies a news article as **Fake** or **Real** using a TF-IDF vectorizer and a Logistic Regression model. The app also displays class-confidence scores, a chart, and a session prediction history that can be downloaded as CSV.

> This is a machine-learning demonstration, not a fact-checking service. Predictions can be wrong and should not be treated as evidence that a claim is true or false.

## Requirements

- Python 3.10 or newer
- The model files `lr_model.jb` and `vectorizer.jb` in the project directory

## Install and run on Windows

Open PowerShell in the directory containing `app.py`, then run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open [http://localhost:8502](http://localhost:8502) in your browser. If the `py` launcher is unavailable, replace `py` with `python` when creating the environment.

To stop the app, press `Ctrl+C` in the terminal.

## Install and run on macOS or Linux

From the directory containing `app.py`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open [http://localhost:8502](http://localhost:8502) in your browser.

## Using the app

1. Paste a news article into the text area.
2. Select **Check News** to see the predicted label and confidence scores.
3. Review earlier predictions in the History tab. Use Download to export the current session history as CSV.

The history is held in the current Streamlit session and is not saved as a permanent record.

## Training

`train_model.py` trains a TF-IDF Logistic Regression model from `Fake.csv` and `True.csv`, which must be in the project directory. Run:

```powershell
python train_model.py
```

The training script reports evaluation metrics and writes `lr_model_new.jb` and `vectorizer_new.jb`. The app currently loads `lr_model.jb` and `vectorizer.jb`; training does not replace those files automatically. To use a newly trained model, update the filenames loaded near the top of `app.py` to the new artifacts.

`news.ipynb` is also included as a notebook for exploring the model and dataset.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit user interface and prediction workflow |
| `lr_model.jb`, `vectorizer.jb` | Model and vectorizer loaded by the app |
| `train_model.py` | Command-line model training script |
| `lr_model_new.jb`, `vectorizer_new.jb` | Artifacts produced by `train_model.py` |
| `Fake.csv`, `True.csv` | Fake and real news datasets used for training |
| `news.ipynb` | Model exploration notebook |
| `requirements.txt` | Python dependencies; scikit-learn is pinned to match the bundled models |

## Troubleshooting

- **`No module named streamlit`:** Activate the project environment and run `python -m pip install -r requirements.txt`.
- **Model file not found:** Run Streamlit from the project directory and make sure `lr_model.jb` and `vectorizer.jb` are present.
- **Model version warning:** Install dependencies from `requirements.txt`; the bundled model files were serialized with scikit-learn 1.6.1.

  ## 🚀 Deployment
The application is deployed on Render and is available online:
🔗 **Live Demo:** https://fake-news-detection-8cpi.onrender.com/



