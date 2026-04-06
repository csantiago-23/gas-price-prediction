# Deploying the Gas Price App

## Run Locally (2 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure the pipeline has been run (creates data files)
python run_pipeline.py

# 3. Launch the app
streamlit run app.py
```
Opens automatically at http://localhost:8501

---

## Deploy to Streamlit Cloud (free, public URL)

1. Push this entire `Gas Price/` folder to a GitHub repo
2. Go to **share.streamlit.io** → Sign in with GitHub
3. Click **New app** → select your repo → set main file to `app.py`
4. Click **Deploy** — live in ~2 minutes

Your app gets a public URL like:
`https://your-username-gas-price-app.streamlit.app`

Put this link on your resume and LinkedIn.

---

## File structure required

```
Gas Price/
├── app.py                   ← Streamlit app (this file)
├── requirements.txt         ← Dependencies
├── run_pipeline.py          ← Run once to generate data files
├── data/
│   ├── gaspricesexcel.csv   ← Raw EIA data (required)
│   ├── gas_prices_features.csv
│   ├── test_predictions.csv
│   └── model_metrics.csv
└── scripts/
    ├── 01_data_collection.py
    ├── 02_data_cleaning.py
    ├── 03_eda.py
    ├── 04_model_training.py
    └── 05_model_evaluation.py
```

**Important:** Run `python run_pipeline.py` once before pushing to GitHub
so that `data/test_predictions.csv` and `data/model_metrics.csv` exist.
Streamlit Cloud will use those pre-computed files.
