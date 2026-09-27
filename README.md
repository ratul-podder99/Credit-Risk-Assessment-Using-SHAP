# Credit Risk Assessment Using SHAP

A machine learning web application that predicts the probability of a borrower defaulting on a loan and serves the prediction through a FastAPI backend with a lightweight HTML/CSS/JS frontend. Model development, preprocessing, and explainability analysis (via SHAP) are documented in the accompanying Jupyter notebook.

> ⚠️ **Disclaimer:** This project is built for educational and portfolio purposes only. It is **not** intended for real-world credit or lending decisions.

---

## 🔗 Live Demo

https://credit-risk-assessment-using-shap-042w.onrender.com/

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [API Reference](#api-reference)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Locally](#running-locally)
- [Deployment (Render)](#deployment-render)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Overview

This project trains a supervised classification model on a credit risk dataset to estimate the probability that a loan applicant will default. The trained pipeline (preprocessing + model) is serialized with `joblib` and served via a FastAPI backend. The API exposes a single `/predict` endpoint that accepts applicant details and returns a default probability, a binary risk prediction, and a human-readable risk label.

Rather than using a fixed 0.5 classification threshold, the project tunes a custom decision threshold (`best_threshold.pkl`) to better balance precision and recall for the credit risk use case, where false negatives (approving a high-risk applicant) tend to be more costly than false positives.

Model interpretability is explored using **SHAP (SHapley Additive exPlanations)** in the notebook to understand which applicant features (e.g. income, loan amount, credit history length) most influence the model's predictions — an important consideration for financial models, which typically require some degree of explainability.

## Features

- 🔍 **Loan default prediction** based on 11 applicant/loan attributes
- ⚙️ **FastAPI backend** with automatic request validation via Pydantic
- 🎯 **Tuned classification threshold** instead of a naive 0.5 cutoff
- 📊 **Model explainability** via SHAP, documented in the training notebook
- 🌐 **Static frontend** (HTML/CSS/JS) served directly by FastAPI — no separate frontend server needed
- ☁️ **Render-ready deployment** via `render.yaml`
- 🔄 **CORS-enabled** for cross-origin frontend/backend setups

## Tech Stack

| Layer | Technology |
|---|---|
| Backend / API | [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/) |
| Data Validation | [Pydantic](https://docs.pydantic.dev/) |
| Modeling | [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/) |
| Explainability | [SHAP](https://shap.readthedocs.io/) |
| Data Handling | [pandas](https://pandas.pydata.org/) |
| Model Serialization | [joblib](https://joblib.readthedocs.io/) |
| Frontend | HTML, CSS, JavaScript (served as static files) |
| Deployment | [Render](https://render.com/) |

## Project Structure

```
Credit-Risk-Assessment-Using-SHAP/
├── static/                        # Frontend assets (HTML, CSS, JS) served by FastAPI
├── credit_risk_assesment.ipynb    # EDA, preprocessing, model training & SHAP analysis
├── main.py                        # FastAPI application and /predict endpoint
├── requirements.txt               # Python dependencies
├── runtime.txt                    # Python version pin for Render
├── render.yaml                    # Render deployment configuration
├── .gitignore
└── README.md
```

> **Note:** The trained model artifacts (`credit_risk_model.pkl` and `best_threshold.pkl`) are required at runtime but are excluded from version control via `.gitignore`. See [Getting Started](#getting-started) for how to obtain or generate them.

## Dataset

The model is trained on tabular loan application data with the following applicant-level features:

| Feature | Description |
|---|---|
| `person_age` | Applicant's age |
| `person_income` | Applicant's annual income |
| `person_home_ownership` | Home ownership status (e.g. RENT, OWN, MORTGAGE) |
| `person_emp_length` | Length of employment (years) |
| `loan_intent` | Purpose of the loan (e.g. EDUCATION, MEDICAL, PERSONAL) |
| `loan_grade` | Assigned loan grade (A–G) |
| `loan_amnt` | Requested loan amount |
| `loan_int_rate` | Loan interest rate |
| `loan_percent_income` | Loan amount as a percentage of income |
| `cb_person_default_on_file` | Whether the applicant has a prior default on record (Y/N) |
| `cb_person_cred_hist_length` | Length of the applicant's credit history (years) |

The target variable is a binary loan default indicator. Full exploratory data analysis, feature distributions, and preprocessing steps are available in `credit_risk_assesment.ipynb`.

## Methodology

The end-to-end pipeline, documented in the notebook, generally follows:

1. **Exploratory Data Analysis (EDA)** — distribution checks, missing value analysis, and outlier inspection.
2. **Preprocessing** — numerical imputation (`SimpleImputer`) and categorical encoding (`OneHotEncoder`) combined via a `ColumnTransformer`/`Pipeline`.
3. **Model Training** — a gradient-boosted classifier (XGBoost) trained on the processed features.
4. **Threshold Tuning** — instead of defaulting to 0.5, a custom probability threshold is selected (e.g. to optimize F1-score or a business-relevant metric) and persisted as `best_threshold.pkl`.
5. **Explainability with SHAP** — SHAP values are computed to interpret global feature importance and individual predictions, helping surface *why* the model flags an applicant as high or low risk.
6. **Serialization** — the final pipeline and threshold are exported with `joblib` for use in the FastAPI service.

> For exact evaluation metrics (accuracy, precision, recall, ROC-AUC) and SHAP visualizations, refer to `credit_risk_assesment.ipynb`.

## API Reference

### `POST /predict`

Predicts the probability of loan default for a given applicant.

**Request Body**

```json
{
  "person_age": 25,
  "person_income": 55000,
  "person_home_ownership": "RENT",
  "person_emp_length": 3.0,
  "loan_intent": "EDUCATION",
  "loan_grade": "B",
  "loan_amnt": 12000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.22,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 4
}
```

**Response**

```json
{
  "default_probability": 0.1732,
  "default_prediction": 0,
  "threshold": 0.35,
  "Result": "Low Risk"
}
```

| Field | Type | Description |
|---|---|---|
| `default_probability` | float | Model's predicted probability of default (0–1) |
| `default_prediction` | int | Binary prediction (`1` = default, `0` = no default), based on the tuned threshold |
| `threshold` | float | The decision threshold used to convert probability into a class label |
| `Result` | string | `"High Risk"` or `"Low Risk"`, human-readable version of `default_prediction` |

Once the server is running, interactive API documentation (Swagger UI) is also available at:

```
http://127.0.0.1:8000/docs
```

### Frontend

The root path (`/`) serves the static frontend (`static/index.html`), which provides a form-based UI for submitting applicant data to `/predict` without needing to call the API directly.

## Getting Started

### Prerequisites

- Python **3.10** or **3.11** (see `runtime.txt`)
- pip
- [Homebrew](https://brew.sh/) (macOS only, required for XGBoost's OpenMP dependency)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ratul-podder99/Credit-Risk-Assessment-Using-SHAP.git
   cd Credit-Risk-Assessment-Using-SHAP
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **macOS only — install the OpenMP runtime required by XGBoost**
   ```bash
   brew install libomp
   ```

5. **Add the trained model artifacts**

   Place `credit_risk_model.pkl` and `best_threshold.pkl` in the project root (these are excluded from the repo via `.gitignore`). Generate them by running `credit_risk_assesment.ipynb` end-to-end, or obtain them separately if provided.

### Running Locally

```bash
uvicorn main:app --reload
```

The app will be available at:

- Frontend: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

## Deployment (Render)

This project includes a `render.yaml` for one-click deployment on [Render](https://render.com/):

```yaml
services:
  - type: web
    name: credit-ledger
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    autoDeploy: true
```

**Steps:**

1. Push this repository to GitHub (with `credit_risk_model.pkl` and `best_threshold.pkl` available to the deployed environment — see note below).
2. In the Render dashboard, choose **New → Blueprint** and connect the repository. Render will read `render.yaml` automatically.
3. Render provisions the service, installs dependencies from `requirements.txt`, and starts the app with Uvicorn.

> **Important:** Since `.pkl` model files are excluded from git, they will **not** be present on Render unless you provide them another way — e.g. commit them via Git LFS, upload them to a persistent disk, or download them from cloud storage (S3/GCS) during startup before `joblib.load()` runs in `main.py`'s `lifespan` handler.

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `InconsistentVersionWarning` / `AttributeError: ... _RemainderColsList` | scikit-learn version mismatch between training and runtime | Pin `scikit-learn` in `requirements.txt` to the exact version used for training |
| `ModuleNotFoundError: No module named 'xgboost'` | XGBoost not installed | `pip install xgboost` and add it to `requirements.txt` |
| `XGBoostError: Library (libxgboost.dylib) could not be loaded` (macOS) | Missing OpenMP runtime | `brew install libomp` |
| `FileNotFoundError` on startup | Missing `.pkl` model files | Ensure `credit_risk_model.pkl` and `best_threshold.pkl` exist in the project root |

## Roadmap

- [ ] Add automated tests for the `/predict` endpoint
- [ ] Expose SHAP-based explanations per prediction via the API
- [ ] Add model versioning and experiment tracking
- [ ] Add CI/CD pipeline for automated testing and deployment
- [ ] Containerize with Docker

## Contributing

Contributions, issues, and feature requests are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project currently has no license specified. If you intend to open-source it, consider adding an [MIT License](https://choosealicense.com/licenses/mit/) or similar.

## Author

**Ratul Podder**
GitHub: [@ratul-podder99](https://github.com/ratul-podder99)
