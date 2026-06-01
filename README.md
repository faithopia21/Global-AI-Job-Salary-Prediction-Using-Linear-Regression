# Global AI Job Salary Prediction Using Linear Regression

Data Source: `kaggle.com`

> Predicting annual salary in USD based on job features like title, experience level, and location.

---

## Project Overview

This project builds and compares multiple machine learning models that predict job salaries in USD using various job-related features such as:

- Job title
- Experience level
- Years of experience
- Employment type
- Remote ratio
- Education level
- Industry
- Company size

The goal is to help candidates, HR professionals, and job boards estimate salary ranges for tech-related roles using historical job posting data.

---

## Problem Statement

> How can we accurately predict a job's salary in USD given its title, required experience, and other posting features?

---

## Dataset Description

- Check the data dictionary file for details.

The dataset consists of job postings with the following columns:

| Feature | Description |
|---|---|
| `job_title` | Job title (e.g., Data Scientist, ML Engineer) |
| `salary_usd` | Target: Annual salary in USD |
| `experience_level` | EN (Entry), MI (Mid), SE (Senior), EX (Executive) |
| `employment_type` | FT (Full-time), PT, CT, FL |
| `company_location` | Country where the company is located |
| `remote_ratio` | 0 = Onsite, 50 = Hybrid, 100 = Fully Remote |
| `company_size` | S (<50), M (50-250), L (>250) |
| `education_required` | Minimum required education level |
| `years_experience` | Required years of experience |
| `industry` | Company industry |

### Features Removed and Why

| Feature | Reason |
|---|---|
| `benefits_score` | Describes the offer itself, not known to a user before receiving an offer |
| `job_description_length` | Same as above, not a user-known value at prediction time |
| `required_skills` | High cardinality, comma-separated text not suitable for this model |
| `employee_residence` | High overlap with company_location |
| `company_name` | High cardinality, too many unique values |
| `salary_currency` | Redundant, salary_usd already standardises currency |
| `posting_date` / `application_deadline` | Not relevant to salary prediction |

---

## Exploratory Data Analysis (EDA)

Key EDA steps included:

- Distribution of salaries across job titles and experience levels
- Correlation between numeric features (e.g., experience vs salary)
- Boxplots of salary by employment type and company size
- Outlier removal using the Tukey IQR rule on `salary_usd`
- Cardinality analysis to identify high-cardinality columns before encoding

---

## Machine Learning Approach

| Step | Details |
|---|---|
| Model Type | Random Forest Regressor (Tuned via GridSearchCV) |
| Models Compared | Linear Regression, Ridge Regression, Random Forest, Gradient Boosting |
| Preprocessing | OneHotEncoding for categorical features, StandardScaler for numerical features |
| Feature Selection | Removed multicollinearity, high-cardinality, and offer-side features |
| Train-Test Split | 80% train / 20% test |
| Hyperparameter Tuning | GridSearchCV with 3-fold cross-validation, 12 parameter combinations |
| Evaluation Metrics | MAE, RMSE, R² Score |

---

## Model Results

All four models were trained and evaluated on the same test set for a fair comparison.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | $14,229 | $18,580 | 0.8650 |
| Ridge Regression | $14,173 | $18,580 | 0.8650 |
| Gradient Boosting | $13,299 | $17,926 | 0.8744 |
| **Random Forest (Tuned)** | **$13,082** | **$17,485** | **0.8805** |

**Baseline MAE: $41,225** (error from guessing the average salary every time, with no model).

**Error reduction: 68.3%** compared to baseline.

### Hyperparameter Tuning

GridSearchCV tested 12 combinations of Random Forest settings using 3-fold cross-validation (36 total fits):

```python
param_grid = {
    "model__n_estimators":      [100, 200],
    "model__max_depth":         [None, 10, 20],
    "model__min_samples_split": [2, 5],
}
```

Best parameters found: `max_depth=20`, `min_samples_split=5`, `n_estimators=200`

---

## Key Insights

- Seniority level and job title are the strongest predictors of salary.
- Years of experience has a strong positive correlation with salary.
- Remote-friendly jobs tend to have higher average salaries.
- Education level and company size also show meaningful correlations.
- Company location is a significant factor, with US-based roles paying the most on average.

---

## Streamlit App

An interactive web app was built using Streamlit allowing real-time salary predictions.

App features:
- Salary prediction with a realistic +-10% range
- Model comparison table showing MAE, RMSE, and R² for all four models
- Feature importance chart showing which factors drive salary most
- Full evaluation metrics displayed for the winning model

Run the app:

```bash
streamlit run main.py
```

---

## Tech Stack

- Python 3.13
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Jupyter Notebook

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/faithopia21/Global-AI-Job-Salary-Prediction-Using-Linear-Regression
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Retrain the model (optional, model files are already included):
   ```bash
   python train_model.py
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

5. Or open the exploration notebook:
   ```bash
   jupyter notebook ai_jobs_explore.ipynb
   ```

---

## Contact

For questions, collaborations, or feedback:

**Iyanu Arowosola**
iarowosola@yahoo.com
[LinkedIn Profile](https://linkedin.com/in/iyanuarowosola)

---

*This project was developed as part of a machine learning portfolio targeting job market insights in the tech and data space.*