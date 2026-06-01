import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("   SALARY PREDICTION MODEL TRAINER")
print("   Python 3.10 | scikit-learn pipeline")
print("=" * 60)

# LOAD AND WRANGLE DATA
print("\n[1/5] Loading and cleaning data...")

def wrangle(path):
    df = pd.read_csv(path)

    # Drop columns with no predictive value
    df.drop(columns=[
        "job_id", "salary_currency", "employee_residence",
        "company_name", "required_skills",
        "posting_date", "application_deadline"
    ], inplace=True)

    # Remove salary outliers using Tukey IQR rule
    Q1 = df["salary_usd"].quantile(0.25)
    Q3 = df["salary_usd"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df["salary_usd"] >= lower) & (df["salary_usd"] <= upper)]

    return df

df = wrangle("ai_job_dataset.csv")
print(f"   Dataset shape after cleaning: {df.shape}")
print(f"   Salary range: ${df['salary_usd'].min():,.0f} - ${df['salary_usd'].max():,.0f}")

# DEFINE FEATURES AND TARGET 
print("\n[2/5] Preparing features...")

# Included years_experience, benefits_score, job_description_length
categorical_features = [
    "job_title", "experience_level", "employment_type",
    "company_location", "company_size", "education_required", "industry"
]
numerical_features = [
    "remote_ratio", "years_experience"
]

X = df[categorical_features + numerical_features]
y = df["salary_usd"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training samples: {len(X_train)} | Test samples: {len(X_test)}")
print(f"   Features used: {len(categorical_features)} categorical + {len(numerical_features)} numerical")

baseline_mae = mean_absolute_error(y_test, [y_train.mean()] * len(y_test))
print(f"\n   Baseline MAE (guessing average): ${baseline_mae:,.0f}")

# BUILD PREPROCESSOR 
# ColumnTransformer lets us apply different transformations to
# different columns at the same time.
# - Categorical columns -> OneHotEncoder (converts labels to 0/1 columns)
# - Numerical columns   -> StandardScaler (normalises to mean 0, std 1)

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ("num", StandardScaler(), numerical_features)
])

# TRAIN AND COMPARE MODELS 
print("\n[3/5] Training and comparing models...")
print("-" * 50)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=10.0),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results[name] = {"pipeline": pipeline, "MAE": mae, "RMSE": rmse, "R2": r2}

    print(f"  {name}")
    print(f"    MAE:  ${mae:>10,.0f}  |  RMSE: ${rmse:>10,.0f}  |  R²: {r2:.4f}")
    print()

# HYPERPARAMETER TUNING ON BEST MODEL
# GridSearchCV tries every combination of parameters you give it,
# using cross-validation, and picks the best combination.

print("[4/5] Hyperparameter tuning on Random Forest...")
print("      (This may take 2-3 minutes, please wait)")

param_grid = {
    "model__n_estimators":  [100, 200],
    "model__max_depth":     [None, 10, 20],
    "model__min_samples_split": [2, 5],
}

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

tuned_mae  = mean_absolute_error(y_test, y_pred_best)
tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))
tuned_r2   = r2_score(y_test, y_pred_best)

print(f"\n  Best parameters found: {grid_search.best_params_}")
print(f"  Tuned Random Forest:")
print(f"    MAE:  ${tuned_mae:>10,.0f}  |  RMSE: ${tuned_rmse:>10,.0f}  |  R²: {tuned_r2:.4f}")

# Update results with tuned model
results["Random Forest (Tuned)"] = {
    "pipeline": best_model,
    "MAE": tuned_mae,
    "RMSE": tuned_rmse,
    "R2": tuned_r2
}

# SAVE BEST MODEL AND METRICS
print("\n[5/5] Saving best model...")

# Pick model with lowest MAE (excluding the pipeline objects for comparison)
best_name = min(
    {k: v for k, v in results.items()},
    key=lambda k: results[k]["MAE"]
)
print(f"   Winner: {best_name} with MAE ${results[best_name]['MAE']:,.0f}")

joblib.dump(results[best_name]["pipeline"], "salary_model.pkl")
print("   Saved: salary_model.pkl")

# Save metrics summary
metrics_summary = {
    name: {"MAE": v["MAE"], "RMSE": v["RMSE"], "R2": v["R2"]}
    for name, v in results.items()
    if name != "Random Forest"  # keep tuned version only
}
joblib.dump(metrics_summary, "model_metrics.pkl")
print("   Saved: model_metrics.pkl")

# Save feature names
feature_names_out = (
    best_model.named_steps["preprocessor"]
    .transformers_[0][1]
    .get_feature_names_out(categorical_features).tolist()
    + numerical_features
)
joblib.dump({
    "categorical_features": categorical_features,
    "numerical_features": numerical_features,
    "feature_names_out": feature_names_out
}, "feature_config.pkl")
print("   Saved: feature_config.pkl")

print("\n" + "=" * 60)
print("   TRAINING COMPLETE")
print(f"   Baseline MAE:   ${baseline_mae:,.0f}")
print(f"   Best model MAE: ${results[best_name]['MAE']:,.0f}")
improvement = (baseline_mae - results[best_name]['MAE']) / baseline_mae * 100
print(f"   Error reduced by: {improvement:.1f}% vs baseline")
print("=" * 60)