Perfect — here’s a single, consolidated documentation draft that covers everything you’ve done so far in one file. It’s structured so you can skim it, track progress, and append future phases easily.

---

# Employee Burnout Analysis — Project Documentation

## Project Overview

This project predicts employee burnout using machine learning models. Users provide work-related details and immediately receive a burnout risk score. The system is designed as a user-facing product with potential for tracking historical predictions.

**Tech Stack:** Python, scikit-learn, XGBoost, Flask, PostgreSQL, React, Tailwind CSS, Docker

---

## Current Progress

### 1. Data Preprocessing

* Dataset loaded and cleaned using `preprocess_data` function
* Missing values handled, categorical features encoded
* Target column **Burn Rate** preserved
* Preprocessing verified through automated test

**Preprocessing Test:** Passed ✅

---

### 2. Model Training & Evaluation

Three models trained and evaluated: Linear Regression, Random Forest, and XGBoost.

**Metrics Summary:**

| Model             | RMSE   | MAE    | R²     |
| ----------------- | ------ | ------ | ------ |
| Linear Regression | 0.0707 | 0.0535 | 0.8687 |
| Random Forest     | 0.0590 | 0.0464 | 0.9085 |
| XGBoost           | 0.0584 | 0.0461 | 0.9105 |

**Insights:**

* Mental Fatigue Score is the dominant predictor across all models
* Resource Allocation is the second most important feature
* Linear Regression coefficients saved for interpretability
* Random Forest and XGBoost feature importances visualized via horizontal bar charts

**Models Saved:**

* `models/lr_model.pkl`
* `models/rf_model.pkl`
* `models/xgb_model.pkl`

---

### 3. Testing

Automated tests implemented for the pipeline:

1. **Preprocessing Test**

   * Ensures output is a DataFrame
   * Checks no missing values
   * Confirms target column exists
     **Result:** Passed ✅

2. **Model Files Existence Test**

   * Confirms all model files exist
     **Result:** Passed ✅

3. **Prediction Tests**

   * Ensures each model returns valid numeric predictions for a sample input
     **Result:** Passed ✅
   * XGBoost prediction type issue resolved (cast to float)

**All Tests Passed Successfully**

---

### 4. Issues & Resolutions

| Issue                                            | Resolution                                 |
| ------------------------------------------------ | ------------------------------------------ |
| XGBoost predictions failed type check            | Cast prediction to float in test function  |
| Missing Linear Regression feature interpretation | Added coefficient table output             |
| Initial Git push pending                         | Completed first push after Phase 1 updates |

---

## Next Steps (Future Documentation Placeholder)

* **Flask API:** `/predict`, `/history`, SHAP explanations
* **Frontend:** React form, result page, history page
* **Database:** PostgreSQL integration for session-based history
* **Testing:** Unit, integration, E2E with Cypress
* **Dockerization:** Full product deployment

---

This document can live as `docs/project_notes.md` and serves as the single source of truth for everything done so far. Future updates can append to the **Next Steps** section as each phase progresses.

---

If you want, I can also **add a small diagram for feature importance and model workflow** to make this documentation visually complete and presentation-ready. Do you want me to do that next?
