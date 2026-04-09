# Travel Insurance Prediction
 
Predicting which customers are likely to buy travel insurance using machine learning.
 
## Goal
 
A travel insurance company wants to know: **which customers will buy insurance?**  
Instead of contacting everyone, the company can focus marketing on the right people and save money.
 
## Dataset
 
- 1,249 customers (after removing duplicates from 1,987 rows)
- 9 features: Age, Annual Income, Family Members, Chronic Diseases, Employment Type, Graduate Status, Frequent Flyer, Ever Travelled Abroad
- Target: `TravelInsurance` — 0 (did not buy) or 1 (bought)
- Class split: 61% No / 39% Yes
 
## What I did
 
1. **Exploratory Data Analysis** — distributions, correlations, outlier detection
2. **Statistical Inference** — 4 hypothesis tests (Welch t-test, z-tests) with confidence intervals
3. **Feature Engineering** — created `Travel_score` and `Income_per_member`
4. **Machine Learning** — compared 6 models, tuned hyperparameters, built an ensemble
5. **Threshold tuning** — found optimal decision threshold for the business use case
 
## Key findings
 
| Feature | Finding |
|---|---|
| EverTravelledAbroad | Strongest — 68% buy vs 31% who haven't |
| FrequentFlyer | Strong — 50% buy vs 35% |
| AnnualIncome | Higher earners buy more (confirmed by t-test) |
| Travel_score | Combined signal — up to 80% buy rate |
| GraduateOrNot | No effect — confirmed by hypothesis test |
 
### Model Evaluation

The models were evaluated based on their ability to predict insurance purchase probability. **CatBoost** was selected as the production model due to its superior performance in terms of Accuracy and F1-score, which are critical for minimizing classification errors in this specific use case.

| Model | ROC-AUC | Accuracy | F1 |
| :--- | :---: | :---: | :---: |
| Ensemble | 0.7387 | 0.7737 | 0.6244 |
| Decision Tree (tuned) | 0.7347 | 0.7577 | 0.6009 |
| Gradient Boosting (tuned) | 0.7220 | 0.7657 | 0.6150 |
| **CatBoost (tuned)** | **0.7141** | **0.7747** | **0.6324** |
| SVM (tuned) | 0.7279 | 0.7427 | 0.5798 |
| Logistic Regression | 0.6980 | 0.7147 | 0.5474 |
| Random Forest | 0.6433 | 0.6686 | 0.5469 |

**Final Model Selection:**
* **Selected Model:** CatBoost (tuned)
* **Optimal Threshold:** 0.30
* **Key Metrics:** Accuracy: 0.7747 | F1-Score: 0.6324

> **Note on Selection:** Although the Ensemble model demonstrated a slightly higher ROC-AUC, **CatBoost (tuned)** was prioritized for its better F1-score and higher overall Accuracy. In the context of insurance lead generation, achieving a higher F1-score is more valuable as it provides a better balance between precision and recall, effectively identifying potential customers while minimizing false positives.
## How to run
 
```bash
pip install -r requirements.txt
```
 
Open `travel_insurance.ipynb` in Jupyter or Google Colab.

## How to improve
 
- Try XGBoost or LightGBM
- Handle class imbalance with SMOTE
- More feature engineering (age groups, income brackets)
- Wider hyperparameter search with Optuna
- More data — 1,249 rows is a small dataset
