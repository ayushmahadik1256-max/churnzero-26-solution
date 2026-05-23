# ChurnZero 26: Predictive Banking Churn Solution

## The Business Problem
Banks lose 15-25% of customers annually to churn, taking their deposits, cross-sell revenue, and customer lifetime value (CLV) with them. Traditional models focus solely on accuracy or AUC, but fail to provide actionable steps for business teams.

## Our Solution
This repository contains a full-stack machine learning pipeline that not only predicts customer churn with high accuracy using XGBoost, but explicitly maps those predictions to **Data-Driven Retention Strategies**. 

By segmenting customers into risk tiers, we optimize the intervention cost—ensuring high-cost actions (like fee waivers and RM calls) are reserved only for high-value, high-risk customers, while low-risk customers are nurtured via automated digital channels.

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Place the datasets in the `data/` folder.
3. Run the pipeline: `python src/main.py`

## Deliverables
- **Model Output:** ROC-AUC optimized XGBoost pipeline handling severe class imbalance.
- **Business Impact:** Outputs a `retention_strategy_results.csv` mapping every test customer to a specific, actionable business strategy.
