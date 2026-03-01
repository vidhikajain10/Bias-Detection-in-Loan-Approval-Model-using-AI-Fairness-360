AI Fairness Audit of Loan Approval Model
Project Overview

This project evaluates a machine learning loan approval model for gender bias using IBM AI Fairness 360.

The objective is to:

Train a classification model

Measure fairness metrics

Identify bias across protected groups

Interpret ethical risks in model predictions

Tools & Technologies

Python

Scikit-learn

IBM AI Fairness 360

NumPy & Pandas

Fairness Metrics Used

Statistical Parity Difference

Disparate Impact

Equal Opportunity Difference

How to Run

Install dependencies:

pip install -r requirements.txt

Train model:

python src/train_model.py

Run fairness audit:

python src/fairness_audit.py
Key Outcome

The audit demonstrates how machine learning systems can produce biased outcomes across demographic groups and highlights the importance of fairness evaluation before deployment.
