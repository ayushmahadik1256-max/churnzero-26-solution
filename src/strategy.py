import pandas as pd
import numpy as np

def generate_business_strategy(test_ids, predictions):
    print("Generating Data-Driven Retention Strategies...")
    
    results = pd.DataFrame({
        'customer_id': test_ids,
        'churn_probability': predictions
    })
    
    # Map probabilities to actionable business tiers
    conditions = [
        (results['churn_probability'] >= 0.75),
        (results['churn_probability'] >= 0.50) & (results['churn_probability'] < 0.75),
        (results['churn_probability'] >= 0.25) & (results['churn_probability'] < 0.50),
        (results['churn_probability'] < 0.25)
    ]
    
    strategies = [
        "CRITICAL: Immediate Relationship Manager (RM) Call + Annual Fee Waiver",
        "HIGH RISK: Targeted Email Campaign + Premium Cross-Sell Discount",
        "MEDIUM RISK: Automated In-App Engagement Nudge + NPS Survey",
        "LOW RISK: Standard Monitoring (No Intervention Cost)"
    ]
    
    results['retention_action'] = np.select(conditions, strategies, default="Unknown")
    
    # Save the final output
    output_file = "retention_strategy_results.csv"
    results.to_csv(output_file, index=False)
    print(f"Business strategy successfully saved to {output_file}")
