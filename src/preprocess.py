import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def load_and_preprocess(train_path, test_path):
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 1. Feature Engineering (Business Domain Logic)
    for df in [train_df, test_df]:
        # Digital Disengagement Score (Low score = high churn risk)
        df['digital_engagement_score'] = df['mobile_app_login_count'] + df['website_login_count']
        
        # Financial Stress Index (High utilization + late payments = exit risk)
        df['financial_stress_index'] = df['credit_utilization_ratio'] * (df['late_credit_card_payment_count'] + 1)
        
        # Service Friction (High complaints + slow resolution)
        df['service_friction_score'] = df['total_complaints'] * df['complaint_resolution_time']

    # 2. Separate Features and Target
    X_train = train_df.drop(columns=['customer_id', 'churn'])
    y_train = train_df['churn']
    X_test = test_df.drop(columns=['customer_id'])
    
    # 3. Handle Categorical Variables
    cat_cols = X_train.select_dtypes(include=['object']).columns
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])
    
    return X_train, y_train, X_test, test_df['customer_id']
