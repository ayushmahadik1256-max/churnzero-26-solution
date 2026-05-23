import os
from preprocess import load_and_preprocess
from train import train_and_evaluate
from strategy import generate_business_strategy

def main():
    # Define file paths (assuming running from root directory)
    TRAIN_PATH = os.path.join("data", "ChurnZero_dataset_v1.csv")
    TEST_PATH = os.path.join("data", "ChurnZero_test_v1.csv")
    
    # 1. Preprocess Data
    X_train, y_train, X_test, test_ids = load_and_preprocess(TRAIN_PATH, TEST_PATH)
    
    # 2. Train Model
    model = train_and_evaluate(X_train, y_train)
    
    # 3. Generate Predictions on Test Set
    print("Scoring test dataset...")
    test_predictions = model.predict_proba(X_test)[:, 1]
    
    # 4. Map to Business Strategy
    generate_business_strategy(test_ids, test_predictions)

if __name__ == "__main__":
    main()
