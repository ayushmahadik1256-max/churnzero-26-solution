import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

def train_and_evaluate(X, y):
    print("Training XGBoost Model...")
    
    # Split for validation
    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle Class Imbalance
    pos_weight = (len(y_t) - y_t.sum()) / y_t.sum()
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric='auc',
        early_stopping_rounds=50
    )
    
    model.fit(
        X_t, y_t,
        eval_set=[(X_v, y_v)],
        verbose=False
    )
    
    # Evaluate
    val_preds = model.predict_proba(X_v)[:, 1]
    auc = roc_auc_score(y_v, val_preds)
    print(f"Validation ROC-AUC Score: {auc:.4f}")
    
    return model
