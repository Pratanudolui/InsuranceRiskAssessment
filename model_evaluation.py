# File: scripts/model_evaluation.py
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report

# Import the preprocess_data function from the same directory
from preprocessing import preprocess_data  

# Call the preprocess function and unpack the values
X_train, X_test, y_train, y_test = preprocess_data()

def evaluate_model(X_test, y_test):
    # Load the trained model from the saved file
    with open('knn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions using the test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Classification Report:\n{report}")

    # Generate risk-based financial assessment (example: using annual income)
    risk_factors = X_test['annual_income'] * 0.1  # Example calculation
    results_df = X_test.copy()
    results_df['Predicted_Risk'] = y_pred
    results_df['Risk_Factor_Payout'] = risk_factors
    
    # Save the results to a CSV file
    results_df.to_csv('risk_assessment_results.csv', index=False)
    print("Results saved to data/risk_assessment_results.csv")

if __name__ == '__main__':
    evaluate_model(X_test, y_test)
