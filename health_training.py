import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load your health data for modeling
health_data = pd.read_csv('generated_health_insurance_risk_data.csv')

# Specify the numeric and categorical features
num_features = ['bmi', 'cholesterol', 'blood_pressure', 'annual_claims', 'num_doctor_visits', 'num_specialist_visits']
cat_features = ['age', 'gender', 'smoking_status', 'exercise_frequency', 'alcohol_consumption', 
                'pre_existing_conditions', 'marital_status', 'annual_income', 
                'residential_area', 'family_medical_history', 'healthcare_access']

# Prepare the DataFrame for processing
health_X = health_data[num_features + cat_features]
health_y = health_data['risk']  # Ensure 'risk' is present in the DataFrame

# Split the data into training and testing sets
health_X_train, health_X_test, health_y_train, health_y_test = train_test_split(
    health_X, health_y, test_size=0.2, random_state=42
)

# Set up your preprocessor
health_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),  # Scale numeric features
        ('cat', OneHotEncoder(), cat_features)     # One-hot encode categorical features
    ]
)

# Fit and transform the training data
try:
    health_X_train_processed = health_preprocessor.fit_transform(health_X_train)
    health_X_test_processed = health_preprocessor.transform(health_X_test)  # Transform test data using the same preprocessor
    print("Preprocessing completed successfully.")
except Exception as e:
    print("An error occurred during preprocessing:", e)

# Train a model (example: KNN Classifier)
model_knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
model_knn.fit(health_X_train_processed, health_y_train)

# Make predictions on the test set
health_y_pred_knn = model_knn.predict(health_X_test_processed)

# Evaluate the model
print("Confusion Matrix for KNN:\n", confusion_matrix(health_y_test, health_y_pred_knn))
print("Classification Report for KNN:\n", classification_report(health_y_test, health_y_pred_knn))
print("Columns in DataFrame:", health_data.columns.tolist())

