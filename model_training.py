import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('generated_insurance_risk_data.csv')

# Verify column names
print("Columns in dataset:", df.columns)

# Check if 'risk' (not 'risk_level') exists in the dataset
if 'risk' not in df.columns:
    raise KeyError("'risk' column not found in the dataset. Please ensure the dataset contains this column.")

# Separate features (X) and labels (y)
X = df.drop(columns=['risk'])  # Features
y = df['risk']  # Labels

# Prepare the label encoder for risk levels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # Encode labels

# Define columns for preprocessing
numeric_features = ['age', 'annual_income', 'vehicle_age', 'engine_size', 
                    'mileage_driven_annually', 'license_duration', 
                    'accident_history', 'traffic_violations', 
                    'claims_history']
categorical_features = ['gender', 'occupation', 'marital_status', 
                        'education_level', 'residential_location', 
                        'vehicle_make']

# Preprocess the data
def preprocess_data(df):
    # Impute missing values for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Impute and encode categorical features
    categorical_transformer = Pipeline
