import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Preprocessing function
def preprocess_data(df=None):
    # Load data if no input DataFrame is provided (for training)
    if df is None:
        df = pd.read_csv('generated_insurance_risk_data.csv')

    # Handle missing values, categorical encoding, etc.
    label_encoder = LabelEncoder()

    # Encode target (risk)
    df['risk'] = label_encoder.fit_transform(df['risk'])

    # Separate features and target
    features = df.drop(['risk'], axis=1)
    target = df['risk']

    # If it's for training, perform train-test split
    if df is not None:
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, label_encoder
    else:
        return features, label_encoder

# Function to preprocess user input (keeping the structure of the training data)
def preprocess_input(input_df, X_train_columns):
    # Convert categorical data to the same structure as the training data
    input_df_processed = pd.get_dummies(input_df)

    # Ensure input matches the training data structure by reindexing
    input_df_processed = input_df_processed.reindex(columns=X_train_columns, fill_value=0)

    return input_df_processed

if __name__ == '__main__':
    preprocess_data()
