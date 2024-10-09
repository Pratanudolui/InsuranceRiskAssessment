from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import pandas as pd
import os
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Flask(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Check if file has allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load and preprocess training data for vehicle and health
def load_data(file_path):
    df = pd.read_csv(file_path)
    X_train = df.drop(columns=['risk'])
    y_train = LabelEncoder().fit_transform(df['risk'])
    return X_train, y_train

# Define preprocessing pipelines
def create_preprocessing_pipelines(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

# Train KNN model
def train_knn_model(X_train, y_train, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

# Define vehicle and health features
vehicle_features = {
    'numeric': ['age', 'annual_income', 'vehicle_age', 'engine_size', 
                'mileage_driven_annually', 'license_duration', 
                'accident_history', 'traffic_violations', 
                'claims_history'],
    'categorical': ['gender', 'occupation', 'marital_status', 
                    'education_level', 'residential_location', 
                    'vehicle_make']
}

health_features = {
    'numeric': ['age', 'bmi', 'cholesterol', 'blood_pressure', 
                'annual_income', 'annual_claims', 
                'num_doctor_visits', 'num_specialist_visits'],
    'categorical': ['gender', 'smoking_status', 'exercise_frequency', 
                    'alcohol_consumption', 'pre_existing_conditions', 
                    'marital_status', 'residential_area', 
                    'family_medical_history', 'healthcare_access']
}

# Load data and train models
vehicle_X_train, vehicle_y_train = load_data('generated_insurance_risk_data.csv')
health_X_train, health_y_train = load_data('generated_health_insurance_risk_data.csv')

# Create preprocessing pipelines
vehicle_preprocessor = create_preprocessing_pipelines(vehicle_features['numeric'], vehicle_features['categorical'])
health_preprocessor = create_preprocessing_pipelines(health_features['numeric'], health_features['categorical'])

# Preprocess training data
vehicle_X_train_processed = vehicle_preprocessor.fit_transform(vehicle_X_train)
health_X_train_processed = health_preprocessor.fit_transform(health_X_train)

# Train KNN models
vehicle_knn_model = train_knn_model(vehicle_X_train_processed, vehicle_y_train)
health_knn_model = train_knn_model(health_X_train_processed, health_y_train)

# Insurance amount calculation
def calculate_insurance_amount(user_data, risk_level, insurance_type='vehicle'):
    if insurance_type == 'vehicle':
        base_amount = 500
        adjustments = {
            'age': (200 if user_data['age'] < 25 else 0),
            'income': (100 if user_data['annual_income'] < 30000 else 0),
            'accidents': (300 if user_data['accident_history'] > 0 else 0),
            'violations': (200 if user_data['traffic_violations'] > 0 else 0),
            'claims': (150 if user_data['claims_history'] > 0 else 0),
            'risk': (50 if risk_level == 'Medium' else 100 if risk_level == 'High' else 0)
        }
    else:  # health
        base_amount = 300
        adjustments = {
            'age': (200 if user_data['age'] > 50 else 0),
            'bmi': (150 if user_data['bmi'] > 30 else 0),
            'cholesterol': (100 if user_data['cholesterol'] > 240 else 0),
            'blood_pressure': (100 if user_data['blood_pressure'] > 140 else 0),
            'smoking': (250 if user_data['smoking_status'] == 'Yes' else 0),
            'risk': (75 if risk_level == 'Medium' else 150 if risk_level == 'High' else 0)
        }

    total_adjustment = sum(adjustments.values())
    return base_amount + total_adjustment

# Route to handle the initial index page
@app.route("/")
def index():
    return render_template("index.html")

# Route for vehicle details page
@app.route('/vehicle', methods=['GET'])
def vehicle():
    return render_template('vehicle.html')

# Route for health details page
@app.route('/health', methods=['GET'])
def health():
    return render_template('form_health.html')

# Route to handle vehicle form submission
@app.route("/vehicle/form", methods=["GET", "POST"])
def vehicle_form():
    if request.method == "POST":
        user_data = {
            'age': int(request.form['age']),
            'gender': request.form['gender'],
            'occupation': request.form['occupation'],
            'marital_status': request.form['marital_status'],
            'education_level': request.form['education_level'],
            'annual_income': int(request.form['annual_income']),
            'residential_location': request.form['residential_location'],
            'vehicle_make': request.form['vehicle_make'],
            'vehicle_age': int(request.form['vehicle_age']),
            'engine_size': float(request.form['engine_size']),
            'mileage_driven_annually': int(request.form['mileage_driven_annually']),
            'accident_history': int(request.form['accident_history']),
            'traffic_violations': int(request.form['traffic_violations']),
            'claims_history': int(request.form['claims_history']),
            'license_duration': int(request.form['license_duration'])
        }

        input_df = pd.DataFrame([user_data])
        X_input_processed = vehicle_preprocessor.transform(input_df)

        predicted_risk = vehicle_knn_model.predict(X_input_processed)
        risk_level = LabelEncoder().fit(['Low', 'Medium', 'High']).inverse_transform(predicted_risk)[0]

        amount = calculate_insurance_amount(user_data, risk_level, insurance_type='vehicle')

        return render_template("result_vehicle.html", risk_level=risk_level, amount=amount)
    return render_template("form_vehicle.html")




# Route to handle CSV upload for vehicle insurance
@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'file' not in request.files:
            return render_template("upload.html", error="No file part")

        file = request.files['file']
        if file.filename == '':
            return render_template("upload.html", error="No selected file")

        if file and allowed_file(file.filename):
            try:
                # Read the CSV file directly
                df = pd.read_csv(file)

                # Preprocess the uploaded data
                X_uploaded = df[vehicle_features['numeric'] + vehicle_features['categorical']]
                X_uploaded_processed = vehicle_preprocessor.transform(X_uploaded)

                # Predict the vehicle risks for uploaded data
                predictions = vehicle_knn_model.predict(X_uploaded_processed)
                df['predicted_risk'] = LabelEncoder().fit(['Low', 'Medium', 'High']).inverse_transform(predictions)

                # Calculate insurance amounts for each row
                df['insurance_amount'] = df.apply(lambda row: calculate_insurance_amount(row, row['predicted_risk'], insurance_type='vehicle'), axis=1)

                # Save the results as CSV
                output_file = 'predicted_vehicle_risks.csv'
                output_path = os.path.join(UPLOAD_FOLDER, output_file)
                df.to_csv(output_path, index=False)

                return render_template("upload.html", tables=[df.to_html(classes='data')], titles=df.columns.values, output_file=output_file)

            except Exception as e:
                return render_template("upload.html", error=f"Error processing file: {e}")

    return render_template("upload.html")

# Route to download the generated CSV file
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

# Route to handle health form submission
@app.route("/health/form", methods=["GET", "POST"])
def health_form():
    if request.method == "POST":
        user_data = {
            'age': int(request.form['age']),
            'gender': request.form['gender'],
            'smoking_status': request.form['smoking_status'],
            'exercise_frequency': request.form['exercise_frequency'],
            'alcohol_consumption': request.form['alcohol_consumption'],
            'bmi': float(request.form['bmi']),
            'cholesterol': int(request.form['cholesterol']),
            'blood_pressure': int(request.form['blood_pressure']),
            'annual_income': int(request.form['annual_income']),
            'num_doctor_visits': int(request.form['num_doctor_visits']),
            'num_specialist_visits': int(request.form['num_specialist_visits']),
            'marital_status': request.form['marital_status'],
            'residential_area': request.form['residential_area'],
            'family_medical_history': request.form['family_medical_history'],
            'healthcare_access': request.form['healthcare_access'],
            'annual_claims': int(request.form['annual_claims']),
            'pre_existing_conditions': request.form['pre_existing_conditions']
        }

        input_df = pd.DataFrame([user_data])
        X_input_processed = health_preprocessor.transform(input_df)

        predicted_risk = health_knn_model.predict(X_input_processed)
        risk_level = LabelEncoder().fit(['Low', 'Medium', 'High']).inverse_transform(predicted_risk)[0]

        amount = calculate_insurance_amount(user_data, risk_level, insurance_type='health')

        return render_template("result_health.html", risk_level=risk_level, amount=amount)
    return render_template("form_health.html")

if __name__ == "__main__":
    app.run(debug=True)
