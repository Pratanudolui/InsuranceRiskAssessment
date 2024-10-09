# File: scripts/generate_data.py
from flask import Flask, render_template, request
import pandas as pd
import random
import string
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Function to generate random unique names
def generate_unique_names(n):
    return [''.join(random.choices(string.ascii_uppercase, k=5)) for _ in range(n)]

# Function to generate random data
def generate_data():
    names = generate_unique_names(100000)  # Generating 10000 unique names
    ages = list(range(18, 70))
    genders = ['M', 'F']
    occupations = ['Engineer', 'Doctor', 'Lawyer', 'Student', 'Retired', 'Teacher', 'Architect', 'Artist', 'Manager']
    marital_statuses = ['Single', 'Married', 'Divorced', 'Widowed']
    education_levels = ['Bachelor', 'Master', 'PhD', 'None']
    annual_incomes = list(range(20000, 150000, 5000))
    locations = ['Urban', 'Rural', 'Suburban']
    vehicle_makes = ['Toyota', 'Ford', 'BMW', 'Honda', 'Nissan', 'Chevrolet', 'Tesla', 'Mercedes', 'Audi']
    vehicle_ages = list(range(1, 15))
    engine_sizes = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    mileages = list(range(5000, 20000, 1000))
    accidents = [0, 1, 2, 3]
    violations = [0, 1, 2, 3, 4]
    claims = [0, 1, 2, 3]
    license_durations = list(range(1, 40))
    risks = ['Low', 'Medium', 'High']

    # Generating 10000 random data points
    data = {
        'name': names,
        'age': [random.choice(ages) for _ in range(100000)],
        'gender': [random.choice(genders) for _ in range(100000)],
        'occupation': [random.choice(occupations) for _ in range(100000)],
        'marital_status': [random.choice(marital_statuses) for _ in range(100000)],
        'education_level': [random.choice(education_levels) for _ in range(100000)],
        'annual_income': [random.choice(annual_incomes) for _ in range(100000)],
        'residential_location': [random.choice(locations) for _ in range(100000)],
        'vehicle_make': [random.choice(vehicle_makes) for _ in range(100000)],
        'vehicle_age': [random.choice(vehicle_ages) for _ in range(100000)],
        'engine_size': [random.choice(engine_sizes) for _ in range(100000)],
        'mileage_driven_annually': [random.choice(mileages) for _ in range(100000)],
        'accident_history': [random.choice(accidents) for _ in range(100000)],
        'traffic_violations': [random.choice(violations) for _ in range(100000)],
        'claims_history': [random.choice(claims) for _ in range(100000)],
        'license_duration': [random.choice(license_durations) for _ in range(100000)],
        'risk': [random.choice(risks) for _ in range(100000)]
    }

    # Creating the DataFrame
    df = pd.DataFrame(data)

    # Displaying the first few rows
    print(df.head())

    # Saving the DataFrame to a CSV file
    csv_file_path = 'generated_insurance_risk_data12.csv'
    df.to_csv(csv_file_path, index=False)

    print(f'Data successfully saved to {csv_file_path}')

if __name__ == '__main__':
    generate_data()
