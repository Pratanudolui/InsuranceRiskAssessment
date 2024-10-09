import pandas as pd
import random
import string

# Function to generate random unique names
def generate_unique_names(n):
    names = set()
    while len(names) < n:
        name = ''.join(random.choices(string.ascii_uppercase, k=5))
        names.add(name)
    return list(names)

# Function to generate random health insurance data
def generate_health_data():
    names = generate_unique_names(10000)  # Generating 10000 unique names
    ages = list(range(18, 100))
    genders = ['M', 'F']
    bmi_values = [round(random.uniform(15.0, 40.0), 1) for _ in range(10000)]
    cholesterol_levels = [random.randint(100, 300) for _ in range(10000)]
    blood_pressure_levels = [random.randint(90, 180) for _ in range(10000)]
    smoking_status = ['Non-smoker', 'Smoker']
    exercise_frequency = ['Never', 'Occasionally', 'Regularly']
    alcohol_consumption = ['Never', 'Occasionally', 'Frequently']
    pre_existing_conditions = ['None', 'Diabetes', 'Hypertension', 'Heart Disease', 'Cancer', 'Asthma']
    marital_statuses = ['Single', 'Married', 'Divorced', 'Widowed']
    income_brackets = list(range(20000, 200001, 10000))
    residential_areas = ['Urban', 'Rural', 'Suburban']
    family_medical_history = ['None', 'Diabetes', 'Hypertension', 'Cancer', 'Heart Disease']
    healthcare_access = ['Excellent', 'Good', 'Average', 'Poor']
    annual_claims = list(range(0, 10001, 1000))
    num_doctor_visits = list(range(0, 12))
    num_specialist_visits = list(range(0, 11))
    risk_levels = ['Low', 'Medium', 'High']

    # Generating 10000 random data points
    data = {
        'name': names,
        'age': [random.choice(ages) for _ in range(10000)],
        'gender': [random.choice(genders) for _ in range(10000)],
        'bmi': bmi_values,
        'cholesterol': cholesterol_levels,
        'blood_pressure': blood_pressure_levels,
        'smoking_status': [random.choice(smoking_status) for _ in range(10000)],
        'exercise_frequency': [random.choice(exercise_frequency) for _ in range(10000)],
        'alcohol_consumption': [random.choice(alcohol_consumption) for _ in range(10000)],
        'pre_existing_conditions': [random.choice(pre_existing_conditions) for _ in range(10000)],
        'marital_status': [random.choice(marital_statuses) for _ in range(10000)],
        'annual_income': [random.choice(income_brackets) for _ in range(10000)],
        'residential_area': [random.choice(residential_areas) for _ in range(10000)],
        'family_medical_history': [random.choice(family_medical_history) for _ in range(10000)],
        'healthcare_access': [random.choice(healthcare_access) for _ in range(10000)],
        'annual_claims': [random.choice(annual_claims) for _ in range(10000)],
        'num_doctor_visits': [random.choice(num_doctor_visits) for _ in range(10000)],
        'num_specialist_visits': [random.choice(num_specialist_visits) for _ in range(10000)],
        'risk': [random.choice(risk_levels) for _ in range(10000)]
    }

    # Creating the DataFrame
    df = pd.DataFrame(data)

    # Ensure that all columns expected in the analysis are generated
    expected_columns = ['name', 'age', 'gender', 'bmi', 'cholesterol', 
                        'blood_pressure', 'smoking_status', 'exercise_frequency', 
                        'alcohol_consumption', 'pre_existing_conditions', 
                        'marital_status', 'annual_income', 'residential_area', 
                        'family_medical_history', 'healthcare_access', 
                        'annual_claims', 'num_doctor_visits', 
                        'num_specialist_visits', 'risk']
    
    # Check for missing columns and raise an error if any are missing
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in generated data: {missing_columns}")

    # Saving the DataFrame to a CSV file
    csv_file_path = 'generated_health_insurance_risk_data.csv'
    df.to_csv(csv_file_path, index=False)
    print(f'Data successfully saved to {csv_file_path}')

if __name__ == '__main__':
    generate_health_data()
