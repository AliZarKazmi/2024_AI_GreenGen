
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_excel('c:\\Users\\Ali Zar Kazmi\\Desktop\\dataset.xlsx')

# Split the data into features (inputs) and target variable (outputs)
X = data.drop(columns=['Reduction Effectiveness', 'Recommendation'])
y = data['Reduction Effectiveness']

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Function to predict reduction effectiveness and provide recommendation
def predict_reduction_effectiveness(CO2_Emissions, PM_Emissions, NOx_Emissions, Temperature, Humidity):
    # Make predictions using the trained model
    prediction = model.predict([[CO2_Emissions, PM_Emissions, NOx_Emissions, Temperature, Humidity]])
    
    reduction_effectiveness = "High" if prediction == "High" else "Moderate"
    
    # Generate recommendation based on prediction
    if reduction_effectiveness == "High":
        recommendation = "Consider reducing usage of the generator or switching to a cleaner fuel source."
    else:
        recommendation = "Regular maintenance of the generator can help optimize its efficiency and reduce emissions."
    
    # Return prediction and recommendation in sentence form
    return reduction_effectiveness, recommendation

# User input for environmental parameters
CO2_Emissions = float(input("Enter CO2 Emissions: "))
PM_Emissions = float(input("Enter PM Emissions: "))
NOx_Emissions = float(input("Enter NOx Emissions: "))
Temperature = float(input("Enter Temperature (°C): "))
Humidity = float(input("Enter Humidity (%): "))

# Call the function to predict reduction effectiveness
reduction_effectiveness, recommendation = predict_reduction_effectiveness(CO2_Emissions, PM_Emissions, NOx_Emissions, Temperature, Humidity)
print(f"The reduction effectiveness is {reduction_effectiveness}. Here are some recommendations: {recommendation}")

# Get predictions for the entire dataset
y_pred = model.predict(X)

# Generate confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(conf_matrix)








