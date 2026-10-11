#made by team insert_team_name, 11 Oct MMXXVI
#this code was written with Gen AI (Gemini)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv('C:/Users/akshat/Projects/AIML Workshop/el plan/ps1/food_delivery_time.csv')
# 2. Select features (X) and target (y)
features = [
    'distance_km', 
    'prep_time_min', 
    'courier_experience_years', 
    'order_items', 
    'traffic_delay_min', 
    'courier_wait_min'
]
target = 'delivery_time_min'

X = df[features]
y = df[target]

# 3. Handle missing values using mean imputation
# (Since your dataset has missing values in distance, experience, etc.)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 4. Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

# 5. Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Make predictions on the test set
y_pred = model.predict(X_test)

# 7. Evaluate the model performance
r2 = r2_score(y_test, y_pred)
# Note: if your scikit-learn version is older and doesn't support 
# root_mean_squared_error, you can use mean_squared_error(..., squared=False)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R-squared (R2 Score): {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} minutes")

# 8. View how much each feature impacts delivery time
print("\nFeature Coefficients:")
for feat, coef in zip(features, model.coef_):
    print(f" - {feat}: {coef:.4f}")

plt.figure(figsize=(14, 6))

# Plot 1: Actual vs Predicted Delivery Time
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='royalblue', alpha=0.6, edgecolors='k')
# Perfect prediction line (y = x)
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Fit')
plt.title('Actual vs Predicted Delivery Time')
plt.xlabel('Actual Delivery Time (min)')
plt.ylabel('Predicted Delivery Time (min)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Plot 2: Feature Coefficients (Impact on Delivery Time)
plt.subplot(1, 2, 2)
colors = ['forestgreen' if c > 0 else 'lightgreen' for c in model.coef_]
plt.barh(features, model.coef_, color=colors, alpha=0.8, edgecolor='k')
plt.title('Feature Coefficients (Impact on Delivery Time)')
plt.xlabel('Coefficient Value')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

def run_cli():
    print("=" * 45)
    print("   FOOD DELIVERY TIME PREDICTION CLI")
    print("=" * 45)
    print("Type 'q' or 'quit' at any prompt to exit.\n")

    while True:
        user_inputs = []
        exit_flag = False

        for feature in features:
            while True:
                val_str = input(f"Enter value for {feature}: ").strip()
                if val_str.lower() in ['q', 'quit']:
                    exit_flag = True
                    break
                try:
                    user_inputs.append(float(val_str))
                    break
                except ValueError:
                    print("  [!] Invalid number. Please try again.")
            
            if exit_flag:
                break

        if exit_flag:
            print("\nExiting. Goodbye!")
            break

        # 3. Predict delivery time for the entered values
        input_data = np.array([user_inputs])
        prediction = model.predict(input_data)[0]

        print("-" * 45)
        print(f"Estimated Delivery Time: {prediction:.2f} minutes")
        print("-" * 45 + "\n")

if __name__ == "__main__":
    run_cli()