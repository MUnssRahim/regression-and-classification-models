import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def dataprocessing(filepath):
    df = pd.read_csv(filepath)
    nulls = df.isnull().sum().sum()
    df = df.fillna(df.mean())
    df['Bedroom/Area'] = df['Bedroom'] / df['Area']
    return nulls, df

def trainsplit(df):
    x = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)
    return model, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def plot_graphs(df, y_test, y_pred):
    plt.figure(figsize=(12, 6))
    sorted_idx = np.argsort(df['Bedroom/Area'].values)
    plt.plot(df['Bedroom/Area'].values[sorted_idx], df['SalePrice'].values[sorted_idx], color='red', linewidth=2, label='Regression Line')
    plt.xlabel('Bedroom/Area')
    plt.ylabel('SalePrice')
    plt.title('Bedroom/Area vs SalePrice')
    plt.show()

    plt.scatter(y_test, y_pred, color='blue', label='Test vs Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Line of Best Fit')
    plt.xlabel('Actual SalePrice')
    plt.ylabel('Predicted SalePrice')
    plt.title('Actual vs Predicted SalePrice')
    plt.legend()
    plt.show()

def predict_user(model, scaler):
    features = ['Area', '1st sqft', '2nd sqft', 'FullBath', 'HalfBath', 'Bedroom']
    user_input = {}
    for feature in features:
        while True:
            try:
                user_input[feature] = float(input(f"{feature}: "))
                break
            except:
                print("Enter numeric value.")
    user_df = pd.DataFrame([user_input], columns=features)
    user_df['Bedroom/Area'] = user_df['Bedroom'] / user_df['Area']
    user_scaled = scaler.transform(user_df)
    prediction = model.predict(user_scaled)
    print(f"Predicted SalePrice: ${prediction[0]}")

base_path = os.path.join("C:", os.sep, "Users", "HP", "Desktop")
data_file = os.path.join(base_path, "Housingcustomdata.csv")
output_file = os.path.join(base_path, "Housingcustomdata_with_new_feature.csv")

null_count, df = dataprocessing(data_file)
df.to_csv(output_file, index=False)

model, X_train, X_test, y_train, y_test, scaler = trainsplit(df)
y_pred = model.predict(X_test)

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

plot_graphs(df, y_test, y_pred)
predict_user(model, scaler)
