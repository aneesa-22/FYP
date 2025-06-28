# The code you provided uses a deep learning model for training, testing, and predicting the outcome of diabetes prediction. Specifically, 
# the model used is a Feedforward Neural Network (FNN), which is trained using Keras with the TensorFlow backend.

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# --- Load and preprocess the dataset ---
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
df = pd.read_csv(url, header=None, names=column_names)

# Separate features and target
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Deep Learning Model ---
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))  # Prevent overfitting
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test), verbose=1)

# Evaluate model accuracy
accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# # Save model and scaler
# model.save("diabetes_model_dl.h5")
# with open("scaler.pkl", "wb") as f:
#     pickle.dump(scaler, f)

# Save model and scaler in the current working directory
model.save("./diabetes_model_dl.h5")
with open("./scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


# # --- Function to make predictions ---
# def predict_diabetes(input_data):
#     # Load the saved model and scaler
#     from keras.models import load_model
#     model = load_model("diabetes_model_dl.h5")
#     with open("scaler.pkl", "rb") as f:
#         scaler = pickle.load(f)

#     # Prepare the input data (scale it using the same scaler)
#     input_scaled = scaler.transform([input_data])

#     # Make the prediction
#     prediction = model.predict(input_scaled)[0][0]
#     result = "DIABETIC" if prediction >= 0.5 else "NON DIABETIC"
#     return result

# # Example usage
# sample_input = [2, 130, 80, 25, 100, 28.0, 0.5, 33]  # Example input
# result = predict_diabetes(sample_input)
# print(f"Prediction: {result}")


from sklearn.metrics import classification_report

# Evaluate model accuracy on the test set
y_pred = model.predict(X_test)
y_pred_binary = (y_pred >= 0.5).astype(int)  # Convert probabilities to binary labels

# Compute and print classification metrics
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))
