# Before coding any models, it is a good idea to install
# the necessary libraries:
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# After that, the data set needs to be loaded and put
# into data frames so that it is easier to work with:
# CSVs live in ml/data/ (resolved relative to this file, so it works
# no matter which folder you run the script from)


DATA_DIR = Path(__file__).resolve().parent / "data"
nutrition_path = DATA_DIR / "nutrition.csv"
exercises_path = DATA_DIR / "Exercises_clean.csv"
nutri_data = pd.read_csv(nutrition_path)
exe_data = pd.read_csv(exercises_path)

# Then the dataset needs to be cleaned, this makes the
# data easier for the ML/DL models to train and test on:
# Drop the leftover index column if the CSV has one, and make the
# body part column name consistent ("Body Part" -> "BodyPart"):
exe_data = exe_data.drop(columns=['Unnamed: 0'], errors='ignore')
exe_data = exe_data.rename(columns={'Body Part': 'BodyPart'})
exe_data['Name'] = exe_data['Name'].str.strip().str.lower()
exe_data['Type'] = exe_data['Type'].str.strip().str.lower()
exe_data['BodyPart'] = exe_data['BodyPart'].str.strip().str.lower()
exe_data['Equipment'] = exe_data['Equipment'].str.strip().str.lower()
exe_data['Difficulty'] = exe_data['Difficulty'].str.strip().str.lower()
exe_data['Equipment'] = exe_data['Equipment'].fillna('Bodyweight')
nutri_data['name'] = nutri_data['name'].str.strip().str.lower()
nutri_data['serving'] = nutri_data['serving'].str.strip().str.lower()
nutri_data['calories'] = pd.to_numeric(nutri_data['calories'], errors='coerce')
nutri_data['fat'] = nutri_data['fat'].str.strip().str.lower()
nutri_data['protein'] = nutri_data['protein'].str.strip().str.lower()
nutri_data['carbs'] = nutri_data['carbs'].str.strip().str.lower()

# Now that is out of the way, the model training can begin. Firstly,
# 'Name' and 'Type' must be dropped as they are strings. (These will
# be stored so that they are available for recommendations later.)
exercise_names = exe_data[['Name', 'Type']]
exe_data_encoded = exe_data.drop(columns=['Name', 'Type'], errors='ignore')

# After that, the data should be one-hot encoded and split into X and Y,
# this converts the categories into numerical data to prepare it for
# testing and training, ultimately improving the model's performance:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
exe_data_encoded['Difficulty'] = le.fit_transform(exe_data_encoded['Difficulty'])

exe_data_encoded = pd.get_dummies(exe_data_encoded, columns=['Equipment', 'BodyPart'])

exe_data_encoded = exe_data_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)

X = exe_data_encoded.drop(columns=['Difficulty'])
y = exe_data_encoded['Difficulty']

# After encoding, it is a good idea to visualise the feature list.
# This will also be beneficial when creating the front end as it
# can be copied directly into the Python script for Streamlit:
print("Feature list for the ML/DL moDELS")
print(list(X.columns))

# Now that the code is cleaned, normalised and encoded, the models
# can begin development, the first being a Random Forest Classifier.

# First the features and target needs to be identified for classification,
# then we can ensure the values are numeric and then initialise them:
x = exe_data_encoded.drop(columns=['Difficulty'])  # Features
y = exe_data_encoded['Difficulty']  # Target

x = x.apply(pd.to_numeric, errors='coerce').fillna(0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=25)

rf_model = RandomForestClassifier(n_estimators=100, random_state=25)
rf_model.fit(x_train, y_train)

# Once that is done, predictions should be made and retrived
y_pred = rf_model.predict(x_test)
predicted_exercises = exercise_names.iloc[y_test.index]
# This should match the test indices 👆

# (Used for Streamlit front end later on)
# import joblib
# joblib.dump(rf_model, "rf_model.pkl")

# For visualisation of the model, here are some random
# exercises that have been classified:
print("\nSample Recommended Exercises:")
print(predicted_exercises.head(10))

# Now that we have showcased supervised learning in the
# form of RF to classify the dataset, Linear Regression
# offers a different method  of predicting exercise difficutly:

# Importing the necessary libraries for the LR model:
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# First the dataset needs scaling and splitting
# into training and testing sets:
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=25
)

# After this training of the the Linear Regression model
# can begin:
model = LinearRegression()
model.fit(X_train, y_train)

# Used to connect to Streamlit later
#import joblib
#lr_model = LinearRegression()
#lr_model.fit(X_train, y_train)
#joblib.dump(lr_model, "lr_model.pkl")

# Now the model has been trained,predictions can be
# made and visualised:
y_pred = model.predict(X_test)
print("Actual:", list(y_test[:10]))
print("Predicted:", list(y_pred[:10]))

# For better visualisation and analysis, the actual and
# predicted values can be compared in a simple bar chart:
indices = np.arange(10)
y_test_values = np.array(y_test[:10])
y_pred_values = np.array(y_pred[:10])
plt.figure(figsize=(10, 5))
bar_width = 0.4
plt.bar(indices, y_test_values, width=bar_width, color='blue', label='Actual')
plt.bar(indices + bar_width, y_pred_values, width=bar_width, color='red', label='Predicted')
plt.xlabel('Sample Exercises')
plt.ylabel('Difficulty (Encoded)')
plt.title('LR Model: Actual vs Predicted')
plt.xticks(indices, [f"Ex {i}" for i in range(10)])
plt.legend()
plt.show()

# MSE was the chosen metric for even further evaluation:
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse:.3f}")

# Now the project has two examples of supervised learning models, the
# development of the unsupervised Artificial Neural Networks (ANN's)
# is the next step to improving the recommendations given by TrainR:

# Importing the necessary libraries for the exercise
# ANN development:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


# Firstly, similar to the ML models, the features being used should be
# normalised and split into test and training sets,:
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# Using early stopping here helps to prevent overfitting 👆

# Next the model can begin to be built and trained:
TrainR_Exercises = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.3),                             # This model has 3 layers, including dropout and batch normalisation.
    Dense(64, activation='relu'),             # which is used to help the model learn better whilst remaining stable
    BatchNormalization(),                     # and also reduces the chance of overfitting.
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

optimizer = Adam(
    learning_rate=0.0005,  # The model also uses an Adam Optimiser to compile the model,
    beta_1=0.9,            # this increases the stability even more since there will
    beta_2=0.999,          # be fewer divide by zero errors.
    epsilon=1e-07
)
TrainR_Exercises.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# Now the model is built it is ready to be trained and evaluated
TrainR_Exercises.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2, callbacks=[early_stop])
loss, accuracy = TrainR_Exercises.evaluate(X_test, y_test)
print(f"Exercise ANN Accuracy: {accuracy:.2f}")

TrainR_Exercises.save("exercise_ann.h5")
y_pred_probs = TrainR_Exercises.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)


# For deeper analysis and model visualisation, both a classification
# report and a confusion matrix are great tools:
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

# Now that the exercise ANN has been created, we can follow a similar approach
# to build and train the nutrition ANN, which will predict calories from protein,
# carbs and fats. This gives TrainR its food-based recommendations:

#Importing the necessary libraries for the nutrition ANN development:
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# First, the macro nutrient data must be cleaned and preprocessed, in this case
# this means removing the grams (g) in the strings and converting everything
# to a float so that it is numerical and can be processed better:
nutri_data['protein'] = nutri_data['protein'].astype(str).str.replace('g', '', regex=False).astype(float)
nutri_data['fat'] = nutri_data['fat'].astype(str).str.replace('g', '', regex=False).astype(float)
nutri_data['carbs'] = nutri_data['carbs'].astype(str).str.replace('g', '', regex=False).astype(float)


# Then the input and targets can be defined, then scaled and split into
# training and testing sets for the ANN to work on:
X_nutri = nutri_data[['protein', 'fat', 'carbs']]
y_nutri = nutri_data['calories']

scaler_nutri = MinMaxScaler()
X_nutri_scaled = scaler_nutri.fit_transform(X_nutri)
y_nutri_scaled = MinMaxScaler().fit_transform(y_nutri.values.reshape(-1, 1))

X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_nutri_scaled, y_nutri_scaled, test_size=0.1, random_state=42)

# Now the data is in the correct forma, the calorie predicting
# model can be built, compiled and trained:
TrainR_Nutrition = Sequential([
    Dense(64, input_dim=X_train_n.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

TrainR_Nutrition.compile(loss='mse', optimizer='adam', metrics=['mae'])
TrainR_Nutrition.fit(X_train_n, y_train_n, epochs=30, batch_size=16, validation_split=0.2)

# After that the model can be evaluated
loss_n, mae = TrainR_Nutrition.evaluate(X_test_n, y_test_n)
print(f"Nutrition ANN Accuracy: {accuracy:.2f}")

mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse:.3f}")

# Necessary Downloads for the Streamlit UI later:
#import joblib
#joblib.dump(scaler_nutri, 'input_scaler.pkl')
#joblib.dump(target_scaler, 'output_scaler.pkl')
#from google.colab import files
#files.download('input_scaler.pkl')
#files.download('output_scaler.pkl')
#TrainR_Nutrition.save("nutrition_ann.keras")
#from google.colab import files
#files.download("nutrition_ann.keras")

# Scale the target and save the scaler for later use.
target_scaler = MinMaxScaler()
y_nutri_scaled = target_scaler.fit_transform(y_nutri.values.reshape(-1, 1))

# Predict and unscale the results for better interpretation
y_pred_n = TrainR_Nutrition.predict(X_test_n)
y_test_unscaled = target_scaler.inverse_transform(y_test_n)
y_pred_unscaled = target_scaler.inverse_transform(y_pred_n)


# For even further evaluation, we can visualise the
# predicted and actual calories
plt.figure(figsize=(8, 4))
plt.scatter(y_test_unscaled, y_pred_unscaled, alpha=0.6, color='blue')
plt.plot([y_test_unscaled.min(), y_test_unscaled.max()], [y_test_unscaled.min(), y_test_unscaled.max()], 'r--')  # reference line
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title('Nutrition ANN: Actual vs Predicted Calories')
plt.tight_layout()
plt.show()

# Now that both supervised and unsupervised models are working,
# the final enhancement to TrainR is introducing reinforcement learning (RL),
# which will help generate more intelligent, situation-based recommendations.

# Importing the required libraries for Q-learning:
import numpy as np
import random
import matplotlib.pyplot as plt
import os
# from google.colab import files  # Colab-only; not available in VS Code

# First on the list for the RL's development, the environment needs to be set up.
# Each of the states are defined by either energy, soreness or progress,
# and each of them range from 0 to 2:
states = [(e, s, p) for e in range(3) for s in range(3) for p in range(3)]
actions = ["increase_calories", "decrease_calories", "rest", "train_hard", "maintain"]
state_to_index = {state: idx for idx, state in enumerate(states)}

# Then the Q-table is created to store the expected reward of taking an action
# in any of the given states:
Q_table = np.zeros((len(states), len(actions)))

# To make sure the model progresses over time the hyperparameters
# for learning need to be tuned, they all play an important role:
alpha = 0.1         # 👈 Alpha decides the learning rate.
gamma = 0.9         # 👈 Gamma decides the reward discount.
epsilon = 1.0       # 👈 Epsilon ensures it starts with full exploration.
epsilon_decay = 0.995
epsilon_min = 0.1
episodes = 500      # 👈 This is how many total training episodes there are.
rewards = []        # 👈 This stores the progress of rewards in a list.

# After the Q-Table is created a function is needed that determines
# a reward for each state & action pair:
def get_reward(state, action):
    energy, soreness, progress = state

    if action == "train_hard":
        if energy == 2 and soreness == 0:
            return 10
        elif soreness == 2:
            return -10
        else:
            return 2
    elif action == "rest":
        return 8 if soreness == 2 else -2
    elif action == "increase_calories":
        return 5 if energy < 2 else -3
    elif action == "decrease_calories":
        return 4 if energy == 2 else -4
    elif action == "maintain":
        return 1
    return 0

# Following that there also needs to be a function that simulates
# how the environment will react to a specific action:
def simulate_next_state(state, action):
    energy, soreness, progress = list(state)

    if action == "train_hard":
        energy = max(0, energy - 1)
        soreness = min(2, soreness + 1)
        progress = min(2, progress + 1)
    elif action == "rest":
        soreness = max(0, soreness - 1)
        energy = min(2, energy + 1)
    elif action == "increase_calories":
        energy = min(2, energy + 1)
    elif action == "decrease_calories":
        energy = max(0, energy - 1)
    # maintain doesn't change the state

    return (energy, soreness, progress)

# Now both functions and the Q-Table has been developed, it can be trained
# using the Q-learning algorithm, this is especially important for recommendations:
for episode in range(episodes):
    state = random.choice(states)
    total_reward = 0

    for _ in range(10):
        state_idx = state_to_index[state]

        if random.random() < epsilon:
            action_idx = random.randint(0, len(actions) - 1)
        else:
            action_idx = np.argmax(Q_table[state_idx])
        action = actions[action_idx]

        reward = get_reward(state, action)
        next_state = simulate_next_state(state, action)
        next_state_idx = state_to_index[next_state]

        current_value = Q_table[state_idx, action_idx]
        future_value = np.max(Q_table[next_state_idx])
        Q_table[state_idx, action_idx] = current_value + alpha * (reward + gamma * future_value - current_value)

        state = next_state
        total_reward += reward

    # After learning, the model should gradually reduce its exploration
    # this is called epsilon decay.
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    rewards.append(total_reward)

    # To visualise progress, this showcases the rewards
    # after every 50 episodes:
    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

# Finally, for  even deeper insights into how the reward system improved over
# training we can use a moving average on a scatter plot:
def moving_average(data, window=20):
    return np.convolve(data, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(moving_average(rewards), label="Smoothed Total Reward")
plt.title("RL Model Learning Progress")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# This is to save the Q-table and file downloads so they can
# be used inside the Streamlit interface later:
#os.makedirs("model", exist_ok=True)
#np.save("model/rl_q_table.npy", Q_table)
# Allow download if running in Colab:
# files.download("model/rl_q_table.npy")