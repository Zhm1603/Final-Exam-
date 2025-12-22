# 1. Project Title

Formula 1 Data Analysis and Machine Learning

# 2. Project Description

A brief overview:

Purpose: Analyze F1 historical data (1950–2024) and apply machine learning models to predict race points and classify whether a driver finishes the race.

Methods: Linear Regression, Logistic Regression, Decision Tree, Gradient Descent with Momentum, Adam optimizer.

Features used: grid position, laps, fastest lap speed, status (finished/not finished).

# 3. Dataset

Name: Formula 1 World Championship (1950–2024)

Source (if public): https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020?select=results.csv

Description: Contains driver, race, and performance statistics including lap times, positions, and points.

# 4. Features

grid — starting position of driver

laps — number of laps completed

fastestLapSpeed — speed on fastest lap

points — points earned in the race

statusId — race status (finished or not)

# 5. Installation

Instructions to set up the environment:

git clone <repo-url>
cd <project-folder>
pip install -r requirements.txt

# 6. Usage

This project uses interactive widgets in Jupyter Notebook to train and test models. You can control the input and hyperparameters using these widgets:

Upload Dataset

Use the file upload widget to load your CSV dataset (results.csv).

Click the “Choose File” button and select your CSV file.

The uploaded file will be read into a DataFrame using load_main_data().

Select Model

A dropdown menu lets you choose the model you want to train:

Linear Regression

Linear Regression (Momentum)

Logistic Regression

Decision Tree

Click the dropdown and select the desired model.

Adjust Hyperparameters

Learning Rate (LR): Use the slider to set the learning rate for gradient descent.

Epochs: Use the slider to choose the number of training iterations.

Train the Model

Click the “Train” button to start training the selected model with the chosen hyperparameters.

After training, results such as coefficients, accuracy, MSE, RMSE, R², confusion matrix, and ROC AUC will be displayed.

# 7. Models Implemented

Linear Regression — predict race points.

Linear Regression with Momentum — optimized gradient descent.

Logistic Regression (with class balancing) — predict whether driver finished.

Decision Tree Classifier — alternative classification model capturing non-linear relationships.

# 8. Results

Linear Regression: MSE, RMSE, R² values for different learning rates and epochs.

Logistic Regression: Accuracy, ROC AUC, confusion matrices.

Observations: Best hyperparameters, effect of learning rate and epochs, model limitations.

# 9. Functions Overview

train_linear_regression

train_linear_regression_momentum

sigmoid, predict_logistic, log_loss

train_logistic_regression_balanced

# 10. Notes / Observations

Linear regression limited by non-linear dependencies in data.

Logistic regression required class weighting due to imbalance.

Decision Tree performed better in capturing interactions and non-linear relationships.


# 11. Contact / Author

Marat Zhanel Borambaev Duman
