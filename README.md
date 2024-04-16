# Neural Network Regression Model

This repository contains a neural network designed to predict the total energy output from a set of features.

## Project Overview

The goal of this project is to build and train a neural network model that takes several categorical and numerical inputs to predict a continuous variable, the total energy (Ry). The data is preprocessed to handle categorical variables via one-hot encoding and to scale numerical features for optimal neural network performance.

## Repository Structure

- `train_model.py`: The main Python script to train the neural network model.
- `preprocessor.joblib`: Serialized preprocessor object containing scaling and encoding information.
- `my_model.h5`: The trained neural network model saved in HDF5 format.
- `requirements.txt`: List of dependencies to install for running the project.

## Model Architecture

The neural network has the following architecture:
- Input Layer: Matches the number of preprocessed features.
- Hidden Layers: Two hidden layers with ReLU activation.
- Output Layer: Single neuron with linear activation for regression output.

## Metrics

The model performance on the test set is as follows:
- R-squared (R2): 1.00
- Mean Squared Error (MSE): 0.81
- Mean Absolute Error (MAE): 0.61
- Root Mean Squared Error (RMSE): 0.90

These metrics indicate a high level of accuracy in the model's predictions.

## Usage

To use the pre-trained model for making predictions with new data:

1. Load the preprocessor and the model:
    ```python
    from joblib import load
    from tensorflow.keras.models import load_model

    preprocessor = load('preprocessor.joblib')
    model = load_model('my_model.h5')
    ```

2. Preprocess the new data using the loaded preprocessor and make predictions:
    ```python
    # Assuming new_data is a pandas DataFrame containing the new input features
    X_new_transformed = preprocessor.transform(new_data)
    predictions = model.predict(X_new_transformed)
    ```

## Installation

To set up the project environment:

1. Clone the repository to your local machine.
2. Install the required packages using:
    ```bash
    pip install -r requirements.txt
    ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
