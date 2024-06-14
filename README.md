# Disease Progression Modeling using LSTM

## Overview

This repository contains an implementation of a Long Short-Term Memory (LSTM) network to model disease progression. The goal is to predict future health metrics of patients based on historical data, helping in understanding and forecasting the progression of diseases.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Disease progression modeling is an important application of machine learning in healthcare. By predicting the future values of patient health metrics, healthcare providers can better understand the course of a disease and make informed decisions about treatment and care. This project demonstrates how LSTM networks can be used to predict these future values based on historical patient data.

## Installation

To run the code in this repository, you'll need to have Python installed along with the following libraries:
- NumPy
- Pandas
- scikit-learn
- TensorFlow

You can install the required libraries using pip:

pip install numpy pandas scikit-learn tensorflow

## Usage
1. Clone the repository:

git clone https://github.com/yourusername/disease-progression-lstm.git
cd disease-progression-lstm

2. Run the disease_progression.py script:

python disease_progression.py

### Code Explanation

## Import Libraries
The necessary libraries are imported, including NumPy for numerical operations, Pandas for data manipulation, scikit-learn for preprocessing, and TensorFlow for building the LSTM model.

## Data Generation
Sample data is generated for illustration purposes. This simulates health metrics of 100 patients over 30 timesteps.

## Data Preprocessing
The data is scaled using MinMaxScaler to bring all values into the range [0, 1]. The scaled data is reshaped to be suitable for LSTM input, i.e., [samples, time steps, features].

## Train-Test Split
The data is split into training and testing sets. 80% of the data is used for training, and 20% is used for testing.

## LSTM Model
An LSTM model is built using TensorFlow's Keras API. The model consists of two LSTM layers followed by a dense layer. The model is compiled with the Adam optimizer and mean squared error (MSE) loss function.

## Model Training
The model is trained on the training data for 10 epochs with a batch size of 32. Validation is performed on the testing data.

## Prediction
The trained model is used to predict the future health metrics for the test patients. The predicted values are then inverse transformed to their original scale.

## Output
The predicted disease progression metrics for the test patients are printed to the console.

## Contributing
Contributions are welcome! If you have any improvements or suggestions, please create a pull request or open an issue to discuss them.

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/JaCar-868/Disease-Progression/blob/main/LICENSE) file for more details.
