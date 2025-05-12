## Disease Progression LSTM Model

This repository contains an implementation of a Long Short-Term Memory (LSTM) network to model disease progression. The goal is to predict future health metrics of patients based on historical data, helping in understanding and forecasting the progression of diseases.

## Key Features

Sliding-Window Sequence Generation: Builds input/output pairs for forecasting using a configurable window size.

Global Scaling: Applies a single MinMaxScaler fit to the entire training dataset for consistent feature ranges.

Modular Code Structure: Functions for data loading & preprocessing, model building, training, evaluation, and plotting.

Reproducibility: Sets random seeds for NumPy and TensorFlow; prints version information.

Training Callbacks: Uses EarlyStopping (with patience) and ModelCheckpoint to save best model weights.

Performance Metrics & Visualization: Tracks MSE and MAE during training; includes plots of loss and MAE.

Model Persistence: Saves the best model to best_model.h5 and demonstrates loading and prediction.

Example Predictions: Prints sample actual vs. predicted values (inverse-transformed).

## Requirements

Python 3.7+

NumPy

TensorFlow 2.x

scikit-learn

matplotlib

Install via:

pip install -r requirements.txt

## Usage

python Disease\ Progression.py

Or:

python3 disease_progression.py

## Script Breakdown

create_sequences(data, window_size)Creates sliding-window sequences for forecasting.

load_data(...)Generates or loads raw data, applies global scaling, creates sequences, and performs train/test split.

build_model(input_shape)Defines the stacked LSTM architecture.

train_model(model, X_train, y_train, X_val, y_val, ...)Trains the model with EarlyStopping and ModelCheckpoint.

plot_metrics(history)Plots training/validation loss and MAE over epochs.

main()Orchestrates version logging, data prep, model training, evaluation, and displays sample predictions.

## Configuration

Adjust hyperparameters at the top of the script or within the main function:

num_patients, timesteps, n_features: Data shape.

window_size: Number of past timesteps for input.

epochs, batch_size: Training parameters.

## Results

After running, you’ll see:

Printed versions of NumPy, TensorFlow, and scikit-learn.

Training and validation loss/MAE plots.

Final test loss & MAE.

Sample actual vs. predicted metrics.

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/JaCar-868/Disease-Progression/blob/main/LICENSE) file for more details.
