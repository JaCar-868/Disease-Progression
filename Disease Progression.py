import numpy as np
import tensorflow as tf
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_sequences(data, window_size):
    """
    Build sliding-window sequences for forecasting.
    Args:
        data: np.array, shape=(n_patients, timesteps, features)
        window_size: int, number of past timesteps to use as input
    Returns:
        X: np.array, shape=(n_samples, window_size, features)
        y: np.array, shape=(n_samples, features)
    """
    X, y = [], []
    n_patients, timesteps, n_features = data.shape
    for i in range(n_patients):
        for t in range(window_size, timesteps):
            X.append(data[i, t-window_size:t, :])
            y.append(data[i, t, :])
    return np.array(X), np.array(y)

def load_data(num_patients=100, timesteps=30, n_features=1, window_size=10, test_size=0.2):
    """
    Generates synthetic data, scales globally, builds sequences, and splits train/test.
    """
    # 1) Generate synthetic data
    raw = np.random.normal(0, 1, (num_patients, timesteps, n_features))

    # 2) Global scaling
    flat = raw.reshape(-1, n_features)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(flat)
    scaled = scaler.transform(flat).reshape(raw.shape)

    # 3) Sliding-window sequences
    X, y = create_sequences(scaled, window_size)

    # 4) Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )

    return X_train, X_test, y_train, y_test, scaler

def build_model(input_shape):
    """
    Builds a stacked LSTM forecasting model.
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(input_shape[1]))  # predict one timestep ahead (n_features)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Trains with EarlyStopping and ModelCheckpoint callbacks.
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2
    )
    return history

def plot_metrics(history):
    """
    Plots training & validation loss and MAE over epochs.
    """
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.title('Loss and MAE')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Print versions for reproducibility
    print(f"NumPy   : {np.__version__}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"scikit-learn: {sklearn.__version__}\n")

    # Load and prepare data
    window_size = 10
    X_train, X_test, y_train, y_test, scaler = load_data(window_size=window_size)

    # Build, train, and evaluate model
    model = build_model(input_shape=(window_size, 1))
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Plot training history
    plot_metrics(history)

    # Final evaluation
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}, Test MAE: {mae:.4f}\n")

    # Predict and inverse-transform
    y_pred = model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test)

    # Show a few sample predictions
    for i in range(5):
        actual = y_test_inv[i].flatten()
        predicted = y_pred_inv[i].flatten()
        print(f"Sample {i+1} — Actual: {actual}, Predicted: {predicted}")

if __name__ == "__main__":
    main()
