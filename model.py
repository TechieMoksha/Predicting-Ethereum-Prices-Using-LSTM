import numpy as np
import pickle
import keras

from datetime import datetime
from tqdm import tqdm

# Paths
MODEL_PATH = r"C:\Users\DELL\OneDrive\Desktop\predicting_ethereum_prices_using_LSTM\model.h5"
SCALER_PATH = r"C:\Users\DELL\OneDrive\Desktop\predicting_ethereum_prices_using_LSTM\scaler.pkl"
WINDOW_PATH = r"C:\Users\DELL\OneDrive\Desktop\predicting_ethereum_prices_using_LSTM\window.npy"
TARGET_PATH = r"C:\Users\DELL\OneDrive\Desktop\predicting_ethereum_prices_using_LSTM\target.npy"

# Load model
model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Load scaler
with open(SCALER_PATH, "rb") as f:
    sc = pickle.load(f)

# Load arrays
windows_sc = np.load(WINDOW_PATH)
target_sc = np.load(TARGET_PATH)

print("window shape:", windows_sc.shape)
print("target shape:", target_sc.shape)

length = 24


def create_steps(to_date):

    start_date = datetime.strptime(
        "2020/04/16",
        "%Y/%m/%d"
    )

    end_date = datetime.strptime(
        to_date,
        "%Y/%m/%d"
    )

    delta = end_date - start_date

    return delta.days * 24


def predict_future(to_date):

    steps_in_future = create_steps(to_date)

    print("Steps:", steps_in_future)

    if steps_in_future <= 0:
        return []

    f_wind = windows_sc[-1]
    f_tar = target_sc[-1]

    predictions = []

    for _ in tqdm(range(steps_in_future)):

        curr = np.append(
            f_wind[1:],
            [f_tar]
        ).reshape(-1, 1)

        next_pred = model.predict(
            curr.reshape(1, length, 1),
            verbose=0
        )

        pred_price = sc.inverse_transform(
            next_pred
        )

        predictions.append(
            float(pred_price[0][0])
        )

        f_wind = curr
        f_tar = next_pred

    print("Generated:", len(predictions))

    return predictions