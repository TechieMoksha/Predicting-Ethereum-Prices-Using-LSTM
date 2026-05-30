import numpy as np
import pickle
import keras

print("Checking project files...\n")

# Check model
try:
    model = keras.models.load_model(
        "model.h5",
        compile=False
    )
    print("✅ model.h5 loaded successfully")
except Exception as e:
    print("❌ model.h5 error:", e)

# Check scaler
try:
    with open("scaler.pkl", "rb") as f:
        sc = pickle.load(f)

    print("✅ scaler.pkl loaded successfully")
except Exception as e:
    print("❌ scaler.pkl error:", e)

# Check window.npy
try:
    windows_sc = np.load("window.npy")

    print("✅ window.npy loaded successfully")
    print("Shape:", windows_sc.shape)
    print("Length:", len(windows_sc))

except Exception as e:
    print("❌ window.npy error:", e)

# Check target.npy
try:
    target_sc = np.load("target.npy")

    print("✅ target.npy loaded successfully")
    print("Shape:", target_sc.shape)
    print("Length:", len(target_sc))

except Exception as e:
    print("❌ target.npy error:", e)

print("\nAll checks completed.")