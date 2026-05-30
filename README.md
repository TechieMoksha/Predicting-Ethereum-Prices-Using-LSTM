# 🚀 Ethereum Price Prediction Using LSTM

A Deep Learning-based web application that predicts future Ethereum (ETH) prices using an LSTM (Long Short-Term Memory) Neural Network. The project uses historical Ethereum price data and provides future price forecasts through an interactive Flask web interface.

---

## 📌 Features

- Predict future Ethereum prices
- LSTM-based Deep Learning model
- Interactive Flask web application
- Historical vs Predicted Price Visualization
- User-friendly interface
- Automated graph generation

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- LSTM Neural Network
- Flask
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn

---

## 📂 Project Structure

```text
predicting_ethereum_prices_using_LSTM/
│
├── app.py
├── model.py
├── model.h5
├── scaler.pkl
├── window.npy
├── target.npy
├── ETH_1H.csv
├── test.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   ├── output.png
│   └── uploads/
│
└── ether_ml.ipynb
```

---

## ⚙️ How It Works

1. Historical Ethereum price data is collected.
2. Data is normalized using MinMaxScaler.
3. Time-series windows are created.
4. An LSTM model is trained on historical prices.
5. The trained model predicts future prices.
6. Flask provides a web interface for user interaction.
7. A prediction graph is generated showing:
   - Historical Prices (White Line)
   - Predicted Prices (Red Line)

---

## 📊 Model Architecture

```text
Input Layer
      ↓
LSTM Layer (100 Units)
      ↓
Dropout Layer (0.2)
      ↓
Dense Layer
      ↓
Predicted Ethereum Price
```

### Clone Repository

```bash
git clone https://github.com/yourusername/predicting_ethereum_prices_using_LSTM.git
cd predicting_ethereum_prices_using_LSTM
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py

---

## 📷 Project Screenshots

### Home Page
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f38e2691-a4c2-4f82-bbbb-9e9a6b80ce25" />


### Prediction Output
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/ce24cbab-77b1-463f-a2bc-655d208e707c" />


---

## 👩‍💻 Author

**Mokshada Patil**

Passionate about Data Analytics, Machine Learning, Deep Learning, and AI-based Applications.

---

⭐ If you found this project useful, don't forget to star the repository.
