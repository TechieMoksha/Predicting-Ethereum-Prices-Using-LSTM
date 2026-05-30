import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request

from model import predict_future, create_steps

# flask app
app = Flask(__name__)

# dataset path
DATASET_PATH = r'C:\Users\DELL\OneDrive\Desktop\predicting_ethereum_prices_using_LSTM\ETH_1H.csv'

# loading dataset
df = pd.read_csv(
    DATASET_PATH,
    parse_dates=['Date'],
    index_col=['Date']
)

df = df.sort_index()


@app.route("/", methods=['GET', 'POST'])
def home():

    name = None

    if request.method == 'POST':

        try:

            # date from form
            to_date = request.form['td']

            # future steps
            steps = create_steps(to_date)

            # validation
            if steps <= 0:
                return render_template(
                    'index.html',
                    name='Please enter a date after 2020/04/16'
                )

            # prediction
            new = predict_future(to_date)

            if len(new) == 0:
                return render_template(
                    'index.html',
                    name='No future predictions generated.'
                )

            # backend
            plt.switch_backend('Agg')

            # figure
            plt.figure(
                figsize=(20, 7),
                facecolor='black',
                dpi=300
            )

            ax = plt.axes()
            ax.set_facecolor("black")

            # historical data length
            past_len = len(df.Close.values)

            # historical data
            plt.plot(
                range(past_len),
                df.Close.values,
                linewidth=1,
                color='white',
                label='Past Data'
            )

            # future x-axis
            future_x = range(
                past_len,
                past_len + len(new)
            )

            # future prediction
            plt.plot(
                future_x,
                new,
                linewidth=4,
                color='red',
                label='Future Prediction'
            )

            # joining line
            plt.plot(
                [past_len - 1, past_len],
                [df.Close.values[-1], new[0]],
                color='yellow',
                linewidth=2
            )

            plt.xlabel("Time", color='white')
            plt.ylabel("Ethereum Price", color='white')

            plt.title(
                f'Ethereum Price Prediction till {to_date}',
                color='white',
                fontsize=16
            )

            plt.grid(alpha=0.2)
            plt.legend()

            # save image
            plt.savefig(
                'static/output.png',
                bbox_inches='tight'
            )

            plt.close()

            name = (
                f'Prediction completed successfully. '
                f'Future hours predicted: {steps}'
            )

        except Exception as e:

            name = f"Error: {str(e)}"

    return render_template(
        'index.html',
        name=name
    )


if __name__ == "__main__":
    app.run(debug=True)