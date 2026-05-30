import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask
from flask import render_template
from flask import request

from model import predict_future
from model import create_steps

app = Flask(__name__)

DATASET_PATH = r"C:\Users\DELL\OneDrive\Desktop\predicting_ethereum_prices_using_LSTM\ETH_1H.csv"

df = pd.read_csv(
    DATASET_PATH,
    parse_dates=["Date"],
    index_col=["Date"]
)

df = df.sort_index()


@app.route("/", methods=["GET", "POST"])
def home():

    message = None

    if request.method == "POST":

        try:

            to_date = request.form["td"]

            steps = create_steps(to_date)

            if steps <= 0:

                message = (
                    "Please enter a date "
                    "after 2020/04/16"
                )

                return render_template(
                    "index.html",
                    name=message
                )

            new = predict_future(to_date)

            print("Prediction Length:", len(new))

            if len(new) == 0:

                return render_template(
                    "index.html",
                    name="No predictions generated."
                )

            plt.switch_backend("Agg")

            plt.figure(
                figsize=(18, 7),
                facecolor="black",
                dpi=300
            )

            ax = plt.gca()
            ax.set_facecolor("black")

            past_len = len(df.Close.values)

            # Historical data
            plt.plot(
                range(past_len),
                df.Close.values,
                color="white",
                linewidth=1,
                label="Historical Price"
            )

            # Future data
            future_x = range(
                past_len,
                past_len + len(new)
            )

            plt.plot(
                future_x,
                new,
                color="red",
                linewidth=3,
                label="Predicted Price"
            )

            # Connecting line
            plt.plot(
                [past_len - 1, past_len],
                [df.Close.values[-1], new[0]],
                color="yellow",
                linewidth=2
            )

            plt.title(
                f"Ethereum Price Prediction Until {to_date}",
                color="white",
                fontsize=16
            )

            plt.xlabel(
                "Time",
                color="white"
            )

            plt.ylabel(
                "Ethereum Price",
                color="white"
            )

            plt.legend()

            plt.grid(
                alpha=0.2
            )

            plt.savefig(
                "static/output.png",
                bbox_inches="tight"
            )

            plt.close()

            message = (
                f"Prediction completed for "
                f"{steps} hours."
            )

        except Exception as e:

            message = f"Error: {str(e)}"

            print(e)

    return render_template(
        "index.html",
        name=message
    )


if __name__ == "__main__":

    app.run(
        debug=True
    )