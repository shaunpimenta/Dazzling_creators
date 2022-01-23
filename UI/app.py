import numpy as np
from turtle import heading
from flask import Flask, render_template, request
import pandas as pd
import pickle
app = Flask(__name__)
# headings = ("Transection ID","Balance", "Amount", "Debited", "OldBalance")
# data = (
#     ("89000", "56000", "4000", "100000"),
#     ("10000", "56000", "4000", "12000"),
#     ("890", "56000", "4000", "100000"),
#     ("19000", "56000", "4000", "100000")
# )
# data{
#     "Transection ID":,
#     "Balance": ,
#     "Amount": ,
#     "Debited": ,
#     "OldBalance":
# }

df = pd.read_csv("AIML Dataset.csv")
df.drop(["isFraud"], axis="columns")
# html=df.to_html()
model = pickle.load(open("FraudPredictor.pkl", "rb"))


@app.route("/")
def hello():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = df[df.nameOrig == int_features[0]]

    prediction = model.predict(final_features)

    headings = ("Transection ID", "Balance", "Amount", "Debited", "OldBalance")
    data = (
        (int_features[0], final_features[1], final_features[2],
         final_features[3], final_features[4])
    )
    #output = round(prediction[0], 2)

    return render_template('table.html', prediction_text=' Temperature (C) {}'.format(prediction), headings=headings, data=data)


# def data():
#     return render_template('index.html', data=' Temperature (C) {}'.format(100))

# @app.route("/sub",methods=['POST'])
# def submit():
#     # html to .py
#     if request.method == "POST":
#         name=request.form["username"]
#     # py to html
#     return render_template("table.html",n=name)


if __name__ == "__main__":
    app.run(debug=True)
