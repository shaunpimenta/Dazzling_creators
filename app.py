# from crypt import methods
from pyexpat import model
import numpy as np
from turtle import heading
import pandas as pd
import pickle

from flask import Flask, render_template, redirect, request
import pandas as pd
import keras
app = Flask(__name__)

# df=pd.read_csv("")
# model=pickel.load(open())
def windowed_dataset(series, time_step):
    dataX, dataY = [], []
    for i in range(len(series) - time_step - 1):
        a = series[i: (i + time_step), 0]
        dataX.append(a)
        dataY.append(series[i + time_step, 0])

    return np.array(dataX), np.array(dataY)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model = keras.models.load_model('lstm.h5')
# file = open("lstm.h5","rb")
# lst = pickle.load(file)
df = pd.read_csv('bitcoin.csv')
@app.route('/')
def index():
    return render_template('home.html', predict=10000, conclusion='Yes', percent=10, volume=10)


@app.route('/predict',methods=['GET','POST'])
def predict():
    # timestamp=request.form['timestamp']
    # timestamp='2011-12-31'
    # pred=df.loc[df['Timestamp'] == timestamp]
    # # df.loc[(df['col1'] == value) & (df['col2'] < value)] multiple
    # X_test, y_test = windowed_dataset(pred, time_step=100)
    # price=model.predict(X_test)
    # print(price)
    return render_template('Analysis.html')


if __name__ == "__main__":
    app.run(debug=True)

@app.route('/home')
def ret_index():
    return redirect('home.html')