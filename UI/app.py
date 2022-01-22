from turtle import heading
from flask import Flask,render_template

app=Flask(__name__)
headings=("Balance","Amount","Debited","OldBalance")
data=(
    ("89000","56000","4000","100000"),
     ("10000","56000","4000","12000"),
      ("890","56000","4000","100000"),
       ("19000","56000","4000","100000")
)
@app.route("/")
def hello():
    return render_template('table.html',headings=headings,data=data)


if __name__ == "__main__":
    app.run(debug=True)