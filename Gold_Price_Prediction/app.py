import flask
from flask import jsonify, render_template, request
import os
import utilis

models = {
    "fbprophet": utilis.FBProphetPredictor,}

app = flask.Flask(__name__)
app.config["debug"] = True

@app.route("/")
def home():
    return render_template("index.html", models=list(models.keys()))

@app.route("/predict", methods=['GET', 'POST'])
def predict_gold():
    """
    Given the date, predict the gold price for next date
    """
    # print(request.form.get)
    try:
        model_name = request.form.get("model_name")
        date_given = request.form.get("date")
        model = models[model_name]()
        pred = model.predict(date_given)
        return render_template('index.html', result=pred[0])
    except KeyError:  # get value from curl header
        model_name = request.headers.get("model_name")
        date_given = request.headers.get("date")
        model = models[model_name]()
        pred = model.predict(date_given)
        return jsonify(
        {
            "given_date": date_given,
            "next_date": model.get_next_date(date_given),
            "price": pred,
        },
    )
    

if __name__=="__main__":
    app.run(host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 8000)))