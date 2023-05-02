from flask import Flask,render_template,url_for,request
import modules

app = Flask(__name__)


@app.route('/')
def index():
    # breakfast,lunch, dinner = modules.recommend(70,168)
    return render_template("base.html")

@app.route('/submit', methods=['POST'])
def submit():
    weight = float(request.form['weight'])
    height=float(request.form["height"])
    breakfast,lunch, dinner,val = modules.recommend(weight,height)
    return render_template("index.html",breakfast=breakfast,lunch=lunch,dinner=dinner, val=val,height=height,weight=weight)


if __name__ == "__main__":
    app.run(debug=True)