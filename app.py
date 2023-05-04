from flask import Flask,render_template,url_for,request
import modules
import random_module as rm
app = Flask(__name__)


@app.route('/')
def index():
    # breakfast,lunch, dinner = modules.recommend(70,168)
    return render_template("base.html")

@app.route('/method1submit', methods=['POST'])
def method1():
    value=request.form["recommendation"]
    if value=="1":
        weight = float(request.form['weight'])
        height=float(request.form["height"])
        breakfast,lunch, dinner,val = modules.recommend(weight,height)
        return render_template("method1.html",breakfast=breakfast,lunch=lunch,dinner=dinner, val=val,height=height,weight=weight)
    else: 
        weight = float(request.form['weight'])
        height=float(request.form["height"])
        food_items = rm.main_fun(height,weight)
        # print(food_items)
        bmi=float(float(weight)/(float(height/100.0))**2)
        return render_template("method2.html",item=food_items,height=height,weight=weight,bmi=bmi)


if __name__ == "__main__":
    app.run(debug=True)