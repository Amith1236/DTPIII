from flask import Flask, render_template, request, jsonify
from ml_model import feature_columns, target_columns, scatterplot_data, run_regression

app = Flask(__name__)

@app.route("/")
def index():
    # Scatterplot page
    return render_template("index.html")

@app.route("/regression")
def regression():
    # Regression page
    return render_template("regression.html")

@app.route("/get_columns", methods=["GET"])
def get_columns():
    # Provide feature and target columns to frontend
    return jsonify({"features": feature_columns, "targets": target_columns})

@app.route("/scatterplot", methods=["POST"])
def scatterplot():
    # Scatterplot data
    x_col = request.json["x_col"]
    y_col = request.json["y_col"]
    scatter_data = scatterplot_data(x_col, y_col)  # Handles logic in ml_model.py
    return jsonify(scatter_data)

@app.route("/run_regression", methods=["POST"])
def run_regression_route():
    # Run regression based on selected features and target
    selected_features = request.json["features"]
    target_variable = request.json["target"]
    metrics, equation, cost_graph = run_regression(selected_features, target_variable)
    return jsonify({"metrics": metrics, "equation": equation, "cost_graph": cost_graph})

if __name__ == "__main__":
    app.run(debug=True)
