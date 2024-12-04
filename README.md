# Machine Learning Model Exploration Website

## Overview
This project is a Flask-based web application designed to showcase the process of building and analyzing a machine learning regression model. It features:

1. **Scatterplot Analysis Page**: Allows users to explore the relationship between different independent variables (X) and a dependent variable (Y). Users can select variables from a dropdown to view graphs.
2. **Regression Analysis Page**: Enables users to select a dependent variable (Y) and multiple independent variables (X) to build a regression model. The application calculates metrics such as Adjusted R², MAE, and MSE, and provides the regression equation. Users can input values for the independent variables to make predictions.

The website is built with the MVC framework:
- **Flask App**: Manages initialization, routing, and data handling.
- **Static and Templates**: Contain front-end JavaScript and HTML for user interactions.
- **Backend Modules**: `ml_model.py` and `data_cleaning.py` handle model training, data processing, and analytics.

## Directory Structure
```
viswebapp/
|
|-- __pycache__          # Cached files for Python modules
|-- datasets             # Data files used for regression analysis
|-- static/              # Front-end JavaScript and CSS files
|   |-- js/
|       |-- regression.js
|       |-- scatter.js
|   |-- plots/           # Stores imgs of Regression plots
|-- templates/           # HTML templates
|   |-- index.html
|   |-- regression.html
|-- Pipfile              # Pipenv environment file
|-- Pipfile.lock         # Locked dependencies
|-- app.py               # Flask application file
|-- data_cleaning.py     # Data preprocessing functions
|-- ml_model.py          # Machine learning model logic
```

## Setup and Installation
### Prerequisites
Ensure you have the following installed on your system:
- Python 3.8+
- Pipenv

### Steps
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Set Up the Environment**:
   Use Pipenv to create a virtual environment and install dependencies:
   ```bash
   pipenv install
   ```

3. **Activate the Virtual Environment**:
   ```bash
   pipenv shell
   ```

4. **Run the Flask Application**:
   ```bash
   flask run
   ```

5. **Access the Application**:
   Open the link provided in the terminal (usually [http://127.0.0.1:5000](http://127.0.0.1:5000)) to view the website hosted locally.

## Features and Navigation

### 1. **Scatterplot Analysis**
- **Purpose**: To visualize the relationships between variables.
- **How to Use**:
  - Navigate to the scatterplot page (default home page).
  - Use the dropdown menu to select variables to plot against the dependent variable.
  - View interactive scatterplots with insights into variable correlations.

### 2. **Regression Analysis**
- **Purpose**: To build and evaluate regression models.
- **How to Use**:
  - Click on "Regression" to navigate to the regression analysis page.
  - Choose a dependent variable (Y) and one or more independent variables (X).
  - Run the regression model to view metrics (Adjusted R², MSE, MAE) and the regression equation.
  - Scroll down to input values for the selected variables to get predictions for the dependent variable.
  - Navigate back to the scatterplot page if needed for additional visual analysis.

## Technology Stack
- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Visualization**: Matplotlib, Seaborn
- **Modeling**: Python

## Acknowledgments
This project is an educational tool designed to demonstrate machine learning workflows in a user-friendly manner. Special thanks to the developers of Flask and Data Driven World SUTD for making this possible.

