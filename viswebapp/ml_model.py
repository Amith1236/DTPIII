#inital imports

import base64
from typing import TypeAlias
from typing import Optional, Any

Number: TypeAlias = int | float

import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import seaborn as sns
from io import BytesIO
import os
#from IPython.display import display

##Import Data
from data_cleaning import merged_df




def get_features_targets(df: pd.DataFrame,
                         feature_names: list[str],
                         target_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_feature = df.loc[:, feature_names]
    df_target = df.loc[:, target_names]
    return df_feature, df_target

#Split features and targets between test and training sets
def split_data(df_feature: pd.DataFrame, df_target: pd.DataFrame,
               random_state: Optional[int]=None,
               test_size: float=0.5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if random_state is not None:
        np.random.seed(random_state)

    total_size = len(df_feature)
    test_data_size = int(total_size * test_size)

    test_indices = np.random.choice(df_feature.index, size=test_data_size, replace=False)

    df_feature_test = df_feature.loc[test_indices]
    df_feature_train = df_feature.drop(test_indices)

    df_target_test = df_target.loc[test_indices]
    df_target_train = df_target.drop(test_indices)

    return df_feature_train, df_feature_test, df_target_train, df_target_test


#normalise column of data
def normalize_z(array: np.ndarray, columns_means: Optional[np.ndarray]=None,
                columns_stds: Optional[np.ndarray]=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if columns_means is None:
        columns_means = np.mean(array, axis=0)
    if columns_stds is None:
        columns_stds = np.std(array, axis=0)

    #accounting for divide by 0 error
    columns_stds = np.where(columns_stds == 0, 1, columns_stds)

    out = (array - columns_means) / columns_stds

    return out, columns_means, columns_stds

# Essential Functions

#Cost Function
def calc_linreg(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.matmul(X, beta)

def compute_cost_linreg(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    y_hat = calc_linreg(X, beta)
    m = X.shape[0]
    J = (1/(2*m))*calc_linreg((np.transpose(y_hat - y)), (y_hat - y))
    #print(f"J is {J}")
    return np.squeeze(J)

def prepare_feature(np_feature: np.ndarray) -> np.ndarray:
    one_column = np.ones((np_feature.shape[0], 1))
    return np.concatenate((one_column, np_feature), axis=1)



#gradient descent
def gradient_descent_linreg(X: np.ndarray, y: np.ndarray, beta: np.ndarray,
                            alpha: float, num_iters: int) -> tuple[np.ndarray, np.ndarray]:
    b_hat = beta
    m = X.shape[0]
    J_storage = np.array([compute_cost_linreg(X, y, beta)])
    for step in range(num_iters):
        b_hat -= (alpha / m) * np.matmul(np.transpose(X) ,(np.matmul(X, b_hat) - y))
        J = compute_cost_linreg(X, y, b_hat)
        J_storage = np.append(J_storage, J)
    return beta, J_storage

#Visualisation
def predict_linreg(array_feature: np.ndarray, beta: np.ndarray,
                   means: Optional[np.ndarray]=None,
                   stds: Optional[np.ndarray]=None) -> np.ndarray:
    df = pd.DataFrame(array_feature)
    norm_features, _, _ = normalize_z(df, means, stds)
    prepped_features = prepare_feature(norm_features)
    y_hat = calc_linreg(prepped_features, beta)
    return y_hat

#Running the model

feature_columns = ["TotalOutbound", "TotalInbound", "EventIndex", "TotalExp", "TotalFoodExp", "TotalFoodImports", "InflationRate"]
target_columns = ["FBSIndex", "RicePrice",	"MealPrice"]


# put Python code to test & evaluate the model

#r^2 score
def r2_score(y: np.ndarray, ypred: np.ndarray) -> float:
    mean = np.mean(y)
    SS_res = np.sum((y - ypred)**2)
    SS_tot = np.sum((y - mean)**2)
    r2 = 1 - SS_res/SS_tot
    return r2

def evaluate_model_metrics(y: np.ndarray, ypred: np.ndarray, n_features: int) -> dict:
    """
    Evaluates the performance of a regression model using multiple metrics.

    Args:
    y (np.ndarray): True target values.
    ypred (np.ndarray): Predicted target values.
    n_features (int): Number of independent variables (features).

    Returns:
    dict: Dictionary containing MSE, MAE, and Adjusted R².
    """
    print("y:")
    print(y)
    print("Ypred:")
    print(ypred)
    print("n_features")
    print(n_features)
    # Mean Squared Error
    #https://www.sciencedirect.com/topics/engineering/mean-square-error
    mse = np.mean((y - ypred) ** 2)

    # Mean Absolute Error
    # Reference: https://www.sciencedirect.com/topics/engineering/mean-absolute-error
    mae = np.mean(np.abs(y - ypred))

    # Adjusted R²
    # Reference: https://www.datacamp.com/tutorial/adjusted-r-squared
    n = len(y)
    r2 = r2_score(y, ypred)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))

    return {"MSE": mse, "MAE": mae, "AdjustedR2": adjusted_r2}



######Functions for routes
# Ensure a directory for temporary plot storage
PLOT_DIR = "static/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Generate scatter plot with regression line
def generate_scatterplot(x_col, y_col):

    # Data for regression
    x = merged_df[x_col].values
    y = merged_df[y_col].values

    # Fit a simple linear regression
    coef = np.polyfit(x, y, 1)  # Returns [slope, intercept]
    y_pred = coef[0] * x + coef[1]

    # Calculate R²
    r2 = r2_score(y, y_pred)

    # Generate the plot
    plt.figure(figsize=(8, 5))
    sns.regplot(x=x, y=y, line_kws={"color": "red"}, ci=None)
    plt.title(f"Scatter Plot with Regression Line: {x_col} vs {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    # Display R² on the plot
    plt.text(
        0.05, 0.95, f"$R^2 = {r2:.3f}$",
        transform=plt.gca().transAxes,  # Coordinates relative to the axis
        fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5)
    )

    # Save the plot as an image in the static directory
    image_path = os.path.join(PLOT_DIR, f"{x_col}_vs_{y_col}.png")
    plt.savefig(image_path)
    plt.close()

    # Return the relative path for the frontend to load the image
    return f"/static/plots/{x_col}_vs_{y_col}.png"

# Run regression
def run_regression(selected_features, target_variable):
    beta_n = len(selected_features) + 1
    
    # get features and targets from data frame
    df_feature, df_target = get_features_targets(merged_df, selected_features , target_variable)

    # split the data into training and test data sets
    df_feature_train, df_feature_test, df_target_train, df_target_test = split_data(df_feature, df_target, 100, 0.3)


    # normalize the feature using z normalization
    array_feature_train_z, means, stds = normalize_z(df_feature_train.to_numpy())

    X: np.ndarray = prepare_feature(array_feature_train_z)
    target: np.ndarray = df_target_train.to_numpy()

    iterations: int = 1500
    alpha: float = 0.01
    beta: np.ndarray = np.zeros((beta_n,1))

    # print(X.shape)
    # print(target.shape)
    # print(beta.shape)

    # call the gradient_descent function
    beta, J_storage = gradient_descent_linreg(X, target, beta, alpha, iterations)

    # call the predict method to get the predicted values
    pred: np.ndarray = predict_linreg(df_feature_test.to_numpy(), beta, means, stds)

    # Metrics
    n_features = df_feature_train.shape[1]  
    metrics = evaluate_model_metrics(df_target_test.to_numpy(), pred, n_features)
    #print(metrics)
    # Equation
    equation = "y = "
    # Add the intercept
    equation += f"{beta[0][0]:.4f}"
    if len(selected_features) > 0:
        equation += " + " + " + ".join([f"{beta[i+1][0]:.4f} * {selected_features[i]}" for i in range(len(selected_features))])
      

    # Plot cost graph
    cost_graph = base64.b64encode(plot_cost_graph(J_storage)).decode("utf-8")

    return metrics, equation, cost_graph, beta.tolist(), means.tolist(), stds.tolist()

def plot_cost_graph(J_storage):
    fig, ax = plt.subplots()
    ax.plot(J_storage)
    ax.set_title("Cost Over Iterations")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cost")
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img.read()