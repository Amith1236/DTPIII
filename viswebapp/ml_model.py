#inital imports

import base64
from typing import TypeAlias
from typing import Optional, Any

Number: TypeAlias = int | float

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import seaborn as sns
from io import BytesIO
#from IPython.display import display



######Import data#########
###Testing Data with Outbound and Inbound against FBSI
#Outbound Departures
df1 = pd.read_csv("datasets\OutboundDeparturesOfSingaporeResidentsByModeOfTransportMonthly.csv")
'''
Independent Variable: Total outbound departures of Singaporean residents
Column Title: Total -> Rename to TotalOutbound

This data shows how many residents are traveling out of Singapore each month.
'''

#Inbound Tourism Markets
df2 = pd.read_csv("datasets\InternationalVisitorArrivalsByInboundTourismMarketsMonthly.csv")
'''
Independent Variable: Total International Visitor Arrivals By Inbound Tourism Markets
Column Title: Total International Visitor Arrivals By Inbound Tourism Markets -> Rename to TotalInbound


'''

#Consumer Price Index
df4 = pd.read_csv("datasets\ConsumerPriceIndexCPI2019AsBaseYearMonthly.csv")

'''
Independent Variable: Inflation Rate
Column Title: All Items -> Rename to CPIndex
'''

#Tourist Expenditure
df5 = pd.read_csv("datasets\TourismReceipts.csv")
'''
Independent Variable: Total Tourist Expenditure in Singapore Monthly
Column Title:
Source: https://www.singstat.gov.sg/find-data/search-by-theme/industry/tourism/latest-data
'''

#Food imports
df6 = pd.read_csv("datasets\MerchandiseImportsByCommodityDivisionMonthly.csv")

###Dependent Variables
#Food Beverage Services Index
df8 = pd.read_csv("datasets\FoodBeverageServicesIndex2017100AtCurrentPricesMonthly.csv")
'''
Dependent Variable: Food and Beverage Services Index
Column Title: Total -> Rename to FBSIndex
'''
#Clean data.gov data, method created since format for these datasets is mostly the same
def preprocess_dataframe(df, column_map, n_rows):
    """
    Preprocess a dataframe for data.gov datasets
    1. Transpose and reset index.
    2. Set first row as headers and clean them.
    3. Retain specific columns by their names.
    4. Limit the number of rows.
    5. Rename columns for clarity.
    6. Convert specified columns to numeric, handling non-numeric values.

    Args:
        df (DataFrame): The input dataframe.
        column_map (dict): A mapping of old column names to new ones (e.g., {"OldName": "NewName"}).
        n_rows (int): The number of rows to retain.

    Returns:
        DataFrame: The processed dataframe.
    """
    df = df.transpose()
    df.reset_index(inplace=True)
    df.columns = df.iloc[0]  # Set the first row as headers
    df = df[1:]  # Remove the first row
    df.columns = df.columns.str.strip()  # Clean up header names

    # Retain only the columns to keep
    columns_to_keep = list(column_map.keys())
    df = df.loc[:, columns_to_keep]  # Select columns dynamically
    df = df.rename(columns=column_map)  # Rename columns for clarity
    df = df.iloc[:n_rows]  # Limit the number of rows

    # Convert all renamed columns to numeric and handle non-numeric values
    for new_col in column_map.values():

        #Numeric not needed for months
        if new_col == "DataSeries":
          continue

        df[new_col] = pd.to_numeric(df[new_col], errors="coerce")
        df[new_col] = df[new_col].fillna(df[new_col].mean())  # Replace NaNs with the column mean

    df = df.reset_index(drop=True)
    return df


n_rows = 165  # Control how many rows of data to retain

# Outbound Departures Data
outbound_column_map = {"DataSeries": "DataSeries", "Total": "TotalOutbound"}
df1 = preprocess_dataframe(df1, outbound_column_map, n_rows)
#display(df1)

# Inbound Arrivals Data
inbound_column_map = {"Total International Visitor Arrivals By Inbound Tourism Markets": "TotalInbound"}
df2 = preprocess_dataframe(df2, inbound_column_map, n_rows)
#display(df2)

# CPI Data
cpi_column_map = {"All Items": "CPIndex"}
df4 = preprocess_dataframe(df4, cpi_column_map, n_rows + 12)
#display(df4)

# Tourist Expenditure
tourist_exp_column_map = {"Data Series": "DataSeries", "Tourism Receipts": "TotalExp", "Food & Beverage": "TotalFoodExp"}
df5 = preprocess_dataframe(df5, tourist_exp_column_map, n_rows)
#display(df5)

#Food Imports
food_imports_column_map = {"DataSeries": "DataSeries",
                           "Meat & Meat Preparations": "Meat",
                           "Fish, Seafood (Excl Marine Mammals) & Preparations" : "Seafood",
                           "Vegetables & Fruit": "Veggies",
                           "Coffee, Tea, Cocoa, Spices & Manufactures": "Coffee",
                           "Beverages" : "Drinks"}
df6 = preprocess_dataframe(df6, food_imports_column_map, n_rows + 1)
df6 = df6[1:]
df6 = df6.reset_index(drop=True)
#display(df6)

# FBS Index Data
fbs_column_map = {"Total": "FBSIndex"}
df8 = preprocess_dataframe(df8, fbs_column_map, n_rows)
#display(df8)

#############Prep d5 - quarterly tourist reports into monthly

# Insert the missing row for 2024 3Q with mean values
mean_total_exp = df5["TotalExp"].mean()
mean_total_food_exp = df5["TotalFoodExp"].mean()
missing_row = pd.DataFrame({
    "DataSeries": ["2024 3Q"],
    "TotalExp": [mean_total_exp],
    "TotalFoodExp": [mean_total_food_exp]
})

# Append the missing row to the original DataFrame
df5 = pd.concat([df5, missing_row], ignore_index=True)
df5 = df5.sort_values(by="DataSeries", ).reset_index(drop=True)  # Sort to maintain chronological order

# Create a list to hold the monthly data
monthly_data = []

# Iterate through each row in the updated DataFrame
for _, row in df5.iterrows():
    quarter = row["DataSeries"].strip()
    year, qtr = quarter.split(" ")
    year = int(year)
    qtr = int(qtr[0])  # Extract the quarter number

    # Map quarters to months
    quarter_months = {
        1: [f"{year}Jan", f"{year}Feb", f"{year}Mar"],
        2: [f"{year}Apr", f"{year}May", f"{year}Jun"],
        3: [f"{year}Jul", f"{year}Aug", f"{year}Sep"],
        4: [f"{year}Oct", f"{year}Nov", f"{year}Dec"]
    }

    # Get values for TotalExp and TotalFoodExp
    total_exp = row["TotalExp"]
    total_food_exp = row["TotalFoodExp"]

    # Calculate monthly values
    monthly_exp = total_exp / 3
    monthly_food_exp = total_food_exp / 3

    # Create rows for each month in the quarter
    for month in quarter_months[qtr]:
        monthly_data.append({
            "DataSeries": month,
            "TotalExp": monthly_exp,
            "TotalFoodExp": monthly_food_exp
        })

# Convert the monthly data to a DataFrame
df_5_q = pd.DataFrame(monthly_data)
# Convert DataSeries to datetime for proper sorting
df_5_q['Datetime'] = pd.to_datetime(df_5_q['DataSeries'], format='%Y%b')

# Sort the DataFrame by Datetime in descending order
df_5_q = df_5_q.sort_values(by='Datetime', ascending=False).drop(columns=['Datetime']).reset_index(drop=True)

# Take only n_rows number of rows
df_5_q = df_5_q.head(n_rows)

# Display the sorted DataFrame
#display(df_5_q)

##########prep df_4_i
# Calculate the inflation rate using the CPI of the same month last year (-12 rows forward)
df4['InflationRate'] = ((df4['CPIndex'] - df4['CPIndex'].shift(-12)) / df4['CPIndex'].shift(-12)) * 100

# Drop the last 12 rows where inflation rate cannot be calculated
df_4_i = df4.iloc[:-12].reset_index(drop=True)
df_4_i = df_4_i.drop(columns=["CPIndex"])

# Display the resulting DataFrame
#display(df_4_i)


###########prep df_6_t
columns_to_sum = ["Meat", "Seafood", "Veggies", "Coffee", "Drinks"]
df6[columns_to_sum] = df6[columns_to_sum].apply(pd.to_numeric, errors='coerce')

# Calculate the total for each row
df6["TotalFoodImports"] = df6[columns_to_sum].sum(axis=1)

# Create the new DataFrame with the desired columns
df_6_t = df6[["DataSeries", "TotalFoodImports"]].copy()

# Display the resulting DataFrame
#display(df_6_t)


# Concatenate all DataFrames
merged_df = pd.concat([df1, df2, df_5_q, df_6_t, df_4_i, df8], axis=1)
# Remove duplicate columns
merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
# Display the final merged DataFrame
#display(merged_df)

#prepare features

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

feature_columns = ["TotalOutbound", "TotalInbound",	"TotalExp",	"TotalFoodImports",	"InflationRate"]
target_columns = ["FBSIndex"]
beta_n = len(feature_columns) + 1

# get features and targets from data frame
df_feature, df_target = get_features_targets(merged_df, feature_columns , ["FBSIndex"])

# split the data into training and test data sets
df_feature_train, df_feature_test, df_target_train, df_target_test = split_data(df_feature, df_target, 100, 0.3)

# normalize the feature using z normalization
array_feature_train_z, means, stds = normalize_z(df_feature_train.to_numpy())

X: np.ndarray = prepare_feature(array_feature_train_z)
target: np.ndarray = df_target_train.to_numpy()

iterations: int = 1500
alpha: float = 0.01
beta: np.ndarray = np.zeros((beta_n,1))

print(X.shape)
print(target.shape)
print(beta.shape)

# call the gradient_descent function
beta, J_storage = gradient_descent_linreg(X, target, beta, alpha, iterations)

# call the predict method to get the predicted values
pred: np.ndarray = predict_linreg(df_feature_test.to_numpy(), beta, means, stds)

# Print the learned beta coefficients
print("Learned Beta Coefficients:")
print(beta)

# Plot the cost function over iterations
plt.figure(figsize=(10, 6))
plt.plot(J_storage, color='blue', linewidth=2)
plt.title("Cost Function (J) Over Iterations", fontsize=14)
plt.xlabel("Iterations", fontsize=12)
plt.ylabel("Cost (J)", fontsize=12)
plt.grid(True)
#plt.show()

# Get predictions for the test set
y_hat_test = predict_linreg(df_feature_test.to_numpy(), beta, means, stds)

# Convert arrays to DataFrame for easier plotting
df_predictions = pd.DataFrame({
    "Actual": df_target_test.squeeze(),
    "Predicted": y_hat_test.squeeze(),
    "Residuals": df_target_test.squeeze() - y_hat_test.squeeze()
})
df_features = pd.DataFrame(df_feature_test, columns=feature_columns)

# Pairwise Plot: Features, Predictions, and Actual Target
# sns.pairplot(pd.concat([df_features, df_predictions], axis=1),
#              diag_kind='kde',
#              plot_kws={"alpha": 0.7})
# plt.suptitle("Pairwise Relationships", y=1.02, fontsize=16)
# plt.show()

# Residual Plot: Residuals vs Predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_predictions["Predicted"], y=df_predictions["Residuals"], color='purple')
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title("Residuals vs Predicted", fontsize=14)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Residuals", fontsize=12)
plt.grid(True)
#plt.show()

# Prediction vs Actual Plot
plt.figure(figsize=(10, 6))
plt.scatter(df_predictions["Actual"], df_predictions["Predicted"], alpha=0.7, color='blue')
plt.plot([df_predictions["Actual"].min(), df_predictions["Actual"].max()],
         [df_predictions["Actual"].min(), df_predictions["Actual"].max()],
         color='red', linestyle='--', linewidth=2)
plt.title("Predicted vs Actual Values", fontsize=14)
plt.xlabel("Actual", fontsize=12)
plt.ylabel("Predicted", fontsize=12)
plt.grid(True)
#plt.show()

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
    # Mean Squared Error
    #https://www.sciencedirect.com/topics/engineering/mean-square-error
    mse = np.mean((y - ypred) ** 2)

    # Mean Absolute Error
    # Reference: https://www.sciencedirect.com/topics/engineering/mean-absolute-error
    mae = np.mean(np.abs(y - ypred))

    # Adjusted R²
    # Reference: https://www.datacamp.com/tutorial/adjusted-r-squared
    n = len(y)
    print(n)
    print(n_features)
    mean = np.mean(y)
    SS_res = np.sum((y - ypred) ** 2)
    SS_tot = np.sum((y - mean) ** 2)
    r2 = 1 - (SS_res / SS_tot)
    print(r2)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))

    return {"MSE": mse, "MAE": mae, "AdjustedR2": adjusted_r2}


# Number of features in your model
n_features = df_feature_train.shape[1]

# Evaluate metrics for training data
metrics = evaluate_model_metrics(df_target_train.to_numpy(), predict_linreg(df_feature_train.to_numpy(), beta, means, stds), n_features)
print("Training Metrics:", metrics)

# Evaluate metrics for test data
test_metrics = evaluate_model_metrics(df_target_test.to_numpy(), predict_linreg(df_feature_test.to_numpy(), beta, means, stds), n_features)
print("Test Metrics:", test_metrics)



######Functions for routes
# Scatterplot data
def scatterplot_data(x_col, y_col):
    scatter = merged_df[[x_col, y_col]].dropna()
    return {"x": scatter[x_col].tolist(), "y": scatter[y_col].tolist()}

# Run regression
def run_regression(selected_features, target_variable):
    # features = merged_df[selected_features]
    # target = merged_df[target_variable]

    # iterations = 1500
    # alpha = 0.01
    # beta = np.zeros((len(selected_features) + 1, 1))

    # # Normalize features
    # normalized_features, means, stds = normalize_z(features.to_numpy())
    # X = prepare_feature(normalized_features)
    # y = target.to_numpy()

    # # Gradient descent
    # beta, J_storage = gradient_descent_linreg(X, y, beta, alpha, iterations)

    #Running the model


    beta_n = len(selected_features) + 1

    # get features and targets from data frame
    df_feature, df_target = get_features_targets(merged_df, selected_features , target_variable)

    # split the data into training and test data sets
    df_feature_train, df_feature_test, df_target_train, df_target_test = split_data(df_feature, df_target, 100, 0.3)
    df_target_train = pd.DataFrame(df_target_train)


    # normalize the feature using z normalization
    array_feature_train_z, means, stds = normalize_z(df_feature_train.to_numpy())

    X: np.ndarray = prepare_feature(array_feature_train_z)
    target: np.ndarray = df_target_train.to_numpy()

    iterations: int = 1500
    alpha: float = 0.01
    beta: np.ndarray = np.zeros((beta_n,1))

    print(X.shape)
    print(target.shape)
    print(beta.shape)

    # call the gradient_descent function
    beta, J_storage = gradient_descent_linreg(X, target, beta, alpha, iterations)

    # call the predict method to get the predicted values
    pred: np.ndarray = predict_linreg(df_feature_test.to_numpy(), beta, means, stds)

    # Metrics
    n_features = df_feature_train.shape[1]
    metrics = evaluate_model_metrics(df_target_test.to_numpy(), predict_linreg(df_feature_test.to_numpy(), beta, means, stds), n_features)

    # Equation
    equation = "y = "
    equation += " + ".join([f"{beta[i][0]:.4f} * {selected_features[i]}" for i in range(len(selected_features))])
    equation += f" + {beta[-1][0]:.4f}"  # Add the intercept

    # Plot cost graph
    cost_graph = base64.b64encode(plot_cost_graph(J_storage)).decode("utf-8")

    return metrics, equation, cost_graph

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