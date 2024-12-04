import pandas as pd

######Import data#########
###Importing datasets

###Independent Variables
#Outbound Departures
'''
Independent Variable: Total outbound departures of Singapore residents.
Column Title: Total (Rename to TotalOutbound)
Description: This dataset tracks how many residents are traveling out of Singapore each month.
Potential Impact: Could indicate reduced local food service usage due to residents traveling abroad.
Source:https://tablebuilder.singstat.gov.sg/table/TS/M650661
data.gov dataset ID: d_a9109fbbdcc9e874fcce773060183cff
'''
df1 = pd.read_csv("datasets\OutboundDeparturesOfSingaporeResidentsByModeOfTransportMonthly.csv")

#Inbound Tourism Markets
'''
Independent Variable: Total international visitor arrivals.
Column Title: Total International Visitor Arrivals By Inbound Tourism Markets (Rename to TotalInbound)
Description: Tracks monthly international visitor arrivals to Singapore.
Potential Impact: Reflects the contribution of tourism to local food and beverage spending.
Source: https://tablebuilder.singstat.gov.sg/table/TS/M550001
data.gov dataset ID: d_7e7b2ee60c6ffc962f80fef129cf306e
'''
df2 = pd.read_csv("datasets\InternationalVisitorArrivalsByInboundTourismMarketsMonthly.csv")


#Event Index
'''
Independent Variable: Event Index
Column Title: EventIndex
Description: Self generated file to capture the number of  public and school holidays at each month on a scale of 1 to 5
Index Assignment:
1: Very few holidays (0–2 holiday days).
2: Few holidays (3–4 holiday days).
3: Moderate holidays (5–7 holiday days).
4: High holidays (8–10 holiday days).
5: Very high holidays (11+ holiday days).
Potential Impact: Could impact tourism in and out of Singapore
Source: Self Made based on https://www.moe.gov.sg/news/press-releases/20230807-school-terms-and-holidays-for-2024
'''
df3 = pd.read_csv("datasets\event_index_data.csv")

#Consumer Price Index
'''
Independent Variable: Inflation rate (calculated from the CPI).
Column Title: All Items (Rename to CPIndex)
Description: Will be used to calculate inflation rate.
Potential Impact: High inflation could reduce consumer spending in the food and beverage sector.
Source: https://tablebuilder.singstat.gov.sg/table/TS/M212881
data.gov dataset ID: d_de7e93a1d0e22c790516a632747bf7f0
'''
df4 = pd.read_csv("datasets\ConsumerPriceIndexCPI2019AsBaseYearMonthly.csv")


#Tourist Expenditure
'''
Independent Variable: Total tourist expenditure.
Column Title: Various categories (Combine and rename columns to TotalExp)
Potential Impact: Reflects the economic contribution of tourism to food and beverage services.
Source: https://www.singstat.gov.sg/find-data/search-by-theme/industry/tourism/latest-data
'''
df5 = pd.read_csv("datasets\TourismReceipts.csv")

#Food imports
'''
Independent Variable: Value of food imports.
Column Titles: Various categories (Aggregate as TotalFoodImports)
Description: Tracks Singapore’s reliance on imported food.
Potential Impact: A heavy reliance on imports could influence food prices and the FBSIndex, especially during global supply disruptions.
Source: https://tablebuilder.singstat.gov.sg/table/TS/M451021
data.gov dataset ID: d_b89e35ce38cb93a17f5c016e71f50690
'''
df6 = pd.read_csv("datasets\MerchandiseImportsByCommodityDivisionMonthly.csv")

###Dependent Variables
#Food Beverage Services Index
'''
Dependent Variable: Food and Beverage Services Index.
Column Title: Total (Rename to FBSIndex)
Description: Measures the monthly performance of Singapore’s food and beverage services sector.
Potential Impact: Serves as the primary measure of how variables like tourism, imports, and inflation impact food service performance.
Source: https://tablebuilder.singstat.gov.sg/table/TS/M601661
data.gov dataset ID: d_d7933d23e5fab92a086585cfb9224ba2
'''
df8 = pd.read_csv("datasets\FoodBeverageServicesIndex2017100AtCurrentPricesMonthly.csv")

#Food prices
'''
Dependent Variable: Food Prices
Column Title: Economical Rice (1 Meat & 2 Vegetables) (Per Plate) - Rename to MealPrice
"Premium Thai Rice (Per 5 Kilogram)" - Rename to "RicePrice"
Description: Shows the prices of commodities such as the food staple Rice, and an average meal
Potential Impact: Shows the direct costs of affording food
Source: https://tablebuilder.singstat.gov.sg/table/TS/M212891
data.gov dataset ID: d_d2467766bca7c1ed64ecd8fe07029df3
'''
df9 = pd.read_csv("datasets\AverageRetailPricesOfSelectedConsumerItemsMonthly.csv")


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

# Event Index Data
event_column_map = {"EventIndex": "EventIndex"}
df3 = preprocess_dataframe(df3, event_column_map, n_rows)
#display(df3)

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

# Food Prices
prices_column_map = {"DataSeries": "DataSeries",
                     "Premium Thai Rice (Per 5 Kilogram)": "RicePrice",
                     "Economical Rice (1 Meat & 2 Vegetables) (Per Plate)": "MealPrice"}
df9 = preprocess_dataframe(df9, prices_column_map, n_rows)
#display(df9)

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
merged_df = pd.concat([df1, df2, df3, df_4_i, df_5_q, df_6_t, df8, df9], axis=1)
# Remove duplicate columns
merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
# Display the final merged DataFrame
#display(merged_df)

#prepare features