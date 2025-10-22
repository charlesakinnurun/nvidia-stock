# %% [markdown]
# Import the neccessary libraries

# %%
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import warnings

# %%
warnings.filterwarnings("ignore") # To supress warnings

# %%
# Set the global plotting style for better aesthetics
plt.style.use("fivethirtyeight")

# %% [markdown]
# Data Loading and Acquitsion

# %%
ticker = "NVDA"

# %%
def load_data(ticker="NVDA",start_date="2023-01-01",end_date=pd.to_datetime("today").strftime('%Y-%m-%d')):
    print(f"Loading data for {ticker}......")
    # Download historical stock data from the Yahoo Finance
    df = yf.download(ticker,start=start_date,end=end_date)
    # Target Variable: The prce we want to predict (Next Day's Close Price)
    # We shift the "Close" column up by -1 day (the future day)
    df["Target"] = df["Close"].shift(-1)
    # Feature Enginneering: Lagged price Yesterday's Close price is a key predictor for today's price
    df["Lag_1"] = df["Close"].shift(-1)
    # Feature Engineering: Daily Volatility (High-Low)
    df["Volatility"] = df["High"] - df["Low"]
    # Drop rows with NaN nvalues created by shift/lag (the first and last rows)
    df.dropna(inplace=True)

    return df

# %%
# Load the NVIDIA data
df = load_data()

# %%
df.reset_index(inplace=True)

# %%
# Check if data was successfully loaded and is not empty
if df.empty:
    print(f"No data retrived for {ticker}. Exiting...")
    exit()

# %%
df

# %% [markdown]
# Data Preprocessing

# %%
# Check for missing values
df_missing = df.isnull().sum()
print("Missing Values")
print(df_missing)

# %%
# Check for dupliated rows
df_duplicated = df.duplicated().sum()
print("Duplicated Rows")
print(df_duplicated)

# %% [markdown]
# Feature Engineering

# %%
# Define the features (X) and target (y)
# Features used for prediction: Today's Close, Yesterday's Close and Today's Volatility
features = ["Close","Lag_1","Volatility"]
X = df[features].values
y = df["Target"].values

# %% [markdown]
# Data Splitting

# %%
# Split the data into training (80%) and testing (20%) sets
# shuffle=False is cruical for time-series data to maintain temporal order
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)

# %% [markdown]
# Data Scaling

# %%
# Stanadardize features by removing the mean and scaling to unit variance
# This is crucial for distance-based algorithms like SVR and Linear Models with regularization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Use the fitted scaler from the training set

# %% [markdown]
# Model Definition and Hyperparameter Tuning

# %%
# Dictionary to store the best model and its name
best_model = {
    "score":-np.inf,
    "name":None,
    "model":None
}

results = {}

# %%
# Linear Regresson (No tuning needed)
print("Training Linear Regression......")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled,y_train)
lr_pred = lr_model.predict(X_test_scaled)
results["Linear Regression"] = {
    "model":lr_model,
    "y_pred":lr_pred
}

# %%
# Ridge Regression (Tuning Alpha)
print("Tuning Ridge Regression.......")
# Define the hyperparameter grid for Ridge (alpha contols regularization strength)
ridge_params = {
    "alpha":np.logspace(-4,2,100)
} # 100 values between 0.0001 and 100
ridge_search = RandomizedSearchCV(Ridge(random_state=42),ridge_params,n_iter=20,cv=5,scoring="neg_mean_squared_error",random_state=42,n_jobs=-1)
ridge_search.fit(X_train_scaled,y_train)
ridge_model = ridge_search.best_estimator_
ridge_pred = ridge_model.predict(X_test_scaled)
results["Ridge Regression"] = {
    "model":ridge_model,
    "y_pred":ridge_pred
}

# %%
# Support Vector Regression (SVR) (Tuning C and Gamma)
print("Tuning Support Vector Regression (SVR)........")
# Define the hyperparameter grid for SVR (C for penalty, gamma for kernel influence)
svr_params = {
    "C":[0.1,1,10,100],
    "gamma":["scale","auto",0.01,0.1,1],
    "kernel":["rbf"] # Radial Basis Function (RBF) is best for non-linear data
}
svr_search = RandomizedSearchCV(SVR(),svr_params,n_iter=10,cv=6,scoring="neg_mean_squared_error",random_state=42,n_jobs=-1)
svr_search.fit(X_train_scaled,y_train)
svr_model = svr_search.best_estimator_
svr_pred = svr_model.predict(X_test_scaled)
results["Support Vector Regression"] = {
    "model":svr_model,
    "y_pred":svr_pred
}

# %%
# Random Forest Regressor (RFR) (Tuning n_estimators and max_depth)
print("Tuning Random Forest Regressor.......")
#  Define the hyperparameter grid for Random Forest o(n_estimator= number of tree,max_depth=depth of the trees)
rfr_params = {
    "n_estimators":[50,100,200],
    "max_depth":[5,10,15,None],
    "min_samples_split":[2,5]
}
rfr_search = RandomizedSearchCV(RandomForestRegressor(random_state=42),rfr_params,n_iter=10,cv=3,scoring="neg_mean_squared_error",random_state=42,n_jobs=1)
rfr_search.fit(X_train_scaled,y_train)
rfr_model = rfr_search.best_estimator_
rfr_pred = rfr_model.predict(X_test_scaled)
results["Random Forest Regressor"] = {
    "model":rfr_model,
    "y_pred":rfr_pred
}

# %% [markdown]
# Model Comparison and Evaluation

# %%
comparison_df = pd.DataFrame(columns=["Model","R2 Score","RMSE","Tuning Params"])


# Evaluate and compare all models
for name,data in results.items():
    model = data["model"]
    y_pred = data["y_pred"]

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2 = r2_score(y_test,y_pred)

    # Check if the model has the best params attribute from RandomizedSearchCV
    #tuning_params = "N/A"
    #if name in ["Ridge Regression","Support Vector Regression","Random Forest Regressor"]:
        #tuning_params = eval(f"{name.split(" ")[0].lower()}_search").best_params_

    # Store  results
    #comparison_df.loc[len(comparison_df)] = [name,r2,rmse,tuning_params]


    # Track the best model based on R2 Score
    if r2 > best_model["score"]:
        best_model["score"] = r2
        best_model["name"] = name
        best_model["model"] = model
        best_model["scaler"] = scaler # Store the fitted scaler
        best_model["features"] = features # Store the features names


print("----- Model Comparsion Reults (R2 Score & RMSE)-----")
print(comparison_df.sort_values(by="R2 Score",ascending=False))
print(f"Best Performing Model: {best_model["name"]} (R2 Score: {best_model["score"]:.4f})")

# %% [markdown]
# Visualization After Training (Actual vs Predicted)

# %%
plt.Figure(figsize=(14,6))
plt.title("NVIDIA Stock Price Prediction Comparison")
plt.plot(df.index[-len(y_test):],y_test,label="Actual Price (Test Data)",color="blue",linewidth=3) # Actual Prices
colors = ["red","green","orange","purple"]

# Plot the predictions for all models
for i,name in enumerate(results.keys()):
    plt.plot(df.index[-len(y_test):],results[name]["y_pred"],label=f"{name} Prediction",alpha=0.7,color=colors[i])

plt.xlabel("Date")
plt.ylabel("Next Day's Closing Price (USD)")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()

# %% [markdown]
# New Prediction Input Function

# %%
def predict_new_price(current_close,yesterday_close,volatility,model_info):
    """
    Predicts the next day's closing price using the best model.
    :param current_close: Today's (last available) closing price.
    :param yesterday_close: The closing price from the day before that.
    :param volatility: Today's High - Low price difference.
    :param model_info: Dictionary containing the best model and scaler.
    """
    # Get the model and scaler from the results
    model = model_info["model"]
    scaler = model_info["scaler"]
    model_name = model_info["name"]

    # Create the new input data array (must match the feature order)
    new_data = np.array([[current_close,yesterday_close,volatility]])

    # Scale the new input data using the fitted training scaler
    new_data_scaled = scaler.transform(new_data)

    # Make the prediction
    predicted_price = model.predict(new_data_scaled)[0]

    print("----- New Price Prediction -----")
    print(f"Using Model: {model_name}")
    print(f"Input Features (Close,Lag_1,Volatility): {new_data[0]}")
    print(f"Predicted Next Day's Close Price: ${predicted_price:.2f}")


# Example Usage of the New prediction function (using the last point from the dataframe)
last_day_features = df.iloc[-1][features].values
# We use the features from the last recorded day to predict the price for the next unrecorded day
predict_new_price(
    current_close=last_day_features[0],
    yesterday_close=last_day_features[1],
    volatility=last_day_features[2],
    model_info=best_model
)


