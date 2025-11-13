# Import necessary libraries
import warnings  # To control warning messages
import argparse  # To parse command-line arguments
import logging   # To log information
import pandas as pd  # For data manipulation (DataFrames)
import numpy as np   # For numerical operations
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # For evaluating the model
from sklearn.model_selection import train_test_split  # To split data into train and test sets
from sklearn.linear_model import ElasticNet  # The machine learning model we will use

# Configure basic logging to show warnings
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# --- Argument Parsing ---
# Set up a way to receive arguments from the command line when running the script
parser = argparse.ArgumentParser()
# Define an argument named '--alpha', which is a float, optional, and defaults to 0.5
parser.add_argument("--alpha", type=float, required=False, default=0.5)
# Define an argument named '--l1_ratio', which is a float, optional, and defaults to 0.5
parser.add_argument("--l1_ratio", type=float, required=False, default=0.5)
# Parse the arguments provided at the command line
args = parser.parse_args()

# --- Evaluation Function ---
# Define a function to calculate evaluation metrics
def eval_metrics(actual, pred):
    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(actual, pred)
    # Calculate R-squared (R2) score
    r2 = r2_score(actual, pred)
    # Return the three calculated metrics
    return rmse, mae, r2


# --- Main Execution Block ---
# Check if the script is being run directly (not imported as a module)
if __name__ == "__main__":
    # Suppress (ignore) any warning messages
    warnings.filterwarnings("ignore")
    # Set a random seed for NumPy to ensure results are reproducible
    np.random.seed(40)

    # --- Data Loading ---
    # Read the wine quality data from a local CSV file into a pandas DataFrame
    data = pd.read_csv("red-wine-quality.csv")
    
    # Note: This line saves the data back out to a different location.
    # This might be for traceability or to stage data for another process.
    data.to_csv("data/red-wine-quality.csv", index=False)

    # --- Data Preprocessing ---
    # Split the DataFrame 'data' into two sets: 'train' and 'test'
    # By default, train_test_split uses a 75% (train) / 25% (test) split
    train, test = train_test_split(data)

    # Separate features (X) from the target variable (y) for the training set
    # 'train_x' gets all columns *except* "quality"
    train_x = train.drop(["quality"], axis=1)
    # 'train_y' gets *only* the "quality" column
    train_y = train[["quality"]]

    # Separate features (X) from the target variable (y) for the test set
    # 'test_x' gets all columns *except* "quality"
    test_x = test.drop(["quality"], axis=1)
    # 'test_y' gets *only* the "quality" column
    test_y = test[["quality"]]

    # --- Model Training ---
    # Get the hyperparameter values from the parsed command-line arguments
    alpha = args.alpha
    l1_ratio = args.l1_ratio

    # Initialize the ElasticNet regression model
    # 'alpha' is the regularization strength
    # 'l1_ratio' balances L1 (Lasso) and L2 (Ridge) regularization
    # 'random_state' ensures reproducibility of the model's internal operations
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    
    # Train (fit) the model on the training features (train_x) and training target (train_y)
    lr.fit(train_x, train_y)

    # --- Model Prediction & Evaluation ---
    # Use the trained model ('lr') to make predictions on the *test* features (test_x)
    predicted_qualities = lr.predict(test_x)

    # Calculate the evaluation metrics by comparing the model's predictions to the actual test values
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # --- Print Results ---
    # Print the model's hyperparameters and its performance metrics
    print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
