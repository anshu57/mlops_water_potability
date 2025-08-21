
import pandas as pd
import numpy as np
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
import dagshub

# Initialize Dags hub and set up mlflow experiment tracking
dagshub.init(repo_owner='anshu57', repo_name='mlops_water_potability', mlflow=True)
mlflow.set_experiment('Experiment_1')
mlflow.set_tracking_uri("https://dagshub.com/anshu57/mlops_water_potability.mlflow")

data = pd.read_csv("/Users/anshugangwar/Desktop/Anshu/mlops_water_potability/data/raw/water_potability.csv")

# split data into training and test set
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)


# Define a function to fill misiing values in the data set with the median value of each column
def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median() 
            df.fillna({column: median_value}, inplace=True)
    return df




# Fill missing values in both the training and test datasets using the median
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)


# Import RandomForestClassifier and pickle for model saving
from sklearn.ensemble import RandomForestClassifier
import pickle



# Separate features (X) and target (y) for training
X_train = train_processed_data.drop(columns=["Potability"], axis=1)  # Features
y_train = train_processed_data["Potability"]  # Target variable


n_estimators = 100  # Number of trees in the Random Forest

# Initialize and train the Random Forest model
clf = RandomForestClassifier(n_estimators=n_estimators)
clf.fit(X_train, y_train)

# Prepare test data for prediction (features and target)
X_test = test_processed_data.iloc[:, 0:-1].values  # Features for testing
y_test = test_processed_data.iloc[:, -1].values  # Target variable for testing
    
# Predict the target for the test data
y_pred = clf.predict(X_test)

# Calculate performance metrics
acc = accuracy_score(y_test, y_pred)  # Accuracy
precision = precision_score(y_test, y_pred)  # Precision
recall = recall_score(y_test, y_pred)  # Recall
f1 = f1_score(y_test, y_pred)  # F1-score

with mlflow.start_run():

    # Save the trained model to a file using pickle
    # pickle.dump(clf, open("model.pkl", "wb"))





    # Load the saved model for prediction
    # model = pickle.load(open('model.pkl', "rb"))



    # Log metrics to MLflow for tracking
    mlflow.log_metric("acc", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1-score", f1)

    # Log the number of estimators used as a parameter
    mlflow.log_param("n_estimators", n_estimators)

    # Generate a confusion matrix to visualize model performance
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True)  # Visualize confusion matrix
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Save the confusion matrix plot as a PNG file
    plt.savefig("confusion_matrix.png")

    # Log the confusion matrix image to MLflow
    mlflow.log_artifact("confusion_matrix.png")

    # Infer the model signature
    signature = infer_signature(X_train, clf.predict(X_train))

    # log model
    mlflow.sklearn.log_model(clf,artifact_path="model", signature=signature)

    mlflow.log_artifact(__file__)

    # Set tags in MLflow to store additional metadata
    mlflow.set_tag("author", "datathinkers")
    mlflow.set_tag("model", "GB")
    # Print out the performance metrics for reference
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
