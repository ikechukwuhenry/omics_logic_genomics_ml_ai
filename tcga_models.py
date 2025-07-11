import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def preprocess_data(df):
    """
    Preprocess the TCGA data.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing TCGA data.
    
    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    # Handle missing values
    df = df.dropna()
    
    # Convert categorical variables to numerical
    df = pd.get_dummies(df, drop_first=True)
    
    return df

def load_data(file_path):
    """
    Load TCGA data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file containing TCGA data.
    
    Returns:
    pd.DataFrame: DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)

def train_model(X_train, y_train, model):
    """
    Train a logistic regression model.
    
    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training labels.
    
    Returns:
    LogisticRegression: Trained logistic regression model.
    """
    # model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    
    Parameters:
    model (LogisticRegression): Trained logistic regression model.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test labels.
    
    Returns:
    dict: Evaluation metrics including accuracy, classification report, and confusion matrix.
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }   

def main(file_path):
    """
    Main function to load data, preprocess, train, and evaluate the model.
    
    Parameters:
    file_path (str): Path to the CSV file containing TCGA data.
    """
    # Load data
    df = load_data(file_path)
    
    # Preprocess data
    # df = preprocess_data(df)
    
    # Split data into features and labels
    X = df.drop('label', axis=1)  # Assuming 'label' is the target variable
    y = df['label']
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model
    logistic_model = LogisticRegression(max_iter=1000)

    # Create and train SVM model
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)

    model = train_model(X_train, y_train, logistic_model)
    
    # Evaluate model
    evaluation_results = evaluate_model(model, X_test, y_test)
    
    print("Model Evaluation Results:")
    print(f"Accuracy: {evaluation_results['accuracy']}")
    print("Classification Report:")
    print(evaluation_results['classification_report'])
    print("Confusion Matrix:")
    print(evaluation_results['confusion_matrix'])

if __name__ == "__main__":
    # Example usage
    import os
    tcga_data_dir = os.getcwd() # Replace with your actual directory path
    file_path = 'transformed_tcga.csv'  # Replace with your actual file path
    file_path = os.path.join(tcga_data_dir, file_path)
    main(file_path)