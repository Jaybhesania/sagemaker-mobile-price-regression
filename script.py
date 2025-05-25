
import argparse
import os
import numpy as np
import pandas as pd
import json
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import joblib
import pathlib
import boto3
from io import StringIO

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ == '__main__':
    print("[INFO] Parsing arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--train_file', type=str, default='train-V1.csv')
    parser.add_argument('--test_file', type=str, default='test-V1.csv')
    args, _ = parser.parse_known_args()

    print("[INFO] Environment and arguments:")
    print("  Model directory:", args.model_dir)
    print("  Train directory:", args.train)
    print("  Test directory:", args.test)
    print("  Train file:", args.train_file)
    print("  Test file:", args.test_file)

    # Print directory contents to debug file mounting
    print("\n[DEBUG] Contents of train directory:")
    print(os.listdir(args.train))
    print("\n[DEBUG] Contents of test directory:")
    print(os.listdir(args.test))

    try:
        train_path = os.path.join(args.train, args.train_file)
        test_path = os.path.join(args.test, args.test_file)
        print("\n[INFO] Reading training data from:", train_path)
        print("[INFO] Reading test data from:", test_path)

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

    except Exception as e:
        print("[ERROR] Could not read training or test file.")
        print("Exception message:", str(e))
        raise


    features = list(train_df.columns)
    label = features.pop(-1)
    
    print("Building training and testing datasets")
    print()
    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[label]
    y_test = test_df[label]

    print('Column order: ')
    print(features)
    print()
    
    print("Label column is: ",label)
    print()
    
    print("Data Shape: ")
    print()
    print("---- SHAPE OF TRAINING DATA (80%) ----")
    print(X_train.shape)
    print(y_train.shape)
    print()
    print("---- SHAPE OF TESTING DATA (20%) ----")
    print(X_test.shape)
    print(y_test.shape)
    print()

    print("\n[INFO] Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, verbose=3, n_jobs=None)
    model.fit(X_train, y_train)

    print("\n[INFO] Saving model...")
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print("Model saved to:", model_path)

    y_pred_test = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    report = classification_report(y_test, y_pred_test)

    print("\n[INFO] Model evaluation on test set:")
    print("Accuracy:", acc)
    print("Classification Report:\n", report)
