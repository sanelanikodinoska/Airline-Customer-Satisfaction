# imports
import os
import mlflow
import argparse

import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# define functions
def main(args):
    # enable auto logging
    mlflow.sklearn.autolog()

    # setup parameters
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "random_state": args.random_state,
    }

    # read in data
    df = pd.read_csv(args.data)

    # process data
    X_train, X_test, y_train, y_test = process_data(df, args.random_state)

    # train model
    model = train_model(params, X_train, X_test, y_train, y_test)
    # Output the model and test data
    mlflow.sklearn.save_model(model, args.model_output)
    X_test.to_csv(Path(args.test_data) / "X_test.csv", index=False)
    y_test.to_csv(Path(args.test_data) / "y_test.csv", index=False)


def process_data(df, random_state):
    # split dataframe into X and y
    X = df.drop(["satisfaction"], axis=1)
    y = df["satisfaction"]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )

    # return split data
    return X_train, X_test, y_train, y_test


def train_model(params, X_train, X_test, y_train, y_test):
    # train model
    model = RandomForestClassifier(**params)
    model = model.fit(X_train, y_train)

    # return model
    return model


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
 
    parser.add_argument("--data", type=str)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type = int, default=1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--test_data", type=str, help="Path of output model")

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
