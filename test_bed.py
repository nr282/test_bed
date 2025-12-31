"""
Test bed to examine if:
(1) Linear Regression is outperformed by Probabilistic Programming.

On the basis of this experiment, which we can disclose to the public,
I can demonstrate that our technology will do better in the Natural
Gas Estimation that uses linear regression by a certain percentage.

This is a publicly available experiment demonstrating Spectral Technologies'
sophistication.

"""

import calendar
import pandas as pd
import numpy as np
import numpy as np
from numpy.ma.core import indices
from sklearn.model_selection import train_test_split
import pymc as pm
import datetime

def create_test_bed(start_date="2020-01-01", end_date="2024-12-31"):

    dates = pd.date_range(start_date, end_date)
    n = len(dates)
    df = pd.DataFrame(index=dates)
    alpha = np.random.normal(loc=0, scale=10)
    x = np.random.normal(loc=0, scale=10, size=(n, 1))
    y = alpha * x
    df["y"] = y
    df["x"] = x
    df["Date"] = df.index
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df["Day"] = df.index.day
    y = df.groupby(["Year", "Month"])["y"].sum().reset_index()
    df.drop(columns=["y"], inplace=True)

    return df, y

def split_data(df, y):

    year = 2024
    y_train = y[y["Year"] < year]
    y_test = y[y["Year"] >= year]
    x_train = df[df["Year"] < year]
    x_test = df[df["Year"] >= year]
    x = df
    return x_train, x_test, y_train, y_test, x

def linear_regression_train(x_train, y_train):
    from sklearn.linear_model import LinearRegression
    x_train = x_train.groupby(["Year", "Month"])["x"].sum().reset_index()
    comparison = x_train.merge(y_train, on=["Year", "Month"])


    model = LinearRegression(fit_intercept=False).fit(comparison["x"].values[:, np.newaxis],
                                                        comparison["y"].values[:, np.newaxis])

    return model


def linear_regression_predict(model, x_test, y_test):

    x_test = x_test.groupby(["Year", "Month"])["x"].sum().reset_index()
    comparison = x_test.merge(y_test, on=["Year", "Month"])
    comparison["y_predict"] = model.predict(comparison["x"].values[:, np.newaxis])
    return comparison


def get_days_in_month(year, month):
    number_of_days_in_month = calendar.monthrange(year, month)[1]
    dates = set()
    for day in range(1, number_of_days_in_month + 1):
        dates.add(datetime.datetime(year, month, day))
    return dates


def prob_prog_train(x, y_train, start_date, end_date):

    dates = pd.date_range(start_date, end_date)
    coords = {
        "dates": list(dates),
    }

    date_to_index = dict()
    index_to_date = dict()
    for i, date in enumerate(dates):
        date_to_index[date] = i
        index_to_date[i] = date

    with pm.Model(coords=coords) as model:

        alpha = pm.Normal("alpha",
                            mu=0,
                            sigma=20)

        y = pm.Normal("y",
                    mu=alpha * x["x"],
                    sigma=5,
                    dims="dates")

        for r_i, row in y_train.iterrows():
            value = row["y"]
            year = int(row["Year"])
            month = int(row["Month"])

            dates_in_month = get_days_in_month(year, month)
            indicies = []
            for date in dates_in_month:
                index = list(dates).index(date)
                indicies.append(index)

            pm.Normal(f"month_{year}_{month}",
                      mu=sum([y[index] for index in indicies]),
                      sigma=10,
                      observed=value)

        pm.sample(draws=10, tune=10, cores=4)

if __name__ == "__main__":
    start_date = "2022-01-01"
    end_date = "2024-12-31"
    df, y = create_test_bed(start_date, end_date)
    x_train, x_test, y_train, y_test, x = split_data(df, y)
    model = linear_regression_train(x_train, y_train)
    y_predict = linear_regression_predict(model, x_test, y_test)
    prob_prog_train(x, y_train, start_date, end_date)
    #Extract y_predict.
















