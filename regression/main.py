import matplotlib.pyplot as plt
import numpy as np
from Regression import Regression
from common import CSV_PATH
import pandas as pd
import ast
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEGREE = 3


def plot(points, target):
    model = Regression(points, DEGREE)
    y_predict = model.predict(target[0])

    fig, ax = plt.subplots()
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    ax.scatter(x_vals, y_vals, color='black', label='Original points')

    x_curve = np.linspace(min(x_vals) - 1, max(x_vals) + 2, 300)
    y_curve = model.predict(x_curve)
    ax.plot(x_curve, y_curve, color='blue', label='Cubic regression')

    ax.scatter([target[0]], [y_predict], color='red',
               label=f'Predicted point ({target[0]:.1f}, {y_predict:.1f})')

    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Cubic Regression')

    plt.show()

def runRegression(csv_path):
    df = pd.read_csv(csv_path)
    y_predicts = []

    for index, row in df.iterrows():
        point_strings = row.iloc[:-1]
        target_string = row.iloc[-1]

        points = [ast.literal_eval(p) for p in point_strings]
        target = ast.literal_eval(target_string)

        model = Regression(points, DEGREE)
        y_pred = model.predict(target[0])
        # plot(points, target)
        y_predicts.append(y_pred)

    targets = [ast.literal_eval(t)[1]
            for t in df.iloc[:, -1]]
    y_true = np.array(targets)
    y_pred = np.array(y_predicts)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

if __name__ == "__main__":
    runRegression(CSV_PATH)