import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pred_file',type=str, help='enter path to the saved prediction file')
args = parser.parse_args()

pred_file = pickle.load(open(args.pred_file, "rb"))

print(f"Metrics for : {args.pred_file}")
change_points = np.array(pickle.load(open("change_points.pkl", "rb")))

for horizon in [3, 6, 12]:

    print(f"Horizon: {horizon}")
    start = 34272 - 6850 - (12 - horizon)
    end = 34272 - (12 - horizon)
    realy = pred_file[f"real{horizon}"]
    preds = pred_file[f"pred{horizon}"]

    mae = []
    mape = []
    mse = []

    for i in range(207):
        change_point = change_points[i]
        change_point = np.array(change_point)
        change_point = change_point[((change_point > start) & (change_point < end))] - start

        y = preds[:, i][change_point]
        y_hat = realy[:, i][change_point]
        y = y[y_hat > 0]
        y_hat = y_hat[y_hat > 0]
        mae.append(np.mean(np.abs(y - y_hat)))
        mape.append(np.mean(np.abs(y - y_hat) / y))
        mse.append(np.mean((y - y_hat) ** 2))

    mae = np.mean(mae)
    print(f"MAE: {mae:.2f}")

    mse = np.mean(mse)
    #print(f"MSE: {mse:.2f}")

    rmse = np.sqrt(mse)
    print(f"RMSE: {rmse:.2f}")

    mape = np.mean(mape) * 100
    print(f"MAPE: {mape:.2f}%")
    