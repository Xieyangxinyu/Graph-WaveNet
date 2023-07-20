import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pred_file',type=str, help='enter path to the saved prediction file')
args = parser.parse_args()


pred_file = pickle.load(open(args.pred_file, "rb"))

def bMAE(mae, realy):
    mae_bin = []
    for interval in [0, 10, 20, 30, 40, 50, 60]:
        mae_bin.append(np.mean(mae[((realy > interval) & (realy <= interval + 10))]))
    #print(mae_bin)
    return np.mean(mae_bin)

def bRMSE(mae, realy):
    mae_bin = []
    for interval in [0, 10, 20, 30, 40, 50, 60]:
        mae_bin.append(np.sqrt(np.mean(mae[((realy > interval) & (realy <= interval + 10))])))
    #print(mae_bin)
    return np.mean(mae_bin)

print(f"Metrics for : {args.pred_file}")

for horizon in [3, 6, 12]:
    print(f"Horizon: {horizon}")
    realy = pred_file[f"real{horizon}"]
    preds = pred_file[f"pred{horizon}"]

    preds = preds[realy > 0]
    realy = realy[realy > 0]

    mae = np.abs(realy - preds)
    for percentile in [.90, .95, .99]:
        print(f"AE VaR {percentile}%: {np.quantile(mae, percentile):.2f}")

    bmae = bMAE(mae, realy)
    mape = mae / realy * 100
    mae = np.mean(mae)
    print(f"MAE: {mae:.2f}")

    for percentile in [.9, .95, .99]:
        print(f"APE VaR {percentile}%: {np.quantile(mape, percentile):.2f}%")
    mape = np.mean(mape)
    print(f"MAPE: {mape:.2f}%")


    mse = (realy - preds) ** 2
    for percentile in [.9, .95, .99]:
        print(f"SE VaR {percentile}%: {np.quantile(mse, percentile):.2f}")
    
    brmse = bRMSE(mse, realy)
    mse = np.mean(mse)
    print(f"MSE: {mse:.2f}")
    print(f"bRMSE: {brmse:.2f}")
    print(f"RMSE: {np.sqrt(mse):.2f}")

    print(f"bMAE: {bmae:.2f}")



