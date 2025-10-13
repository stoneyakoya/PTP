import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score


# --- 1. フィッティング関数の定義（Zecha et al. Nat Commun 13,165 (2022) Eqn 10,11） ---
def label_loss(t, K, A, C):
    # L(t) = A * exp(-K t) + C
    return A * np.exp(-K * t) + C


def label_incorp(t, K, A, C):
    # I(t) = A * (1 - exp(-K t)) + C
    return A * (1 - np.exp(-K * t)) + C


# --- 1. テストセット読み込み ---
test_df = pd.read_csv("data/results/2025_0429_210347/test_results.csv")
print(test_df.head())
print(test_df.columns)


# --- 曲線フィット関数 ---
def fit_curve(t, y, mode="loss"):
    # クリーニング: 0〜1にclip
    y = np.clip(y, 0, 1)

    # 初期値設定
    if mode == "loss":
        p0 = [0.05, y[0] - y[-1], y[-1]]
    elif mode == "incorp":
        p0 = [0.05, y[-1] - y[0], y[0]]
    else:
        raise ValueError("mode must be 'loss' or 'incorp'")

    # 重み設定（早期タイムポイント重視）
    weights = np.exp(-t / t.max())

    # フィット
    if mode == "loss":
        popt, _ = curve_fit(
            label_loss, t, y, p0=p0, sigma=weights, absolute_sigma=True, maxfev=10000
        )
    else:
        popt, _ = curve_fit(
            label_incorp, t, y, p0=p0, sigma=weights, absolute_sigma=True, maxfev=10000
        )

    return popt


# --- メイン処理 ---
results = []

for peptidoform, grp in test_df.groupby("Peptide_ID"):
    t = grp["Timesteps"].values
    loss = grp["Predictions_Loss"].values
    inc = grp["Predictions_Incorporation"].values
    true_loss = grp["True_Values_Loss"].values
    true_incorp = grp["True_Values_Incorporation"].values
    loss_K = grp["loss_K"].values[0]
    incorp_K = grp["incorp_K"].values[0]
    preds_loss_K = np.exp(grp["preds_loss_K"].values[0])
    preds_incorp_K = np.exp(grp["preds_incorp_K"].values[0])
    # print(t)
    # print("prediction_loss :", loss)
    # print("prediction_incorporation :", inc)
    # print("true_loss :", true_loss)
    # print("true_incorp :", true_incorp)
    # print(loss_K)
    # print(incorp_K)
    results.append(
        {
            "Peptidoform": peptidoform,
            "predict_loss_K": preds_loss_K,
            "predict_incorp_K": preds_incorp_K,
            "loss_K": loss_K,
            "incorporation_K": incorp_K + 1e-8,
        }
    )


# --- 2. DataFrame化 ---
results_df = pd.DataFrame(results)
results_df.to_csv("data/results/K/test_results_K.csv", index=False)

# --- 3. 精度計算関数 ---
p_loss, _ = spearmanr(results_df["loss_K"], results_df["predict_loss_K"])
p_incorp, _ = spearmanr(results_df["incorporation_K"], results_df["predict_incorp_K"])


def evaluate_K(true, pred, name):
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    print(f"--- {name} ---")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²  : {r2:.6f}")


# --- 4. 評価 ---
evaluate_K(results_df["loss_K"], results_df["predict_loss_K"], "Loss K Prediction")
print(f"Spearman Correlation: {p_loss:.6f}")
evaluate_K(
    results_df["incorporation_K"],
    results_df["predict_incorp_K"],
    "Incorporation K Prediction",
)
print(f"Spearman Correlation: {p_incorp:.6f}")


import matplotlib.pyplot as plt

# 1. Loss_Kの散布図
plt.figure(figsize=(6, 6))
plt.scatter(results_df["loss_K"], results_df["predict_loss_K"], alpha=0.5)
plt.plot([0, 0.5], [0, 0.5], color="red", linestyle="--")  # 理想線（y=x）
plt.xlabel("True Loss K")
plt.ylabel("Predicted Loss K")
plt.title("True vs Predicted Loss K")
plt.grid(True)
plt.xlim(0, 0.5)
plt.ylim(0, 0.5)
plt.gca().set_aspect("equal")
plt.savefig("data/results/K/loss_K_scatter.png")
plt.show()

# 2. Incorporation_Kの散布図
plt.figure(figsize=(6, 6))
plt.scatter(results_df["incorporation_K"], results_df["predict_incorp_K"], alpha=0.5)
plt.plot([0, 0.5], [0, 0.5], color="red", linestyle="--")  # 理想線（y=x）
plt.xlabel("True Incorporation K")
plt.ylabel("Predicted Incorporation K")
plt.title("True vs Predicted Incorporation K")
plt.grid(True)
plt.xlim(0, 0.5)
plt.ylim(0, 0.5)
plt.gca().set_aspect("equal")
plt.savefig("data/results/K/incorporation_K_scatter.png")
plt.show()
