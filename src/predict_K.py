import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# --- 1. データ読み込み ---
train_df = pd.read_csv("data/dataset/sampling/fold_0/train_fold.csv")
val_df = pd.read_csv("data/dataset/sampling/fold_0/validation_fold.csv")
test_df = pd.read_csv("data/dataset/sampling/fold_0/test_fold.csv")


# --- 2. Peptidoform単位にまとめる関数 ---
def prepare_peptide_level_df(df):
    # Cleaned_Peptidoformでまとめて、最初の行を取る
    grouped = df.groupby("Cleaned_Peptidoform").first().reset_index()
    return grouped[
        ["Cleaned_Peptidoform", "Peptido_Embedding_Path", "loss_K", "incorporation_K"]
    ]


train_peptide_df = prepare_peptide_level_df(train_df)
val_peptide_df = prepare_peptide_level_df(val_df)
test_peptide_df = prepare_peptide_level_df(test_df)


# --- 3. embeddingロード関数 ---
def load_embedding(path):
    try:
        data = np.load(path)
        emb = data["peptido_embedding"]
        emb = data["peptido_embedding"][1:-1]
        return emb.mean(axis=0)
    except Exception as e:
        print(f"[Warning] {path}のロードに失敗: {e}")
        return None


# --- 4. 各DataFrameにembedding追加 ---
def add_embeddings(df):
    embeddings = []
    for path in df["Peptido_Embedding_Path"]:
        emb = load_embedding(path)
        embeddings.append(emb)
    df["embedding"] = embeddings
    return df


train_peptide_df = add_embeddings(train_peptide_df)
val_peptide_df = add_embeddings(val_peptide_df)
test_peptide_df = add_embeddings(test_peptide_df)


# --- 5. embedding展開 ---
def expand_embeddings(df):
    emb_array = np.vstack(df["embedding"].values)
    return emb_array


X_train = expand_embeddings(train_peptide_df)
X_val = expand_embeddings(val_peptide_df)
X_test = expand_embeddings(test_peptide_df)

y_train_loss = train_peptide_df["loss_K"].values
y_train_incorp = train_peptide_df["incorporation_K"].values

y_val_loss = val_peptide_df["loss_K"].values
y_val_incorp = val_peptide_df["incorporation_K"].values

y_test_loss = test_peptide_df["loss_K"].values
y_test_incorp = test_peptide_df["incorporation_K"].values

# --- 6. モデル定義 ---
# model_loss = RandomForestRegressor(n_estimators=100, random_state=42)
# model_incorp = RandomForestRegressor(n_estimators=100, random_state=42)
model_loss = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    # tree_method="gpu_hist",  # ← これにするだけ！
)

model_incorp = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42,
    # tree_method="gpu_hist",  # ← こっちも
)
# --- 7. モデル学習 ---
model_loss.fit(X_train, y_train_loss)
model_incorp.fit(X_train, y_train_incorp)

# --- 8. 予測 ---
y_pred_loss = model_loss.predict(X_test)
y_pred_incorp = model_incorp.predict(X_test)


# --- 9. 評価 ---
def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"--- {name} ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")
    p_incorp, _ = spearmanr(y_true, y_pred)
    print(f"Spearman Correlation: {p_incorp:.4f}")


evaluate(y_test_loss, y_pred_loss, "Loss K Prediction")
evaluate(y_test_incorp, y_pred_incorp, "Incorporation K Prediction")
