import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import os

from ui_components import result_card

# この app.py があるフォルダの場所を取得
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_PATH = os.path.join(BASE_DIR, 'model_artifacts.pkl')
MODEL_PATH = os.path.join(BASE_DIR, '1p19q_model.pth')

# 1 モデル定義
class SimpleNeuralNet(nn.Module):
    def __init__(self, input_dim, dropout_rates, layer_sizes):
        super(SimpleNeuralNet, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_dim, layer_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),
            nn.Linear(layer_sizes[0], layer_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(layer_sizes[1], layer_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rates[2]),
            nn.Linear(layer_sizes[2], layer_sizes[3]) 
        )

    def forward(self, x):
        return self.sequential(x)

# 2 前処理関数
def preprocess_data(df, artifacts):
    scaler = artifacts["scaler"]
    selected_idx = artifacts["selected_idx"]
    
    df_numeric = df.select_dtypes(include=[np.number])
    
    try:
        raw_values = df_numeric.values
        
        if raw_values.shape[1] == len(selected_idx):
            return raw_values.astype(np.float32)
        
        try:
            scaled_values = scaler.transform(raw_values)
        except ValueError:
            st.warning(f"注意: データ列数が異なるため正規化をスキップします")
            scaled_values = raw_values

        final_values = scaled_values[:, selected_idx]
        return final_values.astype(np.float32)

    except Exception as e:
        st.error(f"前処理エラー: {e}")
        return None

# 3 アプリ画面
st.set_page_config(page_title="脳腫瘍 1p/19q 共欠失予測", layout="centered")
st.title("脳腫瘍 1p/19q 共欠失予測 AI")

# 診断結果保存用 state
if "probs" not in st.session_state:
    st.session_state["probs"] = None
if "df" not in st.session_state:
    st.session_state["df"] = None

# ルールブック読み込み
try:
    with open(ARTIFACTS_PATH, 'rb') as f:
        artifacts = pickle.load(f)
    st.sidebar.success("AIモデル: 準備OK")
except Exception as e:
    st.sidebar.error(f"エラー: model_artifacts.pkl を開けませんでした → {ARTIFACTS_PATH}")
    st.sidebar.error(str(e))
    artifacts = None

# ファイルアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])

# 診断処理（ボタン内） ← session_state への保存のみ行う
if uploaded_file is not None and artifacts is not None:
    df = pd.read_csv(uploaded_file)
    st.write("読み込みデータ（先頭5行）:", df.head())

    if st.button("診断を開始する", type="primary"):
        with st.spinner('AIが診断中...'):

            input_data = preprocess_data(df, artifacts)
            if input_data is None:
                st.stop()

            try:
                tensor_data = torch.from_numpy(input_data)

                model = SimpleNeuralNet(64, [0.4, 0.3, 0.2], [128, 64, 32, 1])
                state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
                clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(clean_state_dict)
                model.eval()

                with torch.no_grad():
                    logits = model(tensor_data)
                    probs = torch.sigmoid(logits).numpy().flatten()

                # 結果を session_state に保存
                st.session_state["probs"] = probs
                st.session_state["df"] = df

            except Exception as e:
                st.error(f"予測エラー: {e}")
                st.error("入力データの形式が学習時と異なる可能性があります。")

# 診断結果の表示（ボタンの外に置く）
if st.session_state["probs"] is not None:

    probs = st.session_state["probs"]
    df = st.session_state["df"]

    # —— 統計 ——
    st.divider()
    st.subheader("集計")
    positive_count = sum(prob > 0.5 for prob in probs)
    negative_count = len(probs) - positive_count
    st.write(f"陽性（共欠失あり）: **{positive_count}件**")
    st.write(f"陰性（共欠失なし）: **{negative_count}件**")
    st.write(f"陽性率: **{(positive_count / len(probs)) * 100:.1f}%**")

    # —— ソート ——
    st.divider()
    st.subheader(f"診断結果 ({len(probs)}件)")

    sort_flag = st.checkbox("確率の高い順に並び替える")

    results = list(zip(range(len(probs)), probs))
    if sort_flag:
        results.sort(key=lambda x: x[1], reverse=True)

    # —— カード表示 ——
    for idx, prob in results:
        prob_percent = prob * 100
        positive = prob > 0.5

        res = "ポジティブ (共欠失あり)" if positive else "ネガティブ (共欠失なし)"
        patient_id = df.iloc[idx, 0] if 'patientID' in df.columns else f"症例 {idx+1}"

        result_card(patient_id, res, prob_percent, positive)