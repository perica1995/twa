"""
BTC価格変動予測モデル
- 取引データから特徴量を生成し、短期的な価格上昇/下降を予測する
- RandomForest と LogisticRegression で比較
"""

import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler


def load_data(data_dir="."):
    """全CSVファイルを読み込んで結合"""
    files = sorted(glob.glob(f"{data_dir}/trades_BTC_*.csv"))
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("time").reset_index(drop=True)
    print(f"データ読み込み完了: {len(df):,}行, {len(files)}ファイル")
    return df


def create_features(df, window=50):
    """取引データから特徴量を生成"""
    df = df.copy()

    # 売買フラグ
    df["is_buy"] = (df["side"] == "B").astype(int)

    # ローリング特徴量
    df["px_ma"] = df["px"].rolling(window).mean()
    df["px_std"] = df["px"].rolling(window).std()
    df["sz_ma"] = df["sz"].rolling(window).mean()
    df["notional_ma"] = df["notional"].rolling(window).mean()

    # 買い比率（直近window件のうち買いの割合）
    df["buy_ratio"] = df["is_buy"].rolling(window).mean()

    # 価格の変化率
    df["px_return"] = df["px"].pct_change()
    df["px_return_ma"] = df["px_return"].rolling(window).mean()

    # 価格と移動平均の乖離率
    df["px_ma_diff"] = (df["px"] - df["px_ma"]) / df["px_ma"]

    # 出来高加重平均価格 (VWAP) との乖離
    df["cum_notional"] = df["notional"].rolling(window).sum()
    df["cum_sz"] = df["sz"].rolling(window).sum()
    df["vwap"] = df["cum_notional"] / df["cum_sz"]
    df["vwap_diff"] = (df["px"] - df["vwap"]) / df["vwap"]

    # ターゲット: 次のwindow件後の価格が上がるかどうか
    future_px = df["px"].shift(-window)
    df["target"] = (future_px > df["px"]).astype(int)

    # 欠損値を除去
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "px_std", "sz_ma", "notional_ma", "buy_ratio",
        "px_return_ma", "px_ma_diff", "vwap_diff"
    ]

    print(f"特徴量生成完了: {len(df):,}行, 特徴量{len(feature_cols)}個")
    return df, feature_cols


def train_and_evaluate(df, feature_cols):
    """モデルの学習と評価"""
    X = df[feature_cols].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # 時系列なのでシャッフルしない
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n学習データ: {len(X_train):,}件 / テストデータ: {len(X_test):,}件")
    print(f"ターゲット分布 (学習): 上昇={y_train.mean():.1%} / 下降={1-y_train.mean():.1%}")
    print(f"ターゲット分布 (テスト): 上昇={y_test.mean():.1%} / 下降={1-y_test.mean():.1%}")

    # --- RandomForest ---
    print("\n" + "=" * 50)
    print("RandomForest")
    print("=" * 50)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    print(f"正解率: {accuracy_score(y_test, rf_pred):.4f}")
    print(classification_report(y_test, rf_pred, target_names=["下降", "上昇"]))

    # 特徴量重要度
    print("特徴量重要度:")
    importances = sorted(
        zip(feature_cols, rf.feature_importances_), key=lambda x: -x[1]
    )
    for name, imp in importances:
        print(f"  {name:20s}: {imp:.4f}")

    # --- LogisticRegression ---
    print("\n" + "=" * 50)
    print("LogisticRegression")
    print("=" * 50)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    print(f"正解率: {accuracy_score(y_test, lr_pred):.4f}")
    print(classification_report(y_test, lr_pred, target_names=["下降", "上昇"]))

    return rf, lr, scaler


def main():
    print("=" * 50)
    print("BTC 価格変動予測 (機械学習)")
    print("=" * 50)

    df = load_data(".")
    print(f"価格範囲: ${df['px'].min():,.0f} - ${df['px'].max():,.0f}")
    print(f"期間: {df['datetime_utc'].iloc[0]} ~ {df['datetime_utc'].iloc[-1]}")

    df_feat, feature_cols = create_features(df, window=50)
    rf, lr, scaler = train_and_evaluate(df_feat, feature_cols)

    print("\n完了!")


if __name__ == "__main__":
    main()
