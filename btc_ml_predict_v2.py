"""
BTC価格変動予測 v2 - バックテスト付き改善版
- 時間ベースの特徴量を追加
- ウォークフォワード検証（より現実的な評価）
- 損益シミュレーション（手数料込み）
"""

import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler


def load_data(data_dir="."):
    files = sorted(glob.glob(f"{data_dir}/trades_BTC_*.csv"))
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("time").reset_index(drop=True)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    print(f"データ読み込み完了: {len(df):,}行, {len(files)}ファイル")
    return df


def aggregate_to_bars(df, bar_seconds=60):
    """取引データを時間足（バー）に集約 — より安定した特徴量を生成"""
    df = df.copy()
    df["bar"] = (df["time"] // (bar_seconds * 1000)) * (bar_seconds * 1000)

    bars = df.groupby("bar").agg(
        open=("px", "first"),
        high=("px", "max"),
        low=("px", "min"),
        close=("px", "last"),
        volume=("sz", "sum"),
        notional=("notional", "sum"),
        trade_count=("px", "count"),
        buy_volume=("sz", lambda x: x[df.loc[x.index, "side"] == "B"].sum()),
        datetime_utc=("datetime_utc", "first"),
    ).reset_index()

    bars["vwap"] = bars["notional"] / bars["volume"]
    bars["buy_ratio"] = bars["buy_volume"] / bars["volume"]
    print(f"{bar_seconds}秒足に集約: {len(bars):,}本")
    return bars


def create_features(bars):
    """バーデータから特徴量を生成"""
    df = bars.copy()

    # 価格系
    df["return_1"] = df["close"].pct_change(1)
    df["return_5"] = df["close"].pct_change(5)
    df["return_10"] = df["close"].pct_change(10)

    # ボラティリティ
    df["volatility_10"] = df["return_1"].rolling(10).std()
    df["volatility_30"] = df["return_1"].rolling(30).std()
    df["vol_ratio"] = df["volatility_10"] / df["volatility_30"]

    # 価格レンジ
    df["bar_range"] = (df["high"] - df["low"]) / df["close"]
    df["bar_range_ma"] = df["bar_range"].rolling(10).mean()

    # 移動平均
    for w in [5, 10, 20, 50]:
        df[f"ma_{w}"] = df["close"].rolling(w).mean()
    df["ma_cross_short"] = (df["ma_5"] - df["ma_10"]) / df["close"]
    df["ma_cross_long"] = (df["ma_10"] - df["ma_50"]) / df["close"]
    df["ma_trend"] = (df["close"] - df["ma_20"]) / df["close"]

    # 出来高系
    df["vol_ma_10"] = df["volume"].rolling(10).mean()
    df["vol_ratio_bar"] = df["volume"] / df["vol_ma_10"]
    df["buy_ratio_ma"] = df["buy_ratio"].rolling(10).mean()

    # VWAP乖離
    df["vwap_diff"] = (df["close"] - df["vwap"]) / df["close"]

    # RSI (14期間)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # 取引頻度
    df["trade_count_ma"] = df["trade_count"].rolling(10).mean()
    df["trade_intensity"] = df["trade_count"] / df["trade_count_ma"]

    feature_cols = [
        "return_1", "return_5", "return_10",
        "volatility_10", "vol_ratio",
        "bar_range", "bar_range_ma",
        "ma_cross_short", "ma_cross_long", "ma_trend",
        "vol_ratio_bar", "buy_ratio_ma",
        "vwap_diff", "rsi", "trade_intensity",
    ]

    return df, feature_cols


def walk_forward_backtest(bars, feature_cols, horizon=5, train_size=500, step=100, fee_pct=0.04):
    """
    ウォークフォワード検証 + 損益シミュレーション
    - train_size本で学習 → 次のstep本で予測・取引を繰り返す
    - fee_pct: 片道手数料 (%) — 往復で2倍かかる
    """
    df, feature_cols = create_features(bars)

    # ターゲット: horizon本後の価格変動
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df["target"] = (df["future_return"] > 0).astype(int)
    df = df.dropna(subset=feature_cols + ["target", "future_return"]).reset_index(drop=True)

    print(f"\n特徴量生成完了: {len(df):,}本, 特徴量{len(feature_cols)}個")
    print(f"ウォークフォワード: 学習{train_size}本, ステップ{step}本, 予測horizon={horizon}本")
    print(f"手数料: 片道{fee_pct}% (往復{fee_pct*2}%)\n")

    all_predictions = []
    scaler = StandardScaler()

    start = train_size
    while start + step <= len(df):
        end = min(start + step, len(df))

        # 学習データ
        train_df = df.iloc[start - train_size:start]
        test_df = df.iloc[start:end]

        X_train = train_df[feature_cols].values
        y_train = train_df["target"].values
        X_test = test_df[feature_cols].values

        scaler.fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        # GradientBoosting
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        model.fit(X_train_s, y_train)
        proba = model.predict_proba(X_test_s)[:, 1]

        for i, idx in enumerate(range(start, end)):
            all_predictions.append({
                "idx": idx,
                "datetime": df.iloc[idx]["datetime_utc"],
                "close": df.iloc[idx]["close"],
                "proba": proba[i],
                "actual_return": df.iloc[idx]["future_return"],
                "actual_target": df.iloc[idx]["target"],
            })

        start += step

    results = pd.DataFrame(all_predictions)
    print(f"予測数: {len(results):,}")

    # --- 評価 ---
    # 1) 分類精度
    results["pred"] = (results["proba"] > 0.5).astype(int)
    acc = accuracy_score(results["actual_target"], results["pred"])
    print(f"\n全体正解率: {acc:.4f}")
    print(classification_report(
        results["actual_target"], results["pred"],
        target_names=["下降", "上昇"]
    ))

    # 2) 信頼度フィルタ付き取引シミュレーション
    print("=" * 60)
    print("損益シミュレーション (信頼度別)")
    print("=" * 60)

    for threshold in [0.50, 0.55, 0.60]:
        trades = results[
            (results["proba"] > threshold) | (results["proba"] < (1 - threshold))
        ].copy()

        if len(trades) == 0:
            print(f"\n閾値 {threshold:.0%}: 取引なし")
            continue

        # ロング/ショートのポジション
        trades["position"] = np.where(trades["proba"] > 0.5, 1, -1)
        trades["gross_pnl"] = trades["position"] * trades["actual_return"]
        trades["net_pnl"] = trades["gross_pnl"] - (fee_pct / 100 * 2)  # 往復手数料

        total_trades = len(trades)
        win_rate = (trades["net_pnl"] > 0).mean()
        total_return = trades["net_pnl"].sum()
        avg_return = trades["net_pnl"].mean()
        sharpe = trades["net_pnl"].mean() / trades["net_pnl"].std() * np.sqrt(total_trades) if trades["net_pnl"].std() > 0 else 0

        print(f"\n閾値 {threshold:.0%}:")
        print(f"  取引回数:     {total_trades:,}")
        print(f"  勝率:         {win_rate:.1%}")
        print(f"  累積リターン: {total_return:+.4%}")
        print(f"  平均リターン: {avg_return:+.6%} /取引")
        print(f"  シャープ比:   {sharpe:.2f}")

        if total_return > 0:
            print(f"  → 手数料控除後もプラス ✓")
        else:
            print(f"  → 手数料控除後マイナス ✗")

    # 3) 特徴量重要度（最後のモデル）
    print(f"\n{'=' * 60}")
    print("特徴量重要度 (最終モデル)")
    print("=" * 60)
    importances = sorted(
        zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]
    )
    for name, imp in importances:
        bar = "█" * int(imp * 100)
        print(f"  {name:20s}: {imp:.4f} {bar}")

    return results


def main():
    print("=" * 60)
    print("BTC 価格変動予測 v2 — バックテスト付き")
    print("=" * 60)

    df = load_data(".")
    print(f"価格範囲: ${df['px'].min():,.0f} - ${df['px'].max():,.0f}")
    print(f"期間: {df['datetime_utc'].iloc[0]} ~ {df['datetime_utc'].iloc[-1]}")

    bars = aggregate_to_bars(df, bar_seconds=30)  # 30秒足（データ量を増やす）
    results = walk_forward_backtest(
        bars, feature_cols=None,
        horizon=5,       # 5本(2.5分)後の価格を予測
        train_size=300,  # 直近300本で学習
        step=50,         # 50本ごとにモデル再学習
        fee_pct=0.04,    # 片道0.04% (Binanceメーカー手数料相当)
    )

    print(f"\n{'=' * 60}")
    print("結論")
    print("=" * 60)
    print("・ウォークフォワード検証は過学習を防ぐ最も現実的な評価方法")
    print("・手数料控除後にプラスでなければ実運用は危険")
    print("・このモデルだけで「勝てる」とは限らない — リスク管理が最重要")
    print("・改善案: より多くのデータ、オーダーブック情報、アンサンブル手法")


if __name__ == "__main__":
    main()
