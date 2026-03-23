"""
大口注文吸収パターン検出器 (Absorption Detector)
- 大口の買い(売り)が入った後、価格が動かない or 逆行する場面を検出
- 「見えない対抗勢力」が密かに吸収しているサイン
"""

import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler


def load_data(data_dir="."):
    files = sorted(glob.glob(f"{data_dir}/trades_BTC_*.csv"))
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values("time").reset_index(drop=True)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df["is_buy"] = (df["side"] == "B").astype(int)
    df["signed_notional"] = np.where(df["is_buy"], df["notional"], -df["notional"])
    print(f"データ読み込み: {len(df):,}行")
    return df


def build_bars(df, bar_seconds=30):
    """30秒バーに集約"""
    df["bar"] = (df["time"] // (bar_seconds * 1000)) * (bar_seconds * 1000)

    bars = df.groupby("bar").agg(
        open=("px", "first"),
        high=("px", "max"),
        low=("px", "min"),
        close=("px", "last"),
        volume=("sz", "sum"),
        notional=("notional", "sum"),
        trade_count=("px", "count"),
        buy_notional=("notional", lambda x: x[df.loc[x.index, "is_buy"] == 1].sum()),
        sell_notional=("notional", lambda x: x[df.loc[x.index, "is_buy"] == 0].sum()),
        # 大口取引 (1BTC以上 ≈ $69K+)
        whale_buy_notional=("notional", lambda x: x[(df.loc[x.index, "is_buy"] == 1) & (df.loc[x.index, "sz"] >= 0.5)].sum()),
        whale_sell_notional=("notional", lambda x: x[(df.loc[x.index, "is_buy"] == 0) & (df.loc[x.index, "sz"] >= 0.5)].sum()),
        whale_count=("sz", lambda x: (x >= 0.5).sum()),
        max_trade_sz=("sz", "max"),
        datetime_utc=("datetime_utc", "first"),
    ).reset_index()

    bars["delta"] = bars["buy_notional"] - bars["sell_notional"]
    bars["whale_delta"] = bars["whale_buy_notional"] - bars["whale_sell_notional"]
    bars["return"] = bars["close"].pct_change()
    bars["price_move"] = bars["close"] - bars["open"]

    print(f"{bar_seconds}秒バー: {len(bars):,}本")
    return bars


def detect_absorption_events(bars, lookback=5, forward=10):
    """
    吸収パターンを検出:
    大口買い(売り)が集中した区間の後、価格が期待ほど動かない or 逆行する

    吸収の定義:
    - 直近lookback本で大口デルタが大きい（方向性のある注文フロー）
    - しかし次のforward本で価格が同方向に動かない（吸収されている）
    """
    df = bars.copy()

    # ローリング集計
    df["cum_delta"] = df["delta"].rolling(lookback).sum()
    df["cum_whale_delta"] = df["whale_delta"].rolling(lookback).sum()
    df["cum_notional"] = df["notional"].rolling(lookback).sum()
    df["cum_whale_count"] = df["whale_count"].rolling(lookback).sum()

    # 将来の価格変動
    df["future_return"] = df["close"].shift(-forward) / df["close"] - 1

    # 吸収スコア: デルタの方向と将来リターンの不一致度
    # デルタが大きく正なのに価格が上がらない → 売り吸収
    # デルタが大きく負なのに価格が下がらない → 買い吸収
    delta_direction = np.sign(df["cum_delta"])
    return_direction = np.sign(df["future_return"])
    df["absorbed"] = (delta_direction != return_direction).astype(int)

    df = df.dropna().reset_index(drop=True)
    return df


def create_absorption_features(bars, lookback=5):
    """吸収パターン用の特徴量"""
    df = bars.copy()

    # --- フロー特徴量 ---
    df["cum_delta"] = df["delta"].rolling(lookback).sum()
    df["cum_whale_delta"] = df["whale_delta"].rolling(lookback).sum()
    df["cum_notional"] = df["notional"].rolling(lookback).sum()

    # デルタの偏り（全体に対するネットフローの比率）
    df["delta_imbalance"] = df["cum_delta"] / df["cum_notional"]

    # 大口 vs 小口のデルタ乖離
    small_delta = df["cum_delta"] - df["cum_whale_delta"]
    df["whale_vs_retail"] = np.where(
        df["cum_whale_delta"].abs() > 0,
        small_delta / df["cum_whale_delta"].abs(),
        0
    )

    # 大口の出現頻度
    df["whale_intensity"] = df["whale_count"].rolling(lookback).sum()

    # --- 価格反応の弱さ（吸収の兆候）---
    # デルタに対する価格変動の効率性
    cum_price_move = df["price_move"].rolling(lookback).sum()
    df["price_efficiency"] = np.where(
        df["cum_delta"].abs() > 0,
        cum_price_move / (df["cum_delta"].abs() / df["cum_notional"] * df["close"]),
        0
    )

    # 価格のレンジ vs デルタ（レンジが狭いのにデルタが大きい = 吸収中）
    df["range_lookback"] = df["high"].rolling(lookback).max() - df["low"].rolling(lookback).min()
    df["range_vs_delta"] = np.where(
        df["cum_delta"].abs() > 0,
        df["range_lookback"] / (df["cum_delta"].abs() / 1e6),
        0
    )

    # --- ボラティリティ ---
    df["volatility"] = df["return"].rolling(lookback * 2).std()
    df["vol_change"] = df["volatility"] / df["volatility"].shift(lookback) - 1

    # --- 出来高プロファイル ---
    df["volume_ma"] = df["notional"].rolling(lookback * 4).mean()
    df["volume_surge"] = df["notional"] / df["volume_ma"]

    # 小口取引の細かさ（刻み取引 = iceberg注文の可能性）
    df["avg_trade_size"] = df["notional"] / df["trade_count"]
    df["avg_size_ma"] = df["avg_trade_size"].rolling(lookback * 4).mean()
    df["trade_fragmentation"] = df["avg_size_ma"] / df["avg_trade_size"]

    feature_cols = [
        "delta_imbalance", "whale_vs_retail", "whale_intensity",
        "price_efficiency", "range_vs_delta",
        "volatility", "vol_change",
        "volume_surge", "trade_fragmentation",
    ]

    return df, feature_cols


def analyze_absorption_patterns(bars):
    """吸収パターンの統計分析"""
    print("\n" + "=" * 60)
    print("吸収パターン分析")
    print("=" * 60)

    for lookback in [3, 5, 10]:
        events = detect_absorption_events(bars, lookback=lookback, forward=10)
        if len(events) == 0:
            continue

        # 大口買いが入った区間
        strong_buy = events[events["cum_delta"] > events["cum_delta"].quantile(0.75)]
        # そのうち吸収されたもの
        absorbed_buy = strong_buy[strong_buy["absorbed"] == 1]

        # 大口売りが入った区間
        strong_sell = events[events["cum_delta"] < events["cum_delta"].quantile(0.25)]
        absorbed_sell = strong_sell[strong_sell["absorbed"] == 1]

        print(f"\n--- lookback={lookback}本 ---")
        print(f"大口買い圧力の区間: {len(strong_buy)}件")
        print(f"  うち吸収(価格上がらず): {len(absorbed_buy)}件 ({len(absorbed_buy)/len(strong_buy)*100:.1f}%)")
        print(f"  吸収後の平均リターン: {absorbed_buy['future_return'].mean()*100:+.3f}%")
        print(f"  非吸収の平均リターン: {strong_buy[strong_buy['absorbed']==0]['future_return'].mean()*100:+.3f}%")

        print(f"大口売り圧力の区間: {len(strong_sell)}件")
        print(f"  うち吸収(価格下がらず): {len(absorbed_sell)}件 ({len(absorbed_sell)/len(strong_sell)*100:.1f}%)")
        print(f"  吸収後の平均リターン: {absorbed_sell['future_return'].mean()*100:+.3f}%")
        print(f"  非吸収の平均リターン: {strong_sell[strong_sell['absorbed']==0]['future_return'].mean()*100:+.3f}%")

        # 大口が吸収されるとき、小口はどう動いているか
        if len(absorbed_buy) > 0:
            avg_whale_vs_retail = absorbed_buy["whale_vs_retail"].mean() if "whale_vs_retail" in absorbed_buy.columns else 0
            print(f"\n  [吸収時の大口vs小口デルタ比]: {avg_whale_vs_retail:.2f}")
            print(f"  → 正=小口が大口と逆方向（対抗）, 負=小口も同方向")


def walk_forward_absorption(bars, forward=10):
    """吸収パターンベースのウォークフォワード検証"""
    print("\n" + "=" * 60)
    print("吸収パターン予測モデル (ウォークフォワード)")
    print("=" * 60)

    df, feature_cols = create_absorption_features(bars, lookback=5)

    # ターゲット: forward本後の価格方向
    df["future_return"] = df["close"].shift(-forward) / df["close"] - 1
    df["target"] = (df["future_return"] > 0).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=feature_cols + ["target", "future_return"]
    ).reset_index(drop=True)

    print(f"データ: {len(df):,}本, 特徴量{len(feature_cols)}個")

    train_size = 300
    step = 50
    fee_pct = 0.04
    all_preds = []
    scaler = StandardScaler()

    start = train_size
    while start + step <= len(df):
        end = min(start + step, len(df))
        train = df.iloc[start - train_size:start]
        test = df.iloc[start:end]

        X_train = train[feature_cols].values
        y_train = train["target"].values
        X_test = test[feature_cols].values

        scaler.fit(X_train)

        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        model.fit(scaler.transform(X_train), y_train)
        proba = model.predict_proba(scaler.transform(X_test))[:, 1]

        for i, idx in enumerate(range(start, end)):
            all_preds.append({
                "idx": idx,
                "datetime": df.iloc[idx]["datetime_utc"],
                "close": df.iloc[idx]["close"],
                "proba": proba[i],
                "actual_return": df.iloc[idx]["future_return"],
                "actual_target": df.iloc[idx]["target"],
                "delta_imbalance": df.iloc[idx]["delta_imbalance"],
            })

        start += step

    results = pd.DataFrame(all_preds)
    results["pred"] = (results["proba"] > 0.5).astype(int)

    print(f"\n予測数: {len(results):,}")
    acc = accuracy_score(results["actual_target"], results["pred"])
    print(f"全体正解率: {acc:.4f}")
    print(classification_report(
        results["actual_target"], results["pred"],
        target_names=["下降", "上昇"]
    ))

    # 吸収パターン発生時のみ取引
    print("=" * 60)
    print("戦略: 吸収検出時のみ逆張り取引")
    print("=" * 60)

    for imb_thresh in [0.1, 0.2, 0.3]:
        # デルタが偏っている区間（大口が片方に寄っている）
        biased = results[results["delta_imbalance"].abs() > imb_thresh].copy()
        if len(biased) == 0:
            continue

        # 吸収を予測: デルタ方向と逆のポジション
        # （大口買い集中 → モデルが下降予測 → ショート = 吸収後の反転を狙う）
        biased["position"] = np.where(biased["proba"] > 0.5, 1, -1)
        biased["gross_pnl"] = biased["position"] * biased["actual_return"]
        biased["net_pnl"] = biased["gross_pnl"] - (fee_pct / 100 * 2)

        total = len(biased)
        win_rate = (biased["net_pnl"] > 0).mean()
        cum_return = biased["net_pnl"].sum()
        avg_return = biased["net_pnl"].mean()
        sharpe = (biased["net_pnl"].mean() / biased["net_pnl"].std() * np.sqrt(total)
                  if biased["net_pnl"].std() > 0 else 0)

        mark = "+" if cum_return > 0 else " "
        print(f"\nデルタ偏り閾値 > {imb_thresh:.0%}:")
        print(f"  取引回数:     {total:,}")
        print(f"  勝率:         {win_rate:.1%}")
        print(f"  累積リターン: {mark}{cum_return:.4%}")
        print(f"  平均リターン: {mark}{avg_return:.6%} /取引")
        print(f"  シャープ比:   {sharpe:.2f}")

    # 特徴量重要度
    print(f"\n{'=' * 60}")
    print("特徴量重要度")
    print("=" * 60)
    importances = sorted(
        zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]
    )
    for name, imp in importances:
        bar = "█" * int(imp * 100)
        print(f"  {name:22s}: {imp:.4f} {bar}")

    return results


def main():
    print("=" * 60)
    print("大口注文吸収パターン検出器")
    print("Whale Absorption Detector")
    print("=" * 60)

    df = load_data(".")
    print(f"価格: ${df['px'].min():,.0f} - ${df['px'].max():,.0f}")
    print(f"期間: {df['datetime_utc'].iloc[0]} ~ {df['datetime_utc'].iloc[-1]}")

    # 大口取引の統計
    whale_thresh = 0.5  # BTC
    whales = df[df["sz"] >= whale_thresh]
    whale_buys = whales[whales["is_buy"] == 1]
    whale_sells = whales[whales["is_buy"] == 0]
    print(f"\n大口取引 (>= {whale_thresh} BTC):")
    print(f"  買い: {len(whale_buys):,}件, ${whale_buys['notional'].sum():,.0f}")
    print(f"  売り: {len(whale_sells):,}件, ${whale_sells['notional'].sum():,.0f}")
    print(f"  ネット: ${whale_buys['notional'].sum() - whale_sells['notional'].sum():+,.0f}")

    bars = build_bars(df, bar_seconds=30)

    # 吸収パターンの統計分析
    df_abs, _ = create_absorption_features(bars, lookback=5)
    df_abs = df_abs.replace([np.inf, -np.inf], np.nan).dropna()
    analyze_absorption_patterns(df_abs)

    # ウォークフォワード検証
    walk_forward_absorption(bars, forward=10)

    print(f"\n{'=' * 60}")
    print("考察")
    print("=" * 60)
    print("・大口買いの後に価格が上がらない = 誰かがステルスで吸収している")
    print("・吸収パターンは「見せ板」「アイスバーグ注文」の痕跡")
    print("・板情報(オーダーブック)があればより精密な検出が可能")
    print("・この検出器はアルファの種 — 単独で勝つにはさらなる改良が必要")


if __name__ == "__main__":
    main()
