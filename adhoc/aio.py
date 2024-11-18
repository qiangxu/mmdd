# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
import sys

sys.path.append("../")

import pickle
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from simulator.simulator import Sim
from strategies.rl import (
    A2CNetwork,
    Policy,
    RLStrategy,
    A2C,
    ComputeValueTargets,
    evaluate,
)
from utils.get_info import get_pnl, get_volumes
from utils.load_data import load_md_from_csv
from datetime import datetime
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED=13
N = 300 # NUM OF EPOCHS
S = 500_000 # TRAIN SIZE
SS = 50_000 # EVALU SIZE, SHOULD BE CLOSE TO traj_size
#assert SS < S / 10
TRAJ_SIZE = 5000 # N * traj_size << S 
DELAY = pd.Timedelta(0.1, "s").total_seconds()
HOLD_TIME = pd.Timedelta(10, "s").total_seconds()
LATENCY = pd.Timedelta(10, "ms").total_seconds()
DATENCY = pd.Timedelta(10, "ms").total_seconds()


def preprocess_data(csv_books, csv_trades, num_rows=-1):
    return load_md_from_csv(csv_books, csv_trades, num_rows=num_rows)


def train(dataset, features, means, stds):
    # TODO: HOW DELAY, HOLD_TIME AFFECT
    # TODO: n_actions=10

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = A2CNetwork(n_actions=10, device=DEVICE).to(DEVICE)
    policy = Policy(model)


    strategy = RLStrategy(
        policy,
        features,
        1.0,
        means,
        stds,
        DELAY,
        HOLD_TIME,
        [ComputeValueTargets(policy)],
        trade_size=0.01,
        post_only=True,
        taker_fee=0.0004,
        maker_fee=-0.00004,
    )

    optimizer = torch.optim.RMSprop(model.parameters(), lr=7e-4, alpha=0.99, eps=1e-5)

    agent = A2C(policy, optimizer, value_loss_coef=0.25, entropy_coef=1, device=DEVICE)

    checkpoints = []
    
    for i in tqdm(range(1, N+1)): 
        agent.train(
            strategy,
            dataset[0:S],
            latency=LATENCY, 
            datency=DATENCY,
            traj_size=TRAJ_SIZE,
        ) 


        if i % 50 == 0 or i == N:
            reward, PnL, trajectory = evaluate(
                strategy,
                dataset[S:S+SS],
                latency=LATENCY, 
                datency=DATENCY,
            )
            res = {
            "PnL": PnL,
            "from": datetime.fromtimestamp(dataset[0].receive_ts).strftime("%Y-%m-%dT%H-%M-%S"),
            "to": datetime.fromtimestamp(dataset[-1].receive_ts).strftime("%Y-%m-%dT%H-%M-%S"), 
            "delay": DELAY, 
            "hold_time": HOLD_TIME, 
            "traj_size": TRAJ_SIZE, 
            "latency": LATENCY, 
            "datency": DATENCY, 
            "S": S, 
            "SS": SS,
            "num_ecphos": N, 
            "epoch_i": i}
            print(res)

            checkpoint = "../models/rl_%s.pth" % "_".join([f"{k}_{v}" for k, v in sorted(res.items())])
            checkpoints.append(checkpoint)

            torch.save(model.state_dict(), checkpoint)

    return checkpoints

def test(checkpoint, dataset, features, means, stds):
    breakpoint()
    model = A2CNetwork(n_actions=10, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    policy = Policy(model)

    strategy = RLStrategy(
        policy,
        features,
        1.0,
        means,
        stds,
        DELAY,
        HOLD_TIME,
        [ComputeValueTargets(policy)],
        trade_size=0.01,
        post_only=True,
        taker_fee=0.0004,
        maker_fee=-0.00004,
    )

    strategy.reset()
    sim = Sim(
        dataset, 
        execution_latency=LATENCY,
        datency=DATENCY,
    )
    with torch.no_grad():

        trades_list, md_list, updates_list, actions_history, trajectory = strategy.run(
            sim, mode="test"
        )

        res = {
        "PnL": strategy.realized_pnl + strategy.unrealized_pnl,
        "from": datetime.fromtimestamp(dataset[0].receive_ts).strftime("%Y-%m-%dT%H-%M-%S"),
        "to": datetime.fromtimestamp(dataset[-1].receive_ts).strftime("%Y-%m-%dT%H-%M-%S"), 
        "delay": DELAY, 
        "hold_time": HOLD_TIME, 
        "traj_size": TRAJ_SIZE, 
        "latency": LATENCY, 
        "datency": DATENCY, 
        "S": S, 
        "SS": SS,
        }

        print(res)
        return trades_list, md_list, updates_list, actions_history, trajectory

def visualize(trades_list, md_list, updates_list, actions_history, trajectory):
    df = get_pnl(updates_list, post_only=True)
    breakpoint() 
    #TODO: MAKE SURE df["receive_ts"] IS DATETIME
    df["receive_ts"] = pd.to_datetime(df["receive_ts"])

    # +
    # %%time
    breakpoint() 
    #TODO: MAKE SURE df["receive_ts"] IS DATETIME

    df_fee = get_pnl(updates_list, post_only=True, maker_fee=-0.00004, taker_fee=0.0004)
    df_fee["receive_ts"] = pd.to_datetime(df_fee["receive_ts"])
    # -

    ask_made, bid_made, ask_take, bid_take = get_volumes(trades_list)
    ask_made, bid_made, ask_take, bid_take

    # +
    fig = make_subplots(
        rows=3, cols=1, subplot_titles=("Price", "PnL", "Inventory Size")
    )

    fig.add_trace(
        go.Scatter(
            x=df.iloc[::100, :]["receive_ts"],
            y=df.iloc[::100, :]["mid_price"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.iloc[::100, :]["receive_ts"],
            y=df.iloc[::100, :]["total"],
            name="PnL without fees",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_fee.iloc[::100, :]["receive_ts"],
            y=df_fee.iloc[::100, :]["total"],
            name="PnL including fees",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df.iloc[::100, :]["receive_ts"],
            y=df.iloc[::100, :]["BTC"],
            name="Inventory Size",
        ),
        row=3,
        col=1,
    )

    # fig.add_trace(g
    # o.Scatter(x=pd.to_datetime(actions[0])[::100], y=actions[2][::100], mode='markers',
    #                          marker_color=actions[2][::100],
    #                          name='Actions'), row=4, col=1)

    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="USD", row=2, col=1)
    fig.update_yaxes(title_text="BTC", row=3, col=1)
    # fig.update_yaxes(title_text="Actions ID", row=4, col=1)

    fig.update_layout(title_text="RL Strategy: maker fee = -0.004%", height=700)

    # fig.write_html('../docs/RLStrategy.html')
    fig.write_image("../images/results/RLStrategy.jpeg")
    # fig.show()

    # +
    import matplotlib.pyplot as plt
    import numpy as np

    # 创建三个子图
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8), sharex=True)

    # 子图1: 价格图
    axes[0].plot(df.iloc[::100]["receive_ts"], df.iloc[::100]["mid_price"])
    axes[0].set_title("Price")
    axes[0].set_ylabel("USD")

    # 子图2: PnL图
    axes[1].plot(
        df.iloc[::100]["receive_ts"], df.iloc[::100]["total"], label="PnL without fees"
    )
    axes[1].plot(
        df_fee.iloc[::100]["receive_ts"],
        df_fee.iloc[::100]["total"],
        label="PnL including fees",
    )
    axes[1].set_title("PnL")
    axes[1].set_ylabel("USD")
    axes[1].legend()

    # 子图3: 仓位图
    axes[2].plot(df.iloc[::100]["receive_ts"], df.iloc[::100]["BTC"])
    axes[2].set_title("Inventory Size")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("BTC")

    # 整体布局设置
    plt.suptitle("RL Strategy: maker fee = -0.004%")
    plt.tight_layout(pad=2.0)

    # 保存图像
    plt.show()
    # -

    actions = {i: 0 for i in range(11)}
    for _, _, action_id in actions_history:
        actions[action_id] += 1


def prepare_features(features_snapshot):
    # TODO: MOVE THE FEATURES NOTEBOOK HERE
    with open(features_snapshot, "rb") as f:
        features_dict = pickle.load(f)

    # TODO: DETERMINE THE RIGHT DATA TYPE OF THE RECEIVE_TS COLUMN
    features = (
        pd.DataFrame.from_dict(features_dict, orient="index")
        .reset_index()
        .rename(columns={"index": "receive_ts"})
    )
    features["inventory_ratio"] = 0.0
    features["tpnl"] = 0.0


    means_dict = features.mean(numeric_only=True).to_dict()
    means = [means_dict.get(c, 0) for c in features.columns[1:]]
    stds_dict = features.std(numeric_only=True).to_dict()
    stds = [stds_dict.get(c, 0) for c in features.columns[1:]]

    return features, np.array(means, dtype=float), np.array(stds, dtype=float)


def main():
    dataset = preprocess_data(
        "../data/books.csv", "../data/trades.csv", num_rows=3 * S
    )
        
    print("TOTAL DATA UPDATES: %d" % len(dataset))
    assert len(dataset) >= S + 2 * SS
    # dataset = preprocess_data("../data/books.csv", "../data/trades.csv")

    features, means, stds = prepare_features("../data/features.pickle")
    
    #checkpoints = train(dataset[0:S+SS], features, means, stds)
    checkpoint = "../models/rl_PnL_1175.6750284805894_S_500000_SS_50000_datency_0.01_delay_0.1_epoch_i_200_from_2023-01-09T08-00-00_hold_time_10.0_latency_0.01_num_ecphos_300_to_2023-01-13T16-21-33_traj_size_5000.pth"
    trades_list, md_list, updates_list, actions_history, trajectory = test(checkpoint, dataset[S:S+2000], features, means, stds)
    breakpoint()
    df = get_pnl(updates_list, post_only=True)
    df = get_pnl(updates_list, post_only=True)
    df = get_pnl(updates_list, post_only=True)
    df = get_pnl(updates_list, post_only=True)
    df = get_pnl(updates_list, post_only=True)
    df = get_pnl(updates_list, post_only=True)

if __name__ == "__main__":
    main()
