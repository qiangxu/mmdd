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
)
from utils.get_info import get_pnl, get_volumes
from utils.load_data import load_md_from_csv
from datetime import datetime
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED=13
N = 5000 # NUM OF EPOCHS
S = 200_000 # TRAIN SIZE
SS = 20_000 # EVALU SIZE, SHOULD BE CLOSE TO traj_size
#assert SS < S / 10
TRAJ_SIZE = 100 # N * traj_size << S 
#TRAJ_SIZE = 1797585 # N * traj_size << S 
DELAY = pd.Timedelta(0.1, "s").total_seconds()
HOLD_TIME = pd.Timedelta(60, "s").total_seconds()
FEATURES_LEN = int(HOLD_TIME)
LATENCY = pd.Timedelta(10, "ms").total_seconds()
DATENCY = pd.Timedelta(10, "ms").total_seconds()
N_ACTIONS = 3
TRADE_SIZE = 0.01
TAKER_FEE=0.0004
MAKER_FEE=0.00004


def get_tag(balance, strategy, datetime_from, datetime_to, epoch_i): 
    tag_fields = ["PNL", "COIN", "USDT", "PRICE", "DELAY", "LATENCY", "DATENCY", "HOLD_TIME", "TRAJ_SIZE", "S", "SS", "TRADE_SIZE", "FEATURES_LEN", "MAX_EPOCHS", "EPOCH_NO", "FROM", "TO"]
    res = {
        "COIN": "%06f" % balance[0],
        "USDT": "%02f" % balance[1], 
        "PRICE": "%02f" % balance[2], 
        "PNL": "%06f" % balance[3], 
        "FROM": datetime.fromtimestamp(datetime_from).strftime("%Y%m%dT%H%M%S"),
        "TO": datetime.fromtimestamp(datetime_to).strftime("%Y%m%dT%H%M%S"), 
        "DELAY": DELAY, 
        "HOLD_TIME": HOLD_TIME, 
        "TRAJ_SIZE": TRAJ_SIZE, 
        "TRADE_SIZE": TRADE_SIZE, 
        "FEATURES_LEN": FEATURES_LEN, 
        "LATENCY": LATENCY, 
        "DATENCY": DATENCY, 
        "S": S, 
        "SS": SS,
        "MAX_ECPHOS": N, 
        "EPOCH_NO": "%05d" % epoch_i,
        }
    
    print(res)
    return "_".join([f"{tf}_{res[tf]}" for tf in tag_fields if tf in res.keys()])

def get_strategy(policy, features, means, stds):
    return RLStrategy(
        policy,
        features,
        1.0,
        means,
        stds,
        FEATURES_LEN, 
        DELAY,
        HOLD_TIME,
        [ComputeValueTargets(policy)],
        trade_size=TRADE_SIZE,
        post_only=True,
        taker_fee=TAKER_FEE,
        maker_fee=MAKER_FEE,
    )
    
def preprocess_data(csv_books, csv_trades, num_rows=-1):
    return load_md_from_csv(csv_books, csv_trades, num_rows=num_rows)


def train(dataset, features, means, stds):
    # TODO: HOW DELAY, HOLD_TIME AFFECT
    # TODO: n_actions=10

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
        
    # breakpoint()
    model = A2CNetwork(n_actions=N_ACTIONS, num_features=features.shape[1]-1, features_len=FEATURES_LEN, device=DEVICE).to(DEVICE)
    policy = Policy(model)
    strategy = get_strategy(policy, features, means, stds)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=7e-4, alpha=0.99, eps=1e-5)

    agent = A2C(policy, optimizer, value_loss_coef=0.25, entropy_coef=1, device=DEVICE)

    checkpoints = []
    
    for i in tqdm(range(1, N+1)): 
        agent.train(
            strategy,
            dataset[0:-SS],
            latency=LATENCY, 
            datency=DATENCY,
            traj_size=TRAJ_SIZE,
        ) 

        if ((i > 9) and (i-1) % 1000 == 0) or i == N:
            reward, balance, trajectory = evaluate(
                strategy,
                dataset[-SS:],
                latency=LATENCY, 
                datency=DATENCY,
            )
            # breakpoint()
            tag = get_tag(balance, strategy, dataset[0].receive_ts, dataset[-1].receive_ts, i)
            
            checkpoint = "../models/%s.pth" % tag
            checkpoints.append(checkpoint)
            torch.save(model.state_dict(), checkpoint)
    return checkpoints

def test(checkpoint, dataset, features, means, stds):
    # breakpoint()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = A2CNetwork(n_actions=N_ACTIONS, num_features=features.shape[1]-1, features_len=FEATURES_LEN, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    policy = Policy(model)

    strategy = get_strategy(policy, features, means, stds)

    reward, balance, trajectory = evaluate(
        strategy,
        dataset,
        latency=LATENCY, 
        datency=DATENCY,
    )

def evaluate(strategy, dataset, latency, datency, unlimit=True):
    # breakpoint()
    strategy.reset()
    sim = Sim(dataset, latency, datency)
    with torch.no_grad():
        trades_list, md_list, updates_list, balance, actions_history, trajectory = strategy.run(sim, mode='test', unlimit=unlimit)
   
    return np.sum(trajectory['rewards']), balance, trajectory

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

    means_dict = features.mean(numeric_only=True).to_dict()
    means = [means_dict.get(c, 0) for c in features.columns[1:]]
    stds_dict = features.std(numeric_only=True).to_dict()
    stds = [stds_dict.get(c, 0) for c in features.columns[1:]]

    return features, np.array(means, dtype=float), np.array(stds, dtype=float)


def main():
    #1797585
    dataset = preprocess_data(
        "../data/books.csv", "../data/trades.csv", num_rows=S+10*SS
        #"../data/books.csv", "../data/trades.csv"
    )
        
    print("TOTAL DATA UPDATES: %d" % len(dataset))
    assert len(dataset) >= S + 2 * SS
    # dataset = preprocess_data("../data/books.csv", "../data/trades.csv")

    features, means, stds = prepare_features("../data/features.pickle")
  
    #for i in range(10):
    checkpoints = train(dataset[0:S+SS], features, means, stds)
    SSS =int(SS/10)
    
    """
    for j in range(10):
        test(checkpoints[-1], dataset[S+i*SS+j*SSS:S+i*SS+(j+1)*SSS], features, means, stds)
    """
    #FLUCTUATE
    #2023-01-14 22:46:52.021000 2023-01-15 01:52:23.517000 
    #[410_000:420_000] 
    #2023-01-17 02:21:40.553000 2023-01-17 21:00:56.927000
    #[550_000:600_000]
    #2023-01-19 01:23:31.349000 2023-01-19 04:01:45.552000
    #[700_000:710_000]


    #RISEUP
    #2023-01-14 06:43:42.542000 2023-01-14 08:44:02.336000
    #[310_000:330_000]
    #2023-01-12 22:43:49.164000 2023-01-13 02:35:46.721000
    #[200_000:220_000]
    #2023-01-20 09:08:44.838000 2023-01-21 06:03:06.315000
    #[780_000:850_000]

    #FALLDOWN
    #2023-01-18 22:18:10.844000 2023-01-19 07:48:57.254000
    #[670_000:720_000]
    #2023-01-16 15:28:45.603000 2023-01-16 15:28:45.603000
    #[520_000:530_000]
    #2023-01-14 15:31:13.110000 2023-01-14 18:40:09.836000
    #[370_000:390_000]

if __name__ == "__main__":
    main()
