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
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED=13
N = 100 # NUM OF EPOCHS
S = 200_000 # TRAIN SIZE
SS = 5_000 # EVALU SIZE, SHOULD BE CLOSE TO traj_size
#assert SS < S / 10
TRAJ_SIZE = 100 # N * traj_size << S 
#TRAJ_SIZE = 1797585 # N * traj_size << S 
#DELAY = pd.Timedelta(0.1, "s").total_seconds()
DELAY = pd.Timedelta(3, "s").total_seconds()
HOLD_TIME = pd.Timedelta(300, "s").total_seconds()
FEATURES_LEN = int(300)
LATENCY = pd.Timedelta(10, "ms").total_seconds()
DATENCY = pd.Timedelta(10, "ms").total_seconds()
N_ACTIONS = 16
TRADE_SIZE = 0.01
TAKER_FEE=0.0004
MAKER_FEE=0.00004


def get_tag(run_res, strategy, datetime_from, datetime_to, epoch_i): 
    tag_fields = ["PNL", "COIN", "USDT", "PRICE", "DELAY", "LATENCY", "DATENCY", "HOLD_TIME", "TRAJ_SIZE", "S", "SS", "TRADE_SIZE", "FEATURES_LEN", "MAX_EPOCHS", "EPOCH_NO", "FROM", "TO"]
    res = {
        "COIN": "%02f" % run_res["_position"]["coin"][1],
        "USDT": "%02f" % run_res["_position"]["usdt"][1],
        "FROM": datetime.utcfromtimestamp(datetime_from).strftime("%Y%m%dT%H%M%S"),
        "TO": datetime.utcfromtimestamp(datetime_to).strftime("%Y%m%dT%H%M%S"), 
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

        if ((i > 9) and (i-1) % 100 == 0) or i == N:
            # breakpoint()
            res = evaluate(
                strategy,
                dataset[-SS:],
                latency=LATENCY, 
                datency=DATENCY,
            )
            tag = get_tag(res, strategy, dataset[0].receive_ts, dataset[-1].receive_ts, i)
            
            checkpoint = "../models/%s.pth" % tag
            checkpoints.append(checkpoint)
            torch.save(model.state_dict(), checkpoint)
    return checkpoints

def test(checkpoint, dataset, features, means, stds):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    breakpoint()
    model = A2CNetwork(n_actions=N_ACTIONS, num_features=features.shape[1]-1, features_len=FEATURES_LEN, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    policy = Policy(model)

    strategy = get_strategy(policy, features, means, stds)

    res = evaluate(
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
        md_list, updates_list, run_res = strategy.run(sim, mode='test', unlimit=unlimit)
   
    return run_res

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
        #"../data/books.csv", "../data/trades.csv", num_rows=S+10*SS
        "../data/books.csv", "../data/trades.csv"
    )
        
    print("TOTAL DATA UPDATES: %d" % len(dataset))
    assert len(dataset) >= S + 2 * SS
    # dataset = preprocess_data("../data/books.csv", "../data/trades.csv")

    features, means, stds = prepare_features("../data/features.pickle")
  
    #for i in range(10):
    #i = 1
    #checkpoints = train(dataset[0:S+i*SS], features, means, stds)
    #SSS =int(SS/10)
    
    """
    for j in range(10):
        test(checkpoints[-1], dataset[S+i*SS+j*SSS:S+i*SS+(j+1)*SSS], features, means, stds)
    """

    #checkpoints = ["../models/PNL_1.147918_COIN_-88.250000_USDT_1555912.660625_PRICE_17404.350000_DELAY_0.1_LATENCY_0.01_DATENCY_0.01_HOLD_TIME_60.0_TRAJ_SIZE_100_S_200000_SS_20000_TRADE_SIZE_0.01_FEATURES_LEN_60_EPOCH_NO_00101_FROM_20230109T080000_TO_20230111T114605.pth"]

        #checkpoints = train(dataset[id_range[0]-S:id_range[0]], features, means, stds)
        #test(checkpoints[-1], dataset[id_range[0]:id_range[1]], features, means, stds)
    
    for time_range in [
        #FLUCTUATE
        ["2023-01-19 04:00:00", "2023-01-19 05:00:00"],
        ["2023-01-19 05:00:00", "2023-01-19 06:00:00"],
        ["2023-01-19 06:00:00", "2023-01-19 07:00:00"],
        ["2023-01-19 07:00:00", "2023-01-19 08:00:00"],
        ["2023-01-19 08:00:00", "2023-01-19 09:00:00"],
        ["2023-01-19 09:00:00", "2023-01-19 10:00:00"],
        ["2023-01-19 10:00:00", "2023-01-19 11:00:00"],
        ["2023-01-19 11:00:00", "2023-01-19 12:00:00"],
        ["2023-01-15 04:00:00", "2023-01-15 12:00:00"],
        ["2023-01-13 00:00:00", "2023-01-13 08:00:00"],
        #RISEUP
        #["2023-01-20 16:00:00", "2023-01-22 04:00:00"],
        #["2023-01-13 16:00:00", "2023-01-14 04:00:00"],
        #["2023-01-12 16:00:00", "2023-01-12 19:00:00"],
        #FALLDOWN
        #["2023-01-18 14:00:00", "2023-01-18 17:00:00"],
        #["2023-01-16 06:00:00", "2023-01-16 14:00:00"],
        #["2023-01-14 09:00:00", "2023-01-14 10:00:00"],
        ]:
        
        ta = (datetime.strptime(time_range[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc) - timedelta(hours=2)).timestamp()
        t0 = datetime.strptime(time_range[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
        t1 = datetime.strptime(time_range[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()

        DA = [md for md in dataset if md.receive_ts > ta and md.receive_ts < t0]
        DD = [md for md in dataset if md.receive_ts > t0 and md.receive_ts < t1]

        checkpoints = train(DA + DD[0:SS], features, means, stds)
        test(checkpoints[-1], DD, features, means, stds)
        print(time_range)
        breakpoint()
        pass


if __name__ == "__main__":
    main()
