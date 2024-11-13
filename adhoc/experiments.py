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

# +
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulator.simulator import Sim
from strategies.baselines import BestPosStrategy, StoikovStrategy
from utils.get_info import get_pnl, get_volumes
from utils.load_data import load_md_from_file


NROWS = 10_000_000

# +
from utils.load_data import load_md_from_csv
md = load_md_from_csv("../data/books.csv", "../data/trades.csv", nrows=10_000_000)

latency = pd.Timedelta(10, 'ms').total_seconds()
md_latency = pd.Timedelta(10, 'ms').total_seconds()


breakpoint()
ks = [10, 12, 15]
# ks = [1, 10, 20]

for k in ks:
    sim = Sim(md[:230_000], latency, md_latency)

    delay = pd.Timedelta(0.1, 's').total_seconds()
    hold_time = pd.Timedelta(10, 's').total_seconds()
    terminal_date = pd.to_datetime('2023-01-22')

    strategy = StoikovStrategy(delay=delay, hold_time=hold_time, trade_size=0.01, risk_aversion=0.5, k=k,
                               post_only=True)

    trades_list_st, md_list, updates_list, all_orders = strategy.run(sim)

    df = get_pnl(updates_list, post_only=True, maker_fee=-0.00004, taker_fee=0.0004)
    df['receive_ts'] = pd.to_datetime(df['receive_ts'])
    df.set_index('receive_ts', inplace=True)

    dfs_k.append(df)
# -

dfs_k

# +
ks = ['k1', 'k2', 'k3'] # 对应 dfs_k 中的每个 DataFrame

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

for i, df_ in enumerate(dfs_k):
    df__ = df_[['total', 'BTC']].resample('1s').last().reset_index()
    ax1.plot(df__['receive_ts'], df__['total'], label=f'k: {ks[i]}_USD')
    ax2.plot(df__['receive_ts'], df__['BTC'], label=f'k: {ks[i]}_ETH')

# 设置图表标题和坐标轴标签
fig.suptitle("PnL Depending on k")
ax1.set_xlabel("Time")
ax1.set_ylabel("USDT")
ax2.set_xlabel("Time")
ax2.set_ylabel("ETH")

# 添加图例
ax1.legend()
ax2.legend()

# 调整布局, 以确保标题和坐标轴标签不重叠
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.5)

plt.show()

# -


