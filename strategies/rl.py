from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import wandb
import random

from simulator.simulator import MdUpdate, Order, OwnTrade, Sim, update_best_positions
from datetime import datetime


def weighted_random_index(n, method='exponential', bias=2.0):
    """
    从列表中随机选择索引，偏向于选择尾部的元素
    
    参数:
        lst: 输入列表
        method: 使用的方法 ('exponential' 或 'linear')
        bias: 偏向程度，值越大越倾向于选择尾部元素
        
    返回:
        选中的索引
    """
    if n == 0:
        raise ValueError("列表不能为空")
    
    if method == 'exponential':
        # 使用指数函数生成权重
        weights = [np.exp(bias * i / n) for i in range(n)]
    elif method == 'linear':
        # 使用线性函数生成权重
        weights = [1 + (bias * i / n) for i in range(n)]
    else:
        raise ValueError("不支持的方法")
    
    # 归一化权重
    total = sum(weights)
    weights = [w/total for w in weights]
    
    # 根据权重随机选择索引
    return weights     

class A2CNetwork(nn.Module):
    '''
    input:
        states - tensor, (batch_size x num_features x num_lags)
    output:
        logits - tensor, logits of action probabilities for your actor policy, (batch_size x num_actions)
        V - tensor, critic estimation, (batch_size)
    '''

    def __init__(self, n_actions, device="cpu"):
        super().__init__()

        self.device = device
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(300 * 57, 256),
            nn.ReLU()
        )
        self.logits_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        self.V_net = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            torch.nn.init.orthogonal_(module.weight.data, np.sqrt(2))
        if module.bias is not None:
            module.bias.data.zero_()

    def forward(self, state_t):
        hidden_outputs = self.backbone(torch.as_tensor(np.array(state_t), dtype=torch.float).to(self.device))
        # print(hidden_outputs.shape)
        return self.logits_net(hidden_outputs), self.V_net(hidden_outputs).squeeze()


class Policy:
    def __init__(self, model):
        self.model = model

    def act(self, inputs):
        '''
        input:
            inputs - numpy array, (batch_size x num_features x num_lags)
        output: dict containing keys ['actions', 'logits', 'log_probs', 'values']:
            'actions' - selected actions, numpy, (batch_size)
            'logits' - actions logits, tensor, (batch_size x num_actions)
            'log_probs' - log probs of selected actions, tensor, (batch_size)
            'values' - critic estimations, tensor, (batch_size)
        '''
        logits, values = self.model(inputs)

        probs = F.softmax(logits, dim=-1)
        #         probs = softmax(logits, t=1.0)

        #         distr = torch.distributions.Categorical(probs)
        #         actions = distr.sample()
        actions = np.array(
            [np.random.choice(a=logits.shape[-1], p=prob, size=1)[0]
             for prob in probs.detach().cpu().numpy()]
        )

        eps = 1e-7
        log_probs = torch.log(probs + eps)[np.arange(probs.shape[0]), actions]
        entropy = -torch.sum(probs * torch.log(probs + eps))
        #         entropy = distr.entropy()

        return {
            "actions": actions,
            "logits": logits,
            "log_probs": log_probs,
            "values": values,
            "entropy": entropy,
            #             "inputs": inputs
        }


class ComputeValueTargets:
    def __init__(self, policy, gamma=0.999):
        self.policy = policy
        self.gamma = gamma

    def __call__(self, trajectory, latest_observation):
        '''
        This method should modify trajectory inplace by adding
        an item with key 'value_targets' to it

        input:
            trajectory - dict from runner
            latest_observation - last state, numpy, (num_envs x channels x width x height)
        '''
        trajectory['value_targets'] = [
            torch.empty(0)
            for _ in range(len(trajectory['values']))
        ]

        value_targets = [self.policy.act(latest_observation)["values"]]

        for step in range(len(trajectory['values']) - 2, -1, -1):
            value_targets.append(
                self.gamma * value_targets[-1] + trajectory['rewards'][step]
            )

        value_targets.reverse()
        for step in range(len(trajectory['values'])):
            trajectory['value_targets'][step] = value_targets[step]


class RLStrategy:
    """
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    """

    def __init__(self, policy: Policy, ess_df: pd.DataFrame, max_position: float,
                 means, stds,
                 delay: float, hold_time: Optional[float] = None, transforms=[],
                 trade_size=0.01, post_only=True, taker_fee=0.0004, maker_fee=-0.00004) -> None:
        """
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        """
        self.policy = policy
        self.features_df = ess_df
        self.means = np.broadcast_to(means, (300, means.shape[0])).T
        self.stds = np.broadcast_to(stds, (300, stds.shape[0])).T
        self.max_position = max_position

        self.delay = delay
        if hold_time is None:
            hold_time = min(delay * 5, pd.Timedelta(10, 's').total_seconds())
        self.hold_time = hold_time

        self.coin_position = 1.0
        self.usdt_position = 0.0
        self.tpnl = 1.0

        self.action_dict = {1: (0, 0), 2: (0, 4), 3: (0, 9),
                            4: (4, 0), 5: (4, 4), 6: (4, 9),
                            7: (9, 0), 8: (9, 4), 9: (9, 9)}

        """
        self.action_dict = {0: (-1,-1), 1: (-1,1), 2: (1,-1), 3: (1,1)}
        """

        self.actions_history = []
        self.ongoing_orders = {}

        self.trajectory = {}
        for key in ['actions', 'logits', 'log_probs', 'values', 'entropy', 'observations', 'rewards']:
            self.trajectory[key] = []
        self.transforms = transforms

        self.trade_size = trade_size

        self.post_only = post_only
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee

        #FEATURE

    def reset(self):
        self.features_df['coin_position'] = 1.0
        self.features_df['tpnl'] = 0.0

        self.coin_position = 1.0
        self.usdt_position = 0.0
        self.tpnl = 1.0


        self.actions_history = []
        self.ongoing_orders = {}

        self.trajectory = {}
        for key in ['actions', 'logits', 'log_probs', 'values', 'entropy', 'observations', 'rewards']:
            self.trajectory[key] = []

    def add_ass_features(self, receive_ts, price) -> None:

        self.tpnl = self.usdt_position / price + self.coin_position

        self.features_df.loc[
            self.features_df['receive_ts'] == receive_ts,
            ['coin_position', 'tpnl']
        ] = (self.coin_position, self.tpnl)

    def get_features(self, receive_ts):
        # TODO: WHY ONLY 10 SECONDS AS A FEATURE WINDOW? TOO SMALL?
        features = self.features_df[
            (self.features_df['receive_ts'] <= pd.to_datetime(receive_ts, unit='s')) &
            (self.features_df['receive_ts'] >= (pd.to_datetime(receive_ts, unit='s') - timedelta(seconds=10)))
            ].drop(columns='receive_ts').values.T

        if features.shape[1] < 300:
            try:
                features = np.pad(features, ((0, 0), (300 - features.shape[1], 0)), mode='edge')
            except ValueError:
                features = self.means
        elif features.shape[1] > 300:
            features = features[:, -300:]

        return np.divide(features - self.means, self.stds, out=np.zeros_like(features), where=self.stds != 0)

    def place_order(self, sim, action_id, receive_ts, asks, bids):
        if action_id == 0: 
            return 
        
        ask_level, bid_level = self.action_dict[action_id]
        ask_order = sim.place_order(receive_ts, self.trade_size, 'ASK', asks[ask_level])
        bid_order = sim.place_order(receive_ts, self.trade_size, 'BID', bids[bid_level])

        self.ongoing_orders[bid_order.order_id] = (bid_order, 'LIMIT')
        self.ongoing_orders[ask_order.order_id] = (ask_order, 'LIMIT')

        self.actions_history.append((receive_ts, self.coin_position, action_id))

    def run(self, sim: Sim, mode: str, traj_size=1000) -> \
            Tuple[List[OwnTrade], List[MdUpdate], List[Union[OwnTrade, MdUpdate]], List[Order]]:
        """
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates
                received by strategy(market data and information about executed trades)
                all_orders(List[Order]): list of all placed orders
        """

        md_list: List[MdUpdate] = []
        trades_list: List[OwnTrade] = []
        updates_list = []
        # current best positions
        best_bid = -np.inf
        best_ask = np.inf
        bids = [-np.inf] * 10
        asks = [np.inf] * 10

        # last order timestamp
        prev_time = -np.inf
        # orders that have not been executed/canceled yet
        prev_tpnl = None
        if mode != 'train':
            traj_size = 1e8
        
        est_price = np.inf
      
        # breakpoint()  
        while len(self.trajectory['rewards']) < traj_size:
            # get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            # save updates
            updates_list += updates
            for update in updates:
                # update best position
                if isinstance(update, MdUpdate):
                    if update.orderbook is not None:
                        best_bid, best_ask, asks, bids = update_best_positions(best_bid, best_ask, update, levels=True)
                    md_list.append(update)
                    est_price = (best_bid + best_ask) / 2.0

                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    # delete executed trades from the dict
                    if update.order_id in self.ongoing_orders.keys():
                        _, order_type = self.ongoing_orders[update.order_id]
                        self.ongoing_orders.pop(update.order_id)

                    if self.post_only:
                        if order_type == 'LIMIT' and update.execute == 'TRADE':
                            if update.side == 'BID':
                                self.coin_position += update.size
                                self.usdt_position -= (1 + self.maker_fee) * update.price * update.size
                            else:
                                self.coin_position -= update.size
                                self.usdt_position += (1 - self.maker_fee) * update.price * update.size
                        elif order_type == 'MARKET':
                            if update.side == 'BID':
                                self.coin_position += update.size
                                self.usdt_position -= (1 + self.taker_fee) * update.price * update.size
                            else:
                                self.coin_position -= update.size
                                self.usdt_position += (1 - self.taker_fee) * update.price * update.size
                    else:
                        if update.execute == 'TRADE':
                            fee = self.maker_fee
                        else:
                            fee = self.taker_fee
                        if update.side == 'BID':
                            self.coin_position += update.size
                            self.usdt_position -= (1 + fee) * update.price * update.size
                        else:
                            self.coin_position -= update.size
                            self.usdt_position += (1 - fee) * update.price * update.size

                    est_price = update.price

                else:
                    assert False, 'invalid type of update!'
            #             self.results['receive_ts'].append(receive_ts)
            #             self.results['coin_pos'].append(self.coin_position)
            #             self.results['mid_price'].append((best_ask + best_bid) / 2)

            self.add_ass_features(receive_ts, est_price)
            
            if receive_ts - prev_time >= self.delay:
                if mode == 'train':
                    if prev_tpnl is None:
                        prev_tpnl = 1.0
                        prev_coin_position = 1.0
                    else:
                        if (self.usdt_position / est_price + self.coin_position - prev_tpnl) > 1e-7:
                            #reward = 0
                            #reward = (self.usdt_position / est_price + self.coin_position - prev_tpnl) * 100000 # $1 PRICE CHANGE MEANS 0.1 REWARD
                            reward = (self.usdt_position / est_price + self.coin_position - prev_tpnl) * 10000 # $1 PRICE CHANGE MEANS 0.1 REWARD
                        else: 
                            reward = -0.1

                        """
                        if self.coin_position <= 0.2: #TODO: 0.2 MAGIC NUMBER?
                            reward =  - 0.5 * self.coin_position
                            )
                            print("reward: ", reward)
                        else:
                            reward = -0.1 #TODO: 0.1 MAGIC NUMBER?
                            print("reward: ", reward)
                        """ 

                        prev_tpnl = self.usdt_position / est_price + self.coin_position
                        prev_coin_position = self.coin_position

                        self.trajectory['observations'].append(features)
                        self.trajectory['rewards'].append(reward)

                        for key, val in act.items():
                            self.trajectory[key].append(val)

                # place order
                features = self.get_features(receive_ts)
                act = self.policy.act([features])
                self.place_order(sim, act['actions'][0], receive_ts, asks, bids)

                prev_time = receive_ts


            to_cancel = []
            for ID, (order, order_type) in self.ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order(receive_ts, ID)
                    to_cancel.append(ID)
            for ID in to_cancel:
                self.ongoing_orders.pop(ID)

        if mode == 'train':
            for transform in self.transforms:
                transform(self.trajectory, [features])
        
        if len(md_list) > 0: 
            print("[%s] MODE RUN IN RANGE [%s, %s]" % (mode.upper(), datetime.fromtimestamp(md_list[0].receive_ts), datetime.fromtimestamp(md_list[-1].receive_ts)))
        else:
            print("[%s] MODE BUT EMPTY RUN AT %s" % (mode.upper(), ))

        return trades_list, md_list, updates_list, self.actions_history, self.trajectory


class A2C:
    def __init__(self, policy, optimizer, value_loss_coef=0.1, entropy_coef=0.1, max_grad_norm=0.5, device="cpu"):
        self.policy = policy
        self.optimizer = optimizer
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.last_trajectories = None

    def loss(self, trajectory):
        # compute all losses
        # do not forget to use weights for critic loss and entropy loss
        trajectory['log_probs'] = torch.stack(trajectory['log_probs']).squeeze().to(self.device)
        trajectory['value_targets'] = torch.stack(trajectory['value_targets']).to(self.device)
        trajectory['values'] = torch.stack(trajectory['values']).to(self.device)
        trajectory['entropy'] = torch.stack(trajectory['entropy']).to(self.device)

        policy_loss = (trajectory['log_probs'] * (trajectory['value_targets'] - trajectory['values']).detach()).mean()
        critic_loss = ((trajectory['value_targets'].detach() - trajectory['values']) ** 2).mean()
        entropy_loss = trajectory["entropy"].mean()

        total_loss = self.value_loss_coef * critic_loss - policy_loss - self.entropy_coef * entropy_loss

        # log all losses
        """
        wandb.log({
            'total loss': total_loss.detach().item(),
            'policy loss': policy_loss.detach().item(),
            'critic loss': critic_loss.detach().item(),
            'entropy loss': entropy_loss.detach().item(),
            'train reward': np.mean(trajectory['rewards'])
        })
        print({
            'total loss': total_loss.detach().item(),
            'policy loss': policy_loss.detach().item(),
            'critic loss': critic_loss.detach().item(),
            'entropy loss': entropy_loss.detach().item(),
            'train reward': np.mean(trajectory['rewards'])
        })
        """

        return total_loss

    def train(self, strategy, md, latency, datency, traj_size=1000):
        # collect trajectory using runner
        # compute loss and perform one step of gradient optimization
        # do not forget to clip gradients
        strategy.reset()
        random_slice = random.choices(range(len(md)-traj_size), weights=weighted_random_index(len(md)-traj_size, method='exponential', bias=np.log(2.5)))[0] # RANDOM CHOICE IN [0:md_size - traj_size]
        print("[TRAIN] RANDOM SLICE: %d" % random_slice)
        sim = Sim(md[random_slice:], latency, datency)
        trades_list, md_list, updates_list, actions_history, trajectory = strategy.run(sim, mode='train', traj_size=traj_size)

        self.last_trajectories = trajectory

        loss = self.loss(trajectory)
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = clip_grad_norm_(strategy.policy.model.parameters(), self.max_grad_norm)
        self.optimizer.step()


def evaluate(strategy, md, latency, datency):
    strategy.reset()
    sim = Sim(md, latency, datency)
    with torch.no_grad():
        trades_list, md_list, updates_list, actions_history, trajectory = strategy.run(sim, mode='test')
   
    return np.sum(trajectory['rewards']), strategy.tpnl, trajectory

