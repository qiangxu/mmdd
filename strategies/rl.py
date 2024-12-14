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
from collections import Counter

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

    def __init__(self, n_actions, num_features, features_len, device="cpu"):
        super().__init__()

        self.device = device
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features_len * num_features, 256),
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

        print("[TRAIN] REWARDS: ", np.sum(trajectory['rewards']))

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

    def __init__(self, policy: Policy, ess_df: pd.DataFrame, 
                 means, stds, features_len,
                 delay: float, hold_time: Optional[float] = None, transforms=[],
                 trade_size=0.01, post_only=True, taker_fee=0.0004, maker_fee=-0.00004) -> None:
        """
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        """

        self.policy = policy
        self.features_df = ess_df
        self.delay = delay

        if hold_time is None:
            hold_time = min(delay * 5, pd.Timedelta(10, 's').total_seconds())
        self.hold_time = hold_time
        self.features_size = features_len
        self.means = np.broadcast_to(means, (self.features_size, means.shape[0])).T
        self.stds = np.broadcast_to(stds, (self.features_size, stds.shape[0])).T

   
        """
        self.action_dict = {1: (0, 0), 2: (0, 4), 3: (0, 9),
                            4: (4, 0), 5: (4, 4), 6: (4, 9),
                            7: (9, 0), 8: (9, 4), 9: (9, 9)}
        """

        #self.action_dict = {1: (-1,8), 2: (8,-1)}
        #self.action_dict = {0: (-1,-1), 1: (-1,8), 2: (8,-1)}
        self.action_dict = [0, 3, 5, 7]
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
        #self.features_df['coin_position'] = 1.0

        self.actions_history = []
        self.ongoing_orders = {}

        self.trajectory = {}
        for key in ['actions', 'logits', 'log_probs', 'values', 'entropy', 'observations', 'rewards']:
            self.trajectory[key] = []

    """
    def add_ass_features(self, receive_ts, coin_position) -> None:

        self.features_df.loc[
            self.features_df['receive_ts'] == receive_ts,
            ['coin_position']
        ] = (coin_position)
    """

    def get_features(self, receive_ts):
        # TODO: WHY ONLY 10 SECONDS AS A FEATURE WINDOW? TOO SMALL?
        features = self.features_df[
            (self.features_df['receive_ts'] <= pd.to_datetime(receive_ts, unit='s')) &
            (self.features_df['receive_ts'] >= (pd.to_datetime(receive_ts, unit='s') - timedelta(seconds=self.features_size)))
            ]
        
        features_ts = features.receive_ts.values.astype(np.int64)/10**9
        #print("FEATURES TIMELEN: ", receive_ts - features_ts[0])
        features = features.drop(columns='receive_ts').values.T

        if features.shape[1] < self.features_size:
            try:
                features = np.pad(features, ((0, 0), (self.features_size - features.shape[1], 0)), mode='edge')
            except ValueError:
                features = self.means
        elif features.shape[1] > self.features_size:
            features = features[:, -self.features_size:]
        
        return np.divide(features - self.means, self.stds, out=np.zeros_like(features), where=self.stds != 0)

    def get_reward_f1(self, coin_position, usdt_position, coin_price, prev_coin_position, prev_usdt_position, prev_coin_price): 
        p = (coin_position - prev_coin_position) + (usdt_position - prev_usdt_position) / coin_price
        if coin_position == prev_coin_position: 
            reward = 0.05
        elif p > (1e-4 * self.trade_size): 
            reward = p * 10_000 / self.trade_size
        elif p < -(1e-4 * self.trade_size):
            reward = p * 10_000 / self.trade_size 
        else:
            # breakpoint()
            # DISENCOURAGE USELESS TRADING
            reward = -0.1

        return reward

    def get_reward_f2(self, coin_position, usdt_position, coin_price, prev_coin_position, prev_usdt_position, prev_coin_price): 
        p = coin_position * coin_price - prev_coin_position * prev_coin_price + usdt_position - prev_usdt_position
        if coin_position == prev_coin_position: 
            reward = 0.05
        elif p > 1e4 * 1e-3:
            reward = p 
        elif p < -(1e4 * 1e-3):
            reward = p
        else:
            reward = -0.1

        return reward

    def get_reward_f3(self, coin_position, usdt_position, coin_price, prev_coin_position, prev_usdt_position, prev_coin_price, trades_list, receive_ts): 
        p = (coin_position - prev_coin_position) + (usdt_position - prev_usdt_position) / coin_price
        trades = [t for t in trades_list if t.receive_ts < receive_ts and t.receive_ts > receive_ts - self.delay]

        if coin_position != prev_coin_position:
            # breakpoint()
            pass
        #    reward = len(trades) * 20
        if len(trades) > 0:
            # breakpoint()
            reward = len(trades) * self.trade_size * 2 * 1000
        elif p > (1e-4 * self.trade_size): 
            reward = 5
        elif p < -(1e-4 * self.trade_size):
            reward = -5
        else: 
            reward = -1

        return reward
    def get_reward_f4(self, coin_position, usdt_position, coin_price, prev_coin_position, prev_usdt_position, prev_coin_price, trades_list, receive_ts, ): 
        p = (coin_position - prev_coin_position) + (usdt_position - prev_usdt_position) / coin_price
        trades = [t for t in trades_list if t.receive_ts < receive_ts and t.receive_ts > receive_ts - self.delay]
        
        trades_ask = [t for t in trades if t.side == "ASK" and t.receive_ts < receive_ts and t.receive_ts > receive_ts - self.hold_time]
        trades_bid = [t for t in trades if t.side == "BID" and t.receive_ts < receive_ts and t.receive_ts > receive_ts - self.hold_time]

        orders = [o[0] for o in self.ongoing_orders.values() if o[0].place_ts < receive_ts and o[0].place_ts > receive_ts - self.hold_time]
        orders_ask = [o for o in orders if o.side == "ASK"]
        orders_bid = [o for o in orders if o.side == "BID"]
       
        actions = [a for a in self.actions_history if a[0] < receive_ts and a[0] > receive_ts - self.hold_time]
        actions_ask = [a for a in actions if self.action_dict[a[2]//4 % 4] != 0]
        actions_bid = [a for a in actions if self.action_dict[a[2] % 4] != 0]
        
        #print("ASK: %d %d %d" % (len(orders_ask), len(trades_ask), len(actions_ask)))
        #print("BID: %d %d %d" % (len(orders_bid), len(trades_bid), len(actions_bid)))

        if len(trades) > 0:
            # breakpoint()
            return len(trades) * self.trade_size * 2 * 1000
        else: 
            reward = 0
            if (len(actions_ask) > 0) and (len(actions_ask))/(len(actions_ask) + 10) * len(orders_ask) / len(actions_ask) > 0.45:
                reward += -500 / self.hold_time * self.delay

            if (len(actions_bid) > 0) and (len(actions_bid))/(len(actions_bid) + 10) * len(orders_bid) / len(actions_bid) > 0.45:
                reward += -500 / self.hold_time * self.delay
            return reward 

    def place_order(self, sim, action_id, receive_ts, coin_position, asks, bids):
        ask_level = self.action_dict[(action_id // 4) % 4]
        bid_level = self.action_dict[action_id % 4] 

        #print("ACTION %d: (%d, %d)" % (action_id, ask_level, bid_level))
        #breakpoint() 
        if bid_level != 0:
            p = bids[0] * (1 - 0.0001 * bid_level - self.maker_fee)
            bid_order = sim.place_order(receive_ts, self.trade_size, 'BID', p, cost=(asks[0] + bids[0])/2)
            self.ongoing_orders[bid_order.order_id] = (bid_order, 'LIMIT')

        if ask_level != 0:
            p = asks[0] * (1 + 0.0001 * ask_level + self.maker_fee)
            ask_order = sim.place_order(receive_ts, self.trade_size, 'ASK', p, cost=(asks[0] + bids[0])/2)
            self.ongoing_orders[ask_order.order_id] = (ask_order, 'LIMIT')

        self.actions_history.append((receive_ts, coin_position, action_id))

    def run(self, sim: Sim, mode: str, traj_size=1000, unlimit=True) -> \
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
        if mode != 'train':
            traj_size = 1e8

        prev_coin_position = None
        prev_usdt_position = None
        prev_coin_price = 0.0
        coin_position = 1.0
        usdt_position = 0.0
        coin_price = 0.0
        max_position = 1.0
      
        #while (len(self.trajectory['rewards']) < traj_size) and (coin_position > 0):
        while (len(self.trajectory['rewards']) < traj_size):
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
                        coin_price = (best_bid + best_ask) / 2.0

                    md_list.append(update)

                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    # delete executed trades from the dict
                    if update.order_id in self.ongoing_orders.keys():
                        _, order_type = self.ongoing_orders[update.order_id]
                        self.ongoing_orders.pop(update.order_id)

                    if self.post_only:
                        if order_type == 'LIMIT' and update.execute == 'BOOK':
                            if update.side == 'BID':
                                coin_position += update.size
                                usdt_position -= (1 + self.maker_fee) * update.price * update.size
                            else:
                                coin_position -= update.size
                                usdt_position += (1 - self.maker_fee) * update.price * update.size
                        elif order_type == 'LIMIT' and update.execute == 'TRADE':
                        #elif order_type == 'MARKET':
                            # breakpoint()
                            if update.side == 'BID':
                                coin_position += update.size
                                usdt_position -= (1 + self.taker_fee) * update.price * update.size
                            else:
                                coin_position -= update.size
                                usdt_position += (1 - self.taker_fee) * update.price * update.size
                        else: 
                            pass
                            # breakpoint()
                    else:
                        if update.execute == 'TRADE':
                            fee = self.maker_fee
                        else:
                            fee = self.taker_fee
                        if update.side == 'BID':
                            coin_position += update.size
                            usdt_position -= (1 + fee) * update.price * update.size
                        else:
                            coin_position -= update.size
                            usdt_position += (1 - fee) * update.price * update.size

                    coin_price = update.price

                else:
                    assert False, 'invalid type of update!'
            #             self.results['receive_ts'].append(receive_ts)
            #             self.results['coin_pos'].append(coin_position)
            #             self.results['mid_price'].append((best_ask + best_bid) / 2)


            # breakpoint()
            #self.add_ass_features(receive_ts, coin_position)
            
            if receive_ts - prev_time >= self.delay:
                if mode == 'train':

                    if prev_coin_position is None:
                        prev_coin_position = 1.0
                        prev_usdt_position = 0.0
                        prev_coin_price = updates_list[0].get_price()
                    else:
                        #reward = self.get_reward_f2(coin_position, usdt_position, coin_price, prev_coin_position, prev_usdt_position, prev_coin_price) 
                        #reward = self.get_reward_f3(coin_position, usdt_position, coin_price, prev_coin_position, prev_usdt_position, prev_coin_price, trades_list, receive_ts) 
                        reward = self.get_reward_f4(coin_position, usdt_position, coin_price, prev_coin_position, prev_usdt_position, prev_coin_price, trades_list, receive_ts) 

                        max_position = max(coin_position, max_position)

                        prev_coin_position = coin_position
                        prev_usdt_position = usdt_position
                        prev_coin_price = coin_price

                        self.trajectory['observations'].append(features)
                        self.trajectory['rewards'].append(reward)

                        for key, val in act.items():
                            self.trajectory[key].append(val)

                # place order
                features = self.get_features(receive_ts)

                act = self.policy.act([features])

                if unlimit:
                    self.place_order(sim, act['actions'][0], receive_ts, coin_position, asks, bids)
                else: 
                    action_id = act['actions'][0]

                    if action_id == 1:
                       if usdt_position >= (1 + self.maker_fee) * coin_price * self.trade_size: 
                            self.place_order(sim, act['actions'][0], receive_ts, coin_position, asks, bids)
                    elif action_id == 2:
                        if coin_position > self.trade_size:
                            self.place_order(sim, act['actions'][0], receive_ts, coin_position, asks, bids)
                prev_time = receive_ts

            to_cancel = []
            for ID, (order, order_type) in self.ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order(receive_ts, ID)
                    to_cancel.append(ID)

            for ID in to_cancel:
                #print("STACK: %d, CANCEL: %d" % (len(self.ongoing_orders), len(to_cancel)))
                self.ongoing_orders.pop(ID)
            

        if mode == 'train':
            for transform in self.transforms:
                transform(self.trajectory, [features])

        #pd.DataFrame(self.trajectory['rewards']).plot.hist(bins=100).get_figure().savefig("rewards.pdf")
         
        print(self.trajectory['rewards'])
        
        #ASK
        mid_price = (updates_list[0].get_price() + updates_list[-1].get_price()) / 2

         
        ask_orders = [t for t in trades_list if t.side == "ASK"]
        if len(ask_orders):
            gain_per_ask = sum([(t.price - t.cost) for t in ask_orders]) / len(ask_orders) / mid_price
        else:
            gain_per_ask = 0

        bid_orders = [t for t in trades_list if t.side == "BID"]
        if len(bid_orders): 
            gain_per_bid = sum([(t.cost - t.price) for t in bid_orders]) / len(bid_orders) / mid_price
        else:
            gain_per_bid = 0

        gain_ratio = sum([(t.price - t.cost) for t in ask_orders]) / mid_price + sum([(t.cost - t.price) for t in bid_orders]) / mid_price
        perf = (coin_position, usdt_position, gain_ratio, (len(ask_orders), gain_per_ask), (len(bid_orders), gain_per_bid), len(trades_list) / len(self.actions_history))

        if len(trades_list) > 0: 
            print("[%s] MODE (%d, %d) TRADES IN [%s, %s] IN RANGE [%s, %s]" % (mode.upper(), len([t for t in trades_list if t.side == "ASK"]), len([t for t in trades_list if t.side == "BID"]), datetime.utcfromtimestamp(trades_list[0].receive_ts), datetime.utcfromtimestamp(trades_list[-1].receive_ts), datetime.utcfromtimestamp(md_list[0].receive_ts), datetime.utcfromtimestamp(md_list[-1].receive_ts)))
            print("[%s] MODE PERFORMANCE %s" % (mode.upper(), perf))
            print("[%s] MODE PRICE CHANGE FROM %f TO %f" % (mode.upper(), updates_list[0].get_price(), updates_list[-1].get_price()))
       
        print("[%s] TRADES: " % mode.upper(), len(trades_list))
        print("[%s] ACTIONS: " % mode.upper(), dict(sorted(dict(Counter([a[2] for a in self.actions_history])).items(), reverse=True)))

        return trades_list, md_list, updates_list, perf, self.actions_history, self.trajectory


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
        print("[TRAIN] RANDOM SLICE %d" % random_slice)
        sim = Sim(md[random_slice:], latency, datency)
        trades_list, md_list, updates_list, perf, actions_history, trajectory = strategy.run(sim, mode='train', traj_size=traj_size)

        self.last_trajectories = trajectory

        loss = self.loss(trajectory)
        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = clip_grad_norm_(strategy.policy.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return perf 



