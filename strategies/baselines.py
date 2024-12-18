from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulator.simulator import MdUpdate, Order, OwnTrade, Sim, update_best_positions


class BestPosStrategy:
    """
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    """

    def __init__(self, delay: float, hold_time: Optional[float]=None,
                 max_position: Optional[float]=None, trade_size: float=0.001) -> None:
        """
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        """
        self.delay = delay
        if hold_time is None:
            hold_time = min(delay * 5, pd.Timedelta(10, 's').total_seconds())
        self.hold_time = hold_time
        self.max_position = max_position
        self.trade_size = trade_size
        self.coin_position = 0

    def run(self, sim: Sim) -> \
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

        # market data list
        md_list: List[MdUpdate] = []
        # executed trades list
        trades_list: List[OwnTrade] = []
        # all updates list
        updates_list = []
        # current best positions
        best_bid = -np.inf
        best_ask = np.inf

        # last order timestamp
        prev_time = -np.inf
        # orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            # get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            # save updates
            updates_list += updates
            for update in updates:
                # update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    # delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                    
                    if update.side == 'BID':
                        self.coin_position += update.size
                    else:
                        self.coin_position -= update.size
                else:
                    assert False, 'invalid type of update!'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                # place order
                if self.max_position is not None and abs(self.coin_position) > self.max_position:
                    if self.coin_position > 0:
                        ask_order = sim.place_order(receive_ts, self.trade_size, 'ASK', best_ask)
                        ongoing_orders[ask_order.order_id] = ask_order
                        all_orders += [ask_order]
                    else:
                        bid_order = sim.place_order(receive_ts, self.trade_size, 'BID', best_bid)
                        ongoing_orders[bid_order.order_id] = bid_order
                        all_orders += [bid_order]
                else:
                    bid_order = sim.place_order(receive_ts, self.trade_size, 'BID', best_bid)
                    ask_order = sim.place_order(receive_ts, self.trade_size, 'ASK', best_ask)
                    ongoing_orders[bid_order.order_id] = bid_order
                    ongoing_orders[ask_order.order_id] = ask_order

                    all_orders += [bid_order, ask_order]

            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order(receive_ts, ID)
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)

        return trades_list, md_list, updates_list, all_orders

    
class StoikovStrategy:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''
    def __init__(self, delay: float, hold_time:Optional[float] = None, trade_size:Optional[float] = 0.01, risk_aversion:Optional[float] = 0.5, k:Optional[float] = 1.5, post_only = False) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').total_seconds())
        self.hold_time = hold_time
        self.order_size = trade_size
        self.last_mid_prices = []
        self.post_only = post_only
        self.asset_position = 0
        self.gamma = risk_aversion
        self.k = k
        self.current_bid_order_id = None
        self.current_ask_order_id = None
        self.previous_bid_order_id = None
        self.previous_ask_order_id = None
        self.trades_dict = {'place_ts' :[], 'exchange_ts': [], 'receive_ts': [], 'trade_id': [],'order_id': [],'side': [], 'size': [], 'price': [],'execute':[], 'mid_price':[]}
        
    def run(self, sim: Sim ) ->\
        Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''

        #market data list
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        best_bid = -np.inf
        best_ask = np.inf

        #last order timestamp
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            #get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            #save updates
            updates_list += updates
            for update in updates:
                #update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    mid_price = (best_bid + best_ask)/2
                    if len(self.last_mid_prices) < 500:
                        self.last_mid_prices.append(mid_price)
                    else:
                        self.last_mid_prices.append(mid_price)
                        self.last_mid_prices.pop(0)
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    self.trades_dict['place_ts'].append(update.place_ts)
                    self.trades_dict['exchange_ts'].append(update.exchange_ts)
                    self.trades_dict['receive_ts'].append(update.receive_ts)
                    self.trades_dict['trade_id'].append(update.trade_id)
                    self.trades_dict['order_id'].append(update.order_id)
                    self.trades_dict['side'].append(update.side)
                    self.trades_dict['size'].append(update.size)
                    self.trades_dict['price'].append(update.price)
                    self.trades_dict['execute'].append(update.execute)
                    self.trades_dict['mid_price'].append(mid_price)
                    trades_list.append(update)
                    if self.post_only and update.execute == 'TRADE':
                        if update.side == "ASK":
                            self.asset_position -= update.size
                        elif update.side == "BID":
                            self.asset_position += update.size
                    elif not self.post_only:
                        if update.side == "ASK":
                            self.asset_position -= update.size
                        elif update.side == "BID":
                            self.asset_position += update.size                        
                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else: 
                    assert False, 'invalid type of update!'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                #place order
                '''
                reservation_price = s - q * gamma * (sigma**2) * (T - t)
                delta_bid and delta_ask are equivalently distant from the reservation_orice
                delta_bid + delta_ask = gamma * (sigma**2) * (T-t) + 2/gamma * ln(1 + gamma/k)
                k = K*alpha
                
                s      : current mid_price
                q      : current position in asset
                sigma  : parameter in the Geometric Brownian Motion equation [dS_t = sigma dw_t]
                T      : termination time
                t      : current time
                gamma  : risk-aversion parameter of the optimizing agents (across the economy)
                K      : higher K means that market order volumes have higher impact on best price changes
                alpha  : higher alpha means higher probability fo large market orders
                
                '''
                if len(self.last_mid_prices)==500:
                    sigma = np.std(self.last_mid_prices)## per update --> need to scale it to the "per second" terminology
                else:
                    sigma = 1
                sigma = sigma*np.sqrt(1/0.032)
                delta_t = 0.032 ## there is approximately 0.032 seconds in between the orderbook uprates (nanoseconds / 1e9 = seconds)
                q = self.asset_position
                ## mid_price = (best_bid + best_ask)/2 ## was defined previously
                reservation_price = mid_price - q*self.gamma*(sigma**2)*delta_t
                deltas_ = self.gamma * (sigma**2) * delta_t + 2/self.gamma * np.log(1 + self.gamma/self.k)
                bid_price = np.round(reservation_price - deltas_/2, 1)
                ask_price = np.round(reservation_price + deltas_/2, 1)
                
                bid_order = sim.place_order( receive_ts, self.order_size, 'BID', bid_price)
                ask_order = sim.place_order( receive_ts, self.order_size, 'ASK', ask_price)
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order
                
                self.previous_bid_order_id = self.current_bid_order_id
                self.previous_ask_order_id = self.current_ask_order_id
                
                self.current_bid_order_id = bid_order.order_id
                self.current_ask_order_id = ask_order.order_id

                all_orders += [bid_order, ask_order]
            
            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            if self.previous_bid_order_id in ongoing_orders.keys():
                sim.cancel_order( receive_ts, self.previous_bid_order_id )
                to_cancel.append(self.previous_bid_order_id)
            if self.previous_ask_order_id in ongoing_orders.keys():
                sim.cancel_order( receive_ts, self.previous_ask_order_id )
                to_cancel.append(self.previous_ask_order_id)
            for ID in to_cancel:
                try:
                    ongoing_orders.pop(ID)
                except:
                    continue
            
                
        return trades_list, md_list, updates_list, all_orders


class StoikovStrategy_old:
    """
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    """

    def __init__(
            self,
            trade_size: float,
            delay: float,
            terminal_date,
            k=1.5,
            hold_time: Optional[float] = None,
            risk_preference: Optional[float] = 0.01,
            initial_vol: Optional[float] = 2.19346e-08,
            vol_freq: Optional[float] = 1,
            lamb: Optional[float] = 0.95,

    ) -> None:
        """
            Args:
                delay: delay between orders in nanoseconds
                hold_time: holding time in nanoseconds
                vol_freq:: volatility frequency in seconds
                risk_preference: >0 for risk aversion, ~0 for risk-neutrality, or <0 for risk-seeking
                initial_vol: initial volatility estimated on history
                vol_freq: volatility frequency in seconds
                lamb: lambda in EWMA for updating volatility
        """

        self.trade_size = trade_size
        self.delay = delay
        if hold_time is None:
            hold_time = min(delay * 5, pd.Timedelta(10, 's').total_seconds())
        self.hold_time = hold_time

        # market data list
        self.md_list = []
        # executed trades list
        self.trades_list = []
        # all updates list
        self.updates_list = []

        self.risk_preference = risk_preference
        self.current_time = None
        self.coin_position = 0
        self.prev_midprice = None
        self.current_midprice = None
        self.terminal_date = terminal_date

        self.volatility = initial_vol
        self.lamb = lamb
        self.vol_freq = pd.Timedelta(vol_freq, 's').total_seconds()
        self.reservation_price = None
        self.spread = None
        self.k = k

        self.quotes_history = []

    def update_volatility(self) -> None:
        ret = (self.current_midprice - self.prev_midprice) / self.prev_midprice
        self.volatility = self.lamb * self.volatility + (1 - self.lamb) * ret ** 2

    def update_reservation_price(self) -> None:
        breakpoint()
        #TODO: make sure pd.to_datetime's unit
        time_to_terminal = (self.terminal_date - pd.to_datetime(self.current_time)).total_seconds() / self.vol_freq

        self.reservation_price = (
                self.current_midprice - (self.coin_position / self.trade_size)
                * self.risk_preference * self.volatility * time_to_terminal
        )

    def update_spread(self) -> None:
        breakpoint()
        #TODO: make sure pd.to_datetime's unit

        time_to_terminal = (self.terminal_date - pd.to_datetime(self.current_time)).total_seconds() / self.vol_freq

        self.spread = (
                self.risk_preference * self.volatility * time_to_terminal +
                (2 / self.risk_preference) * np.log(1 + self.risk_preference / self.k)
        )

    def run(self, sim: Sim) -> \
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
                quotes_history: list of tuples(time, coin_pos, bid, mid, reservation, ask)
        """

        # current best positions
        best_bid = -np.inf
        best_ask = np.inf

        # last order timestamp
        prev_time = -np.inf
        vol_prev_time = -np.inf
        # orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        while True:
            # get update from simulator
            self.current_time, updates = sim.tick()
            if updates is None:
                break
            # save updates
            self.updates_list += updates
            for update in updates:
                # update best position
                if isinstance(update, MdUpdate):
                    if update.orderbook is not None:
                        best_bid, best_ask = update_best_positions(update)
                    self.current_midprice = (best_bid + best_ask) / 2

                elif isinstance(update, OwnTrade):
                    self.trades_list.append(update)
                    # delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)

                    if update.side == 'BID':
                        self.coin_position += update.size
                    else:
                        self.coin_position -= update.size
                else:
                    assert False, 'invalid type of update!'

            if self.current_time - vol_prev_time >= self.vol_freq:
                if self.prev_midprice:
                    self.update_volatility()

                self.prev_midprice = self.current_midprice
                vol_prev_time = self.current_time

            if self.current_time - prev_time >= self.delay:
                # place order
                self.update_reservation_price()
                self.update_spread()

                bid_price = round(self.reservation_price - self.spread / 2, 1)  # increment
                ask_price = round(self.reservation_price + self.spread / 2, 1)
                bid_order = sim.place_order(self.current_time, self.trade_size, 'BID', bid_price)
                ask_order = sim.place_order(self.current_time, self.trade_size, 'ASK', ask_price)
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order

                prev_time = self.current_time
                self.quotes_history.append((self.current_time, self.coin_position,
                                            bid_price, self.current_midprice, self.reservation_price, ask_price))

            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < self.current_time - self.hold_time:
                    sim.cancel_order(self.current_time, ID)
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)

        return self.trades_list, self.md_list, self.updates_list, self.quotes_history

    @staticmethod
    def visualize_bids(quotes_history, freq=10):
        time, pos, bid, mid, reservation, ask = list(map(list, zip(*quotes_history[::freq])))

        df = pd.DataFrame([time, pos, bid, mid, reservation, ask]).T
        df.columns = ['time', 'pos', 'bid', 'mid', 'reservation', 'ask']
        breakpoint()
        #TODO: MAKE SURE df["receive_ts"] IS DATETIME
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=df.index, y=df['bid'], name="bid"),
            secondary_y=False)

        fig.add_trace(
            go.Scatter(x=df.index, y=df['mid'], name="mid"),
            secondary_y=False)

        fig.add_trace(
            go.Scatter(x=df.index, y=df['reservation'], name="reservation"),
            secondary_y=False)

        fig.add_trace(
            go.Scatter(x=df.index, y=df['ask'], name="ask"),
            secondary_y=False)

        fig.add_trace(
            go.Scatter(x=df.index, y=df['pos'], name="pos"),
            secondary_y=True)

        fig.update_layout(
            title_text="The mid-price and the optimal bid and ask quotes"
        )

        fig.update_xaxes(title_text="Time")

        fig.update_yaxes(title_text="<b>Prices</b>: USDT", secondary_y=False)
        fig.update_yaxes(title_text="<b>Coin Position</b>: BTC", secondary_y=True)
        fig.show()
        return fig, df


class LimitMarketStrategy:
    """
        This strategy places limit or market orders every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    """

    def __init__(
            self,
            line_coefficients: Tuple[float, float],
            parabola_coefficients: Tuple[float, float, float],
            trade_size: Optional[float] = 0.001,
            price_tick: Optional[float] = 0.1,
            delay: Optional[int] = 1e8,
            hold_time: Optional[int] = 1e10
    ) -> None:
        """
            Args:
                line_coefficients: line coefficients [k, b] y = kx + b
                parabola_coefficients: parabola coefficients [a, b, c] y = ax^2 + bx + c
                trade_size: volume of each trade
                price_tick: a value by which we increase a bid (reduce an ask) limit order
                delay: delay between orders in nanoseconds
                hold_time: holding time in nanoseconds
        """

        self.trade_size = trade_size
        self.delay = delay
        if hold_time is None:
            hold_time = min(delay * 5, pd.Timedelta(10, 's').total_seconds())
        self.hold_time = hold_time

        # market data list
        self.md_list = []
        # executed trades list
        self.trades_list = []
        # all updates list
        self.updates_list = []

        self.current_time = None
        self.coin_position = 0
        self.prev_midprice = None
        self.current_midprice = None
        self.current_spread = None
        self.price_tick = price_tick

        self.line_k, self.line_b = line_coefficients
        self.parabola_a, self.parabola_b, self.parabola_c = parabola_coefficients

        self.actions_history = []

    def get_normalized_data(self) -> Tuple[float, float]:
        # implement normalization
        return self.coin_position, self.current_spread

    def run(self, sim: Sim) -> \
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
                actions_history: list of tuples(time, coin_pos, spread, action)
        """

        # current best positions
        best_bid = -np.inf
        best_ask = np.inf

        # last order timestamp
        prev_time = -np.inf
        # orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        while True:
            # get update from simulator
            self.current_time, updates = sim.tick()
            if updates is None:
                break
            # save updates
            self.updates_list += updates
            for update in updates:
                # update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    self.md_list.append(update)
                    self.current_spread = best_ask - best_bid

                elif isinstance(update, OwnTrade):
                    self.trades_list.append(update)
                    # delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)

                    if update.side == 'BID':
                        self.coin_position += update.size
                    else:
                        self.coin_position -= update.size
                else:
                    assert False, 'invalid type of update!'

            if self.current_time - prev_time >= self.delay:
                # place order
                inventory, spread = self.get_normalized_data()

                if (self.parabola_a * inventory ** 2 + self.parabola_b * inventory + self.parabola_c) > spread:
                    bid_market_order = sim.place_order(self.current_time, self.trade_size, 'BID', best_ask)
                    ongoing_orders[bid_market_order.order_id] = bid_market_order
                    action = 'market buy'
                elif (self.parabola_a * inventory ** 2 + self.parabola_b * (-inventory) + self.parabola_c) > spread:
                    ask_market_order = sim.place_order(self.current_time, self.trade_size, 'ASK', best_bid)
                    ongoing_orders[ask_market_order.order_id] = ask_market_order
                    action = 'market sell'
                else:
                    above_line1 = (self.line_k * inventory + self.line_b) < spread
                    above_line2 = (self.line_k * (-inventory) + self.line_b) < spread

                    bid_price = best_bid + self.price_tick * above_line1
                    ask_price = best_ask - self.price_tick * above_line2

                    bid_limit_order = sim.place_order(self.current_time, self.trade_size, 'BID', bid_price)
                    ask_limit_order = sim.place_order(self.current_time, self.trade_size, 'ASK', ask_price)
                    ongoing_orders[bid_limit_order.order_id] = bid_limit_order
                    ongoing_orders[ask_limit_order.order_id] = ask_limit_order
                    action = 'limit order'

                prev_time = self.current_time
                self.actions_history.append((self.current_time, self.coin_position,
                                             self.current_spread, action))

            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < self.current_time - self.hold_time:
                    sim.cancel_order(self.current_time, ID)
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)

        return self.trades_list, self.md_list, self.updates_list, self.actions_history
    
class StoikovStrategyGeneralizedSingleAsset:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''
    def __init__(self, trade_size: float, position_limit: float, delay: float, hold_time:Optional[float]=None,
                 risk_aversion:Optional[float]=0, k:Optional[float]=1, A:Optional[float]=1) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        self.k = k
        self.A = A
        if hold_time is None:
            hold_time = max(delay * 5, pd.Timedelta(10, 's').total_seconds() )
        self.hold_time = hold_time
        self.order_size = trade_size
        self.last_mid_prices = []
        self.gamma = risk_aversion
        self.Q = position_limit
        self.asset_position = 0
        self.current_bid_order_id = None
        self.current_ask_order_id = None
        self.previous_bid_order_id = None
        self.previous_ask_order_id = None
        self.trades_dict = {'place_ts' :[], 'exchange_ts': [], 'receive_ts': [], 'trade_id': [],'order_id': [],'side': [], 'size': [], 'price': [],'execute':[], 'mid_price':[]}  
    
    @staticmethod
    def visualize_bids(quotes_history, freq=10):
        time, pos, bid, mid, ask = list(map(list, zip(*quotes_history[::freq])))

        df = pd.DataFrame([time, pos, bid, mid, ask]).T
        df.columns = ['time', 'pos', 'bid', 'mid', 'ask']
        breakpoint()
        #TODO: MAKE SURE df["time"] IS DATETIME

        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=df.index, y=df['bid'], name="bid"),
            secondary_y=False)

        fig.add_trace(
            go.Scatter(x=df.index, y=df['mid'], name="mid"),
            secondary_y=False)

        fig.add_trace(
            go.Scatter(x=df.index, y=df['ask'], name="ask"),
            secondary_y=False)

        fig.add_trace(
            go.Scatter(x=df.index, y=df['pos'], name="pos"),
            secondary_y=True)

        fig.update_layout(
            title_text="The mid-price and the optimal bid and ask quotes"
        )

        fig.update_xaxes(title_text="Time")

        fig.update_yaxes(title_text="<b>Prices</b>: USDT", secondary_y=False)
        fig.update_yaxes(title_text="<b>Coin Position</b>: BTC", secondary_y=True)
        fig.show()
        return fig, df
    
    def run(self, sim: Sim ) ->\
        Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''

        #market data list
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        best_bid = -np.inf
        best_ask = np.inf

        #last order timestamp
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        quotes_history = []
        while True:
            #get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            #save updates
            updates_list += updates
            for update in updates:
                #update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    mid_price = (best_bid + best_ask)/2
                    if len(self.last_mid_prices) < 500:
                        self.last_mid_prices.append(mid_price)
                    else:
                        self.last_mid_prices.append(mid_price)
                        self.last_mid_prices.pop(0)
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    self.trades_dict['place_ts'].append(update.place_ts)
                    self.trades_dict['exchange_ts'].append(update.exchange_ts)
                    self.trades_dict['receive_ts'].append(update.receive_ts)
                    self.trades_dict['trade_id'].append(update.trade_id)
                    self.trades_dict['order_id'].append(update.order_id)
                    self.trades_dict['side'].append(update.side)
                    self.trades_dict['size'].append(update.size)
                    self.trades_dict['price'].append(update.price)
                    self.trades_dict['execute'].append(update.execute)
                    self.trades_dict['mid_price'].append(mid_price)
                    if update.side == "ASK":
                        self.asset_position -= update.size
                    elif update.side == "BID":
                        self.asset_position += update.size
                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else: 
                    assert False, 'invalid type of update!'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                #place order
                '''
                reservation_price = s - q * gamma * (sigma**2) * (T - t)
                delta_bid and delta_ask are equivalently distant from the reservation_orice
                delta_bid + delta_ask = gamma * (sigma**2) * (T-t) + 2/gamma * ln(1 + gamma/k)
                k = K*alpha
                
                s      : current mid_price
                q      : current position in asset
                sigma  : parameter in the Geometric Brownian Motion equation [dS_t = sigma dw_t]
                gamma  : risk-aversion parameter of the optimizing agents (across the economy)
                xi     : often referred to as the gamma parameter, but has slightly different meaning - the magnitude towards exponential utility function over linear utility function
                delta  : or equvalently the size of limit orders
                K      : higher K means that market order volumes have higher impact on best price changes
                alpha  : higher alpha means higher probability fo large market orders
                A      : scaling parameter in the density function of market order size 
                Q      : limit of position, the market maker stops providing the limit orders that could make the asset_position violate the limit
                
                '''
                xi = self.gamma
                if len(self.last_mid_prices)==500:
                    sigma = np.std(self.last_mid_prices)## per update --> need to scale it to the "per second" terminology
                else:
                    sigma = 1
                sigma = sigma*np.sqrt(1/0.032)
                delta_t = 0.032 ## there is approximately 0.032 seconds in between the orderbook uprates (nanoseconds / 1e9 = seconds)
                k = self.k
                A = self.A
                q = self.asset_position / self.order_size 
                delta_ = 1
                ## mid_price was defined previously
                if xi != 0:
                    delta_ask = 1/xi/delta_ * np.log(1 + xi*delta_/k) - (2*q - delta_)/2*np.sqrt( self.gamma*(sigma**2)/2/A/delta_/k * (1+xi*delta_/k)**(1+k/xi/delta_) )
                    delta_bid = 1/xi/delta_ * np.log(1 + xi*delta_/k) + (2*q + delta_)/2*np.sqrt( self.gamma*(sigma**2)/2/A/delta_/k * (1+xi*delta_/k)**(1+k/xi/delta_) )
                else:
                    delta_ask = 1 / k + (2 * q + delta_) / 2 * np.sqrt(self.gamma * sigma**2 * np.e / (2 * A * delta_ * k))
                    delta_bid = 1 / k - (2 * q - delta_) / 2 * np.sqrt(self.gamma * sigma**2 * np.e / (2 * A * delta_ * k))
                
                bid_price = np.round(mid_price - delta_bid, 1)
                ask_price = np.round(mid_price + delta_ask, 1)
                if (self.asset_position < self.Q):
                    bid_order = sim.place_order( receive_ts, self.order_size, 'BID', bid_price)
                    ongoing_orders[bid_order.order_id] = bid_order
                    self.previous_bid_order_id = self.current_bid_order_id
                    self.current_bid_order_id = bid_order.order_id
#                     all_orders.append(bid_order)
                    
                if (self.asset_position > -self.Q):
                    ask_order = sim.place_order( receive_ts, self.order_size, 'ASK', ask_price)
                    ongoing_orders[ask_order.order_id] = ask_order
                    self.previous_ask_order_id = self.current_ask_order_id
                    self.current_ask_order_id = ask_order.order_id
#                     all_orders.append(ask_order)
                    
                quotes_history.append((receive_ts, self.asset_position,
                                       bid_price, mid_price, ask_price))
            
            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            if self.previous_bid_order_id in ongoing_orders.keys():
                sim.cancel_order( receive_ts, self.previous_bid_order_id )
                to_cancel.append(self.previous_bid_order_id)
            if self.previous_ask_order_id in ongoing_orders.keys():
                sim.cancel_order( receive_ts, self.previous_ask_order_id )
                to_cancel.append(self.previous_ask_order_id)
            for ID in to_cancel:
                ongoing_orders.pop(ID)
            
                
        return trades_list, md_list, updates_list, quotes_history
