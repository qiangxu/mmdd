#!/usr/bin/env python
# CREATED DATE: 2024年12月18日 星期三 15时46分09秒
# CREATED BY: qiangxu, toxuqiang@gmail.com

import json
import os
import pandas as pd

def aggregate_orderbook(input_file):
    """
    聚合订单簿数据，按价格区间聚合买单和卖单
    
    :param input_file: 输入的JSON日志文件
    """
       
    outputs = [] 
    
    # 处理文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 用于存储聚合结果的字典
                aggregated_data = {
                    'bids': {},  # 买单聚合
                    'asks': {}   # 卖单聚合
                }

                # 解析JSON行
                entry = json.loads(line)
                message = entry.get('message', {})
                output = {} 
                # 聚合买单
                for bid in message.get('b', []):
                    price = float(bid[0])
                    quantity = float(bid[1])
                    
                    # 取整到小数点第一位
                    rounded_price = round(price, 1)
                    
                    # 累加相同价格区间的数量
                    if rounded_price not in aggregated_data['bids']:
                        aggregated_data['bids'][rounded_price] = 0
                    aggregated_data['bids'][rounded_price] += quantity
                
                # 聚合卖单（逻辑同买单）
                for ask in message.get('a', []):
                    price = float(ask[0])
                    quantity = float(ask[1])
                    
                    # 取整到小数点第一位
                    rounded_price = round(price, 1)
                    
                    # 累加相同价格区间的数量
                    if rounded_price not in aggregated_data['asks']:
                        aggregated_data['asks'][rounded_price] = 0
                    aggregated_data['asks'][rounded_price] += quantity
                   
                bids = sorted(aggregated_data['bids'].items(), reverse=True)
                for i in range(10):
                    if len(bids) > i: 
                        output["bid_price_%d" % i] = bids[i][0]
                        output["bid_vol_%d" % i] = bids[i][1]
                    else: 
                        output["bid_price_%d" % i] = bids[-1][0] - 0.1
                        output["bid_vol_%d" % i] = 0

                asks = sorted(aggregated_data['asks'].items())
                for i in range(10):
                    if len(asks) > i: 
                        output["ask_price_%d" % i] = asks[i][0]
                        output["ask_vol_%d" % i] = asks[i][1]
                    else: 
                        output["ask_price_%d" % i] = asks[-1][0] + 0.1
                        output["ask_vol_%d" % i] = 0

                output["receive_ts"] = entry['timestamp']
                output["exchange_ts"] = entry['timestamp']

                outputs.append(output)
            except json.JSONDecodeError:
                print(f"跳过无效的JSON行: {line}")
    
    return outputs

def main():
    # 从命令行参数获取输入文件
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python script.py <input_file> [output_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
   
    # 执行聚合
    result = aggregate_orderbook(input_file)

    breakpoint()
    df = pd.DataFrame(result)
    df.to_csv(os.path.splitext(input_file)[0] + '.csv')
    
if __name__ == "__main__":
    main()

# 使用示例：
# python script.py paste.txt
# python script.py paste.txt output_aggregated.json
