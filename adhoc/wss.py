#!/usr/bin/env python
# CREATED DATE: 2024年12月17日 星期二 16时54分03秒
# CREATED BY: qiangxu, toxuqiang@gmail.com

import websocket
import json
import threading
import os
from datetime import datetime

class BinanceOrderbookWebSocket:
    def __init__(self, symbol='btcusdt', log_dir='./orderbook_logs'):
        """
        初始化Binance订单簿WebSocket
        :param symbol: 交易对符号，默认为btcusdt（小写）
        """
        self.symbol = symbol.lower()
        self.socket_url = f'wss://stream.binance.com:9443/ws/{self.symbol}@depth@1000ms'
        self.ws = None

        self.is_running = threading.Event()
        
        # 创建日志目录
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建日志文件
        self.log_file = self._create_log_file()
        
        self.orderbook = {
            'bids': {},  # 买单
            'asks': {}   # 卖单
        }
        
        # 线程安全锁，用于文件写入
        self.file_lock = threading.Lock()

    def _create_log_file(self):
        """
        创建带时间戳的日志文件
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.symbol}_orderbook_{timestamp}.json"
        return os.path.join(self.log_dir, filename)

    def _write_message_to_file(self, message):
        """
        将原始消息写入日志文件
        :param message: WebSocket接收的原始消息
        """
        try:
            with self.file_lock:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    # 添加时间戳和换行
                    log_entry = json.dumps({
                        'timestamp': datetime.now().isoformat(),
                        'message': json.loads(message)
                    }) + '\n'
                    f.write(log_entry)
        except Exception as e:
            print(f"日志写入错误: {e}")

    def on_message(self, ws, message):
        """
        处理接收到的WebSocket消息
        """
        data = json.loads(message)
        self._write_message_to_file(message)
        
        data = json.loads(message)
      

    def on_error(self, ws, error):
        """
        处理WebSocket错误
        """
        print(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """
        WebSocket连接关闭时的处理
        """
        print("WebSocket Connection Closed")

    def on_open(self, ws):
        """
        WebSocket连接建立时的处理
        """
        print(f"Connected to Binance WebSocket for {self.symbol.upper()} Orderbook")

    def start(self):
        """
        启动WebSocket连接
        """
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            self.socket_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        # 在独立线程中运行WebSocket
        #wst = threading.Thread(target=self.ws.run_forever)
        #wst.daemon = True
        #wst.start()
        self.ws.run_forever()

    def stop(self):
        """
        停止WebSocket连接
        """
        if self.ws:
            self.ws.close()

def main():
    # 创建BTC/USDT订单簿WebSocket实例
    btc_orderbook = BinanceOrderbookWebSocket(symbol='btcusdt')
    
    try:
        # 启动WebSocket连接
        btc_orderbook.start()
        
        # 保持主线程运行
        input("按回车键退出...\n")
    
    except KeyboardInterrupt:
        print("程序被中断")
    
    finally:
        # 关闭WebSocket连接
        btc_orderbook.stop()

if __name__ == "__main__":
    main()

# 使用说明
# 1. 安装依赖: pip install websocket-client
# 2. 运行脚本即可开始接收实时订单簿数据
# 3. 可以修改symbol参数获取其他交易对的订单簿信息
