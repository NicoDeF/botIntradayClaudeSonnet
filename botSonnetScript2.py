import asyncio
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance import ThreadedWebsocketManager
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional, Tuple
import threading

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scalping_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ScalpingStrategy:
    """
    Implementación de la estrategia Scalping PullBack Tool basada en:
    - EMAs (89, 200, 600)
    - PAC Channel (EMA 34 de High, Low, Close)
    - Detección de pullbacks y recuperaciones
    """

    def __init__(self,
                 pac_length: int = 34,
                 fast_ema: int = 89,
                 medium_ema: int = 200,
                 slow_ema: int = 600,
                 lookback: int = 3):
        self.pac_length = pac_length
        self.fast_ema = fast_ema
        self.medium_ema = medium_ema
        self.slow_ema = slow_ema
        self.lookback = lookback

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todos los indicadores necesarios para la estrategia"""

        # EMAs
        df['fast_ema'] = talib.EMA(df['close'], timeperiod=self.fast_ema)
        df['medium_ema'] = talib.EMA(df['close'], timeperiod=self.medium_ema)
        df['slow_ema'] = talib.EMA(df['close'], timeperiod=self.slow_ema)

        # PAC Channel (Price Action Channel)
        df['pac_close'] = talib.EMA(df['close'], timeperiod=self.pac_length)
        df['pac_high'] = talib.EMA(df['high'], timeperiod=self.pac_length)
        df['pac_low'] = talib.EMA(df['low'], timeperiod=self.pac_length)

        # Trend Direction
        df['trend_direction'] = np.where(
            (df['fast_ema'] > df['medium_ema']) & (df['pac_low'] > df['medium_ema']), 1,
            np.where(
                (df['fast_ema'] < df['medium_ema']) & (df['pac_high'] < df['medium_ema']), -1, 0
            )
        )

        # Bar Colors según posición relativa al PAC
        df['bar_color'] = np.where(
            df['close'] > df['pac_high'], 'blue',  # Arriba del PAC
            np.where(df['close'] < df['pac_low'], 'red', 'gray')  # Abajo/Dentro del PAC
        )

        return df

    def detect_pullback_signals(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """Detecta señales de pullback según la estrategia"""

        if len(df) < max(self.fast_ema, self.medium_ema, self.pac_length) + self.lookback:
            return False, False

        current = df.iloc[-1]

        # Condiciones para Buy Signal (Pullback al alza)
        buy_conditions = []

        # 1. Trend alcista
        buy_conditions.append(current['trend_direction'] == 1)

        # 2. Pullback: precio sale del PAC hacia arriba
        # Verificamos si en los últimos lookback períodos estuvo debajo del PAC center
        recent_below_pac = False
        for i in range(1, self.lookback + 1):
            if len(df) > i and df.iloc[-i]['close'] < df.iloc[-i]['pac_close']:
                recent_below_pac = True
                break

        # 3. Recuperación: precio cruza por encima del PAC upper
        pac_exit_up = (current['open'] < current['pac_high'] and
                       current['close'] > current['pac_high'] and
                       recent_below_pac)

        buy_conditions.append(pac_exit_up)

        buy_signal = all(buy_conditions)

        # Condiciones para Sell Signal (Pullback a la baja)
        sell_conditions = []

        # 1. Trend bajista
        sell_conditions.append(current['trend_direction'] == -1)

        # 2. Pullback: precio sale del PAC hacia abajo
        recent_above_pac = False
        for i in range(1, self.lookback + 1):
            if len(df) > i and df.iloc[-i]['close'] > df.iloc[-i]['pac_close']:
                recent_above_pac = True
                break

        # 3. Recuperación: precio cruza por debajo del PAC lower
        pac_exit_down = (current['open'] > current['pac_low'] and
                         current['close'] < current['pac_low'] and
                         recent_above_pac)

        sell_conditions.append(pac_exit_down)

        sell_signal = all(sell_conditions)

        return buy_signal, sell_signal


class BinanceScalpingBot:
    """Bot principal de scalping para SOL en Binance Futures usando WebSocket nativo de Binance"""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        # Configuración del cliente
        if testnet:
            self.client = Client(api_key, api_secret, testnet=True)
            self.client.API_URL = 'https://testnet.binancefuture.com/fapi'
        else:
            self.client = Client(api_key, api_secret)

        # Configuración de trading
        self.symbol = 'SOLUSDT'
        self.timeframe = '1m'  # 1 minuto para scalping
        self.quantity = 0.1  # Cantidad por trade (ajustar según capital)
        self.leverage = 10  # Apalancamiento
        self.stop_loss_pct = 0.5  # 0.5% stop loss
        self.take_profit_pct = 1.0  # 1% take profit

        # Estado del bot
        self.position = None
        self.is_running = False
        self.strategy = ScalpingStrategy()

        # WebSocket Manager
        self.twm = None

        # Configurar apalancamiento
        try:
            self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
            logger.info(f"Apalancamiento configurado a {self.leverage}x para {self.symbol}")
        except Exception as e:
            logger.error(f"Error configurando apalancamiento: {e}")

    def get_historical_data(self, limit: int = 1000) -> pd.DataFrame:
        """Obtiene datos históricos para inicializar los indicadores"""
        try:
            klines = self.client.futures_klines(
                symbol=self.symbol,
                interval=self.timeframe,
                limit=limit
            )

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # Convertir a tipos numéricos
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            return df[numeric_columns]

        except Exception as e:
            logger.error(f"Error obteniendo datos históricos: {e}")
            return pd.DataFrame()

    def get_current_position(self) -> Optional[Dict]:
        """Obtiene la posición actual"""
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
            for pos in positions:
                if float(pos['positionAmt']) != 0:
                    return pos
            return None
        except Exception as e:
            logger.error(f"Error obteniendo posición: {e}")
            return None

    def place_market_order(self, side: str, quantity: float) -> Optional[Dict]:
        """Coloca una orden de mercado"""
        try:
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            logger.info(f"Orden {side} ejecutada: {order}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Error en orden {side}: {e}")
            return None

    def place_stop_order(self, side: str, quantity: float, stop_price: float) -> Optional[Dict]:
        """Coloca una orden stop loss"""
        try:
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type='STOP_MARKET',
                quantity=quantity,
                stopPrice=stop_price,
                reduceOnly=True
            )
            logger.info(f"Stop Loss colocado: {order}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Error en stop loss: {e}")
            return None

    def place_take_profit_order(self, side: str, quantity: float, price: float) -> Optional[Dict]:
        """Coloca una orden take profit"""
        try:
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type='TAKE_PROFIT_MARKET',
                quantity=quantity,
                stopPrice=price,
                reduceOnly=True
            )
            logger.info(f"Take Profit colocado: {order}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Error en take profit: {e}")
            return None

    def execute_buy_signal(self, current_price: float):
        """Ejecuta señal de compra"""
        if self.position is not None:
            logger.info("Ya hay una posición abierta, ignorando señal de compra")
            return

        # Abrir posición long
        order = self.place_market_order('BUY', self.quantity)
        if order:
            # Calcular precios de stop loss y take profit
            stop_price = current_price * (1 - self.stop_loss_pct / 100)
            tp_price = current_price * (1 + self.take_profit_pct / 100)

            # Colocar órdenes de protección
            self.place_stop_order('SELL', self.quantity, stop_price)
            self.place_take_profit_order('SELL', self.quantity, tp_price)

            self.position = {
                'side': 'LONG',
                'entry_price': current_price,
                'quantity': self.quantity,
                'stop_price': stop_price,
                'tp_price': tp_price
            }

            logger.info(f"Posición LONG abierta: Precio={current_price}, SL={stop_price}, TP={tp_price}")

    def execute_sell_signal(self, current_price: float):
        """Ejecuta señal de venta"""
        if self.position is not None:
            logger.info("Ya hay una posición abierta, ignorando señal de venta")
            return

        # Abrir posición short
        order = self.place_market_order('SELL', self.quantity)
        if order:
            # Calcular precios de stop loss y take profit
            stop_price = current_price * (1 + self.stop_loss_pct / 100)
            tp_price = current_price * (1 - self.take_profit_pct / 100)

            # Colocar órdenes de protección
            self.place_stop_order('BUY', self.quantity, stop_price)
            self.place_take_profit_order('BUY', self.quantity, tp_price)

            self.position = {
                'side': 'SHORT',
                'entry_price': current_price,
                'quantity': self.quantity,
                'stop_price': stop_price,
                'tp_price': tp_price
            }

            logger.info(f"Posición SHORT abierta: Precio={current_price}, SL={stop_price}, TP={tp_price}")

    def handle_socket_message(self, msg):
        """Maneja mensajes del WebSocket de klines"""
        try:
            # Solo procesar klines cerradas
            if msg['k']['x']:  # kline is closed
                self.process_new_kline(msg['k'])

        except Exception as e:
            logger.error(f"Error procesando mensaje WebSocket: {e}")

    def process_new_kline(self, kline_data: Dict):
        """Procesa una nueva kline y verifica señales"""
        try:
            # Obtener datos históricos actualizados
            df = self.get_historical_data(limit=800)  # Suficientes datos para indicadores

            if df.empty:
                return

            # Calcular indicadores
            df = self.strategy.calculate_indicators(df)

            # Detectar señales
            buy_signal, sell_signal = self.strategy.detect_pullback_signals(df)

            current_price = df['close'].iloc[-1]

            # Log de estado actual
            current_trend = df['trend_direction'].iloc[-1]
            trend_text = "ALCISTA" if current_trend == 1 else "BAJISTA" if current_trend == -1 else "NEUTRAL"

            logger.info(f"Precio: {current_price:.4f} | Trend: {trend_text} | "
                        f"PAC: {df['pac_close'].iloc[-1]:.4f} | "
                        f"Buy: {buy_signal} | Sell: {sell_signal}")

            # Ejecutar señales
            if buy_signal:
                logger.info("🟢 SEÑAL DE COMPRA DETECTADA")
                self.execute_buy_signal(current_price)
            elif sell_signal:
                logger.info("🔴 SEÑAL DE VENTA DETECTADA")
                self.execute_sell_signal(current_price)

            # Verificar estado de posición actual
            current_position = self.get_current_position()
            if current_position is None and self.position is not None:
                logger.info("Posición cerrada por SL/TP")
                self.position = None

        except Exception as e:
            logger.error(f"Error procesando nueva kline: {e}")

    def start_websocket(self):
        """Inicia el WebSocket usando ThreadedWebSocketManager"""
        try:
            self.twm = ThreadedWebsocketManager(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet
            )
            self.twm.start()

            # Iniciar stream de klines
            self.twm.start_kline_futures_socket(
                callback=self.handle_socket_message,
                symbol=self.symbol,
                interval=self.timeframe
            )

            logger.info("WebSocket iniciado correctamente")

        except Exception as e:
            logger.error(f"Error iniciando WebSocket: {e}")

    def run(self):
        """Inicia el bot de scalping"""
        logger.info(f"Iniciando bot de scalping para {self.symbol}")
        logger.info(f"Configuración: Apalancamiento={self.leverage}x, "
                    f"SL={self.stop_loss_pct}%, TP={self.take_profit_pct}%")

        self.is_running = True

        # Inicializar datos históricos
        df = self.get_historical_data()
        if df.empty:
            logger.error("No se pudieron obtener datos históricos")
            return

        # Iniciar WebSocket
        self.start_websocket()

        try:
            while self.is_running:
                # Verificar conexión y estado cada minuto
                time.sleep(60)

                # Verificar posición actual
                current_pos = self.get_current_position()
                if current_pos:
                    pnl = float(current_pos['unrealizedPnl'])
                    logger.info(f"Posición actual: {current_pos['side']} | PnL: {pnl:.4f} USDT")

        except KeyboardInterrupt:
            logger.info("Bot detenido por el usuario")
        finally:
            self.stop()

    def stop(self):
        """Detiene el bot"""
        self.is_running = False
        if self.twm:
            self.twm.stop()
        logger.info("Bot detenido")


# Configuración y ejecución
if __name__ == "__main__":
    # IMPORTANTE: Reemplaza con tus credenciales reales
    API_KEY = 'bPT08S5D4JijR0qJDAZh2ILB35yjQ1s5ozHxmVcY0qGL4eywVeVHPy1PjfgCkkwK'
    API_SECRET = '2xIUXyeIA82A367HPrHXklozZl6DafBQJjT9sVfsv6TfdOoxMCt7LTKRc2BPsagj'

    # Usar testnet para pruebas (cambiar a False para trading real)
    USE_TESTNET = False

    # Crear y ejecutar el bot
    bot = BinanceScalpingBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        testnet=USE_TESTNET
    )

    # Configuraciones adicionales del bot
    bot.quantity = 0.1  # Ajustar según tu capital
    bot.leverage = 10  # Apalancamiento
    bot.stop_loss_pct = 0.5  # 0.5% stop loss
    bot.take_profit_pct = 1.0  # 1% take profit

    # Ejecutar el bot
    bot.run()