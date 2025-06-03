import asyncio
import fileinput
import logging
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance import ThreadedWebsocketManager
import pandas as pd
import numpy as np
import talib
import os
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional, Tuple
import threading
import sys

# Configuración de logging con encoding UTF-8 y formato mejorado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scalping_bot.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class LoggingHelper:
    """Clase auxiliar para manejo de logging estructurado y legible"""

    @staticmethod
    def log_section_header(title: str, symbol: str = "="):
        """Crea un encabezado de sección con separadores"""
        separator = symbol * 80
        logger.info(separator)
        logger.info(f"  {title.upper()}")
        logger.info(separator)

    @staticmethod
    def log_subsection_header(title: str, symbol: str = "-"):
        """Crea un encabezado de subsección"""
        separator = symbol * 60
        logger.info(separator)
        logger.info(f"  {title}")
        logger.info(separator)

    @staticmethod
    def log_action_start(action_name: str):
        """Log inicio de una acción específica"""
        logger.info("=" * 80)
        logger.info(f"🔄 INICIANDO: {action_name.upper()}")
        logger.info("=" * 80)

    @staticmethod
    def log_action_success(action_name: str, details: str = ""):
        """Log éxito de una acción"""
        logger.info("=" * 80)
        logger.info(f"✅ COMPLETADO: {action_name.upper()}")
        if details:
            logger.info(f"   {details}")
        logger.info("=" * 80)

    @staticmethod
    def log_action_error(action_name: str, error: str):
        """Log error de una acción"""
        logger.info("=" * 80)
        logger.info(f"❌ ERROR EN: {action_name.upper()}")
        logger.info(f"   {error}")
        logger.info("=" * 80)

    @staticmethod
    def log_market_status(price: float, trend: str, pac_price: float, buy_signal: bool, sell_signal: bool):
        """Log estado del mercado de forma estructurada"""
        LoggingHelper.log_subsection_header("ESTADO DEL MERCADO")
        logger.info(f"   💰 Precio actual: ${price:.4f}")
        logger.info(f"   📊 Tendencia: {trend}")
        logger.info(f"   🎯 PAC Center: ${pac_price:.4f}")
        logger.info(f"   🚀 Señal Compra: {'✅ ACTIVA' if buy_signal else '❌ Inactiva'}")
        logger.info(f"   🔻 Señal Venta: {'✅ ACTIVA' if sell_signal else '❌ Inactiva'}")
        logger.info("-" * 60)


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
        LoggingHelper.log_action_start("CÁLCULO DE INDICADORES TÉCNICOS")

        try:
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

            LoggingHelper.log_action_success("CÁLCULO DE INDICADORES TÉCNICOS",
                                             f"EMAs: {self.fast_ema}, {self.medium_ema}, {self.slow_ema} | PAC: {self.pac_length}")

            return df

        except Exception as e:
            LoggingHelper.log_action_error("CÁLCULO DE INDICADORES TÉCNICOS", str(e))
            return df

    def detect_pullback_signals(self, df: pd.DataFrame) -> Tuple[bool, bool]:
        """Detecta señales de pullback según la estrategia"""
        LoggingHelper.log_action_start("DETECCIÓN DE SEÑALES DE TRADING")

        try:
            if len(df) < max(self.fast_ema, self.medium_ema, self.pac_length) + self.lookback:
                logger.info("⚠️ Datos insuficientes para análisis de señales")
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
            pac_exit_up = (current['open'] < current['pac_high'] < current['close'] and
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

            # Log detallado de condiciones
            logger.info("📋 ANÁLISIS DE CONDICIONES:")
            logger.info(
                f"   🔄 Trend Direction: {current['trend_direction']} ({'Alcista' if current['trend_direction'] == 1 else 'Bajista' if current['trend_direction'] == -1 else 'Neutral'})")
            logger.info(f"   📊 PAC Exit Up: {'✅' if pac_exit_up else '❌'}")
            logger.info(f"   📊 PAC Exit Down: {'✅' if pac_exit_down else '❌'}")
            logger.info(f"   🎯 Recent Below PAC: {'✅' if recent_below_pac else '❌'}")
            logger.info(f"   🎯 Recent Above PAC: {'✅' if recent_above_pac else '❌'}")

            LoggingHelper.log_action_success("DETECCIÓN DE SEÑALES DE TRADING",
                                             f"Buy Signal: {'✅' if buy_signal else '❌'} | Sell Signal: {'✅' if sell_signal else '❌'}")

            return buy_signal, sell_signal

        except Exception as e:
            LoggingHelper.log_action_error("DETECCIÓN DE SEÑALES DE TRADING", str(e))
            return False, False


class BinanceScalpingBot:
    """Bot principal de scalping para SOL en Binance Futures usando WebSocket nativo de Binance"""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        LoggingHelper.log_action_start("INICIALIZACIÓN DEL BOT")

        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        # Configuración del cliente
        if testnet:
            self.client = Client(api_key, api_secret, testnet=True)
            self.client.API_URL = 'https://testnet.binancefuture.com/fapi'
            logger.info("🧪 Modo TESTNET activado")
        else:
            self.client = Client(api_key, api_secret)
            logger.info("🔴 Modo MAINNET activado")

        ##################################################
        #
        # Configuración de trading
        #
        ##################################################

        self.symbol = 'SOLUSDT'
        self.timeframe = '1m'  # 1 minuto para scalping
        self.quantity = 0.2  # Cantidad por trade (ajustar según capital)
        self.leverage = 10  # Apalancamiento
        self.stop_loss_pct = 1.0  # 1% stop loss
        self.take_profit_pct = 2.0  # 2% take profit

        # Estado del bot
        self.position = None
        self.is_running = False
        self.strategy = ScalpingStrategy()
        self.price_precision = 4  # Precisión de precio para SOL
        self.qty_precision = 1  # Precisión de cantidad para SOL
        self.bot_name = os.path.splitext(os.path.basename(__file__))[0]

        # WebSocket Manager
        self.twm = None

        # Variables para tracking de órdenes de SL/TP
        self.active_stop_order_id = None
        self.active_tp_order_id = None

        # Obtener información del símbolo para precisión
        self.get_symbol_info()

        # Configurar apalancamiento
        self.setup_leverage()

        LoggingHelper.log_action_success("INICIALIZACIÓN DEL BOT", "Configuración completada")

    def setup_leverage(self):
        """Configura el apalancamiento para el símbolo"""
        LoggingHelper.log_action_start("CONFIGURACIÓN DE APALANCAMIENTO")

        try:
            self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
            LoggingHelper.log_action_success("CONFIGURACIÓN DE APALANCAMIENTO",
                                             f"{self.leverage}x para {self.symbol}")
        except Exception as e:
            LoggingHelper.log_action_error("CONFIGURACIÓN DE APALANCAMIENTO", str(e))

    def get_symbol_info(self):
        """Obtiene información del símbolo para configurar precisión"""
        LoggingHelper.log_action_start("OBTENCIÓN DE INFORMACIÓN DEL SÍMBOLO")

        try:
            exchange_info = self.client.futures_exchange_info()
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == self.symbol:
                    # Obtener precisión de precio
                    for filter_info in symbol_info['filters']:
                        if filter_info['filterType'] == 'PRICE_FILTER':
                            tick_size = float(filter_info['tickSize'])
                            self.price_precision = len(str(tick_size).rstrip('0').split('.')[1]) if '.' in str(
                                tick_size) else 0
                        elif filter_info['filterType'] == 'LOT_SIZE':
                            step_size = float(filter_info['stepSize'])
                            self.qty_precision = len(str(step_size).rstrip('0').split('.')[1]) if '.' in str(
                                step_size) else 0
                    break

            LoggingHelper.log_action_success("OBTENCIÓN DE INFORMACIÓN DEL SÍMBOLO",
                                             f"Precio: {self.price_precision} decimales | Cantidad: {self.qty_precision} decimales")
        except Exception as e:
            LoggingHelper.log_action_error("OBTENCIÓN DE INFORMACIÓN DEL SÍMBOLO", str(e))

    def round_price(self, price: float) -> float:
        """Redondea el precio según la precisión del símbolo"""
        return round(price, self.price_precision)

    def round_quantity(self, quantity: float) -> float:
        """Redondea la cantidad según la precisión del símbolo"""
        return round(quantity, self.qty_precision)

    def get_historical_data(self, limit: int = 1000) -> pd.DataFrame:
        """Obtiene datos históricos para inicializar los indicadores"""
        LoggingHelper.log_action_start("OBTENCIÓN DE DATOS HISTÓRICOS")

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

            LoggingHelper.log_action_success("OBTENCIÓN DE DATOS HISTÓRICOS",
                                             f"{len(df)} velas obtenidas para {self.symbol}")

            return df[numeric_columns]

        except Exception as e:
            LoggingHelper.log_action_error("OBTENCIÓN DE DATOS HISTÓRICOS", str(e))
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
            logger.error(f"❌ Error obteniendo posición: {e}")
            return None

    def log_order_details(self, order_type: str, order: Dict, additional_info: str = ""):
        """Log detallado de órdenes con información de SL/TP"""
        LoggingHelper.log_subsection_header(f"DETALLES DE ORDEN - {order_type}")

        if order:
            order_id = order.get('orderId', 'N/A')
            side = order.get('side', 'N/A')
            quantity = order.get('origQty', order.get('quantity', 'N/A'))
            price = order.get('price', order.get('stopPrice', 'N/A'))

            logger.info(f"   📋 ID de Orden: {order_id}")
            logger.info(f"   🔄 Dirección: {side}")
            logger.info(f"   📊 Cantidad: {quantity} {self.symbol}")
            logger.info(f"   💰 Precio: {price}")
            if additional_info:
                logger.info(f"   ℹ️ Información adicional: {additional_info}")

        logger.info("-" * 60)

    def place_market_order(self, side: str, quantity: float) -> Optional[Dict]:
        """Coloca una orden de mercado"""
        LoggingHelper.log_action_start(f"ORDEN DE MERCADO - {side}")

        try:
            quantity = self.round_quantity(quantity)
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )

            if order:
                LoggingHelper.log_action_success(f"ORDEN DE MERCADO - {side}",
                                                 f"Cantidad: {quantity} {self.symbol}")
                self.log_order_details("ORDEN DE MERCADO EJECUTADA", order)

            return order
        except BinanceAPIException as e:
            LoggingHelper.log_action_error(f"ORDEN DE MERCADO - {side}", str(e))
            return None

    def place_stop_order(self, side: str, quantity: float, stop_price: float) -> Optional[Dict]:
        """Coloca una orden stop loss"""
        LoggingHelper.log_action_start(f"COLOCACIÓN DE STOP LOSS - {side}")

        try:
            quantity = self.round_quantity(quantity)
            stop_price = self.round_price(stop_price)
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type='STOP_MARKET',
                quantity=quantity,
                stopPrice=stop_price,
                reduceOnly=True
            )

            # Guardar ID de la orden de stop loss
            if order:
                self.active_stop_order_id = order.get('orderId')
                LoggingHelper.log_action_success(f"COLOCACIÓN DE STOP LOSS - {side}",
                                                 f"Precio: ${stop_price} | Cantidad: {quantity}")
                self.log_order_details("STOP LOSS COLOCADO", order, "🛡️ Protección activada")

            return order
        except BinanceAPIException as e:
            LoggingHelper.log_action_error(f"COLOCACIÓN DE STOP LOSS - {side}", str(e))
            return None

    def place_take_profit_order(self, side: str, quantity: float, price: float) -> Optional[Dict]:
        """Coloca una orden take profit"""
        LoggingHelper.log_action_start(f"COLOCACIÓN DE TAKE PROFIT - {side}")

        try:
            quantity = self.round_quantity(quantity)
            price = self.round_price(price)
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type='TAKE_PROFIT_MARKET',
                quantity=quantity,
                stopPrice=price,
                reduceOnly=True
            )

            # Guardar ID de la orden de take profit
            if order:
                self.active_tp_order_id = order.get('orderId')
                LoggingHelper.log_action_success(f"COLOCACIÓN DE TAKE PROFIT - {side}",
                                                 f"Precio: ${price} | Cantidad: {quantity}")
                self.log_order_details("TAKE PROFIT COLOCADO", order, "🎯 Objetivo establecido")

            return order
        except BinanceAPIException as e:
            LoggingHelper.log_action_error(f"COLOCACIÓN DE TAKE PROFIT - {side}", str(e))
            return None

    def calculate_risk_reward(self, entry_price: float, stop_price: float, tp_price: float, side: str) -> Dict:
        """Calcula el riesgo/recompensa de la operación"""
        LoggingHelper.log_action_start("CÁLCULO DE RIESGO/RECOMPENSA")

        try:
            if side == 'LONG':
                risk = abs(entry_price - stop_price)
                reward = abs(tp_price - entry_price)
            else:  # SHORT
                risk = abs(stop_price - entry_price)
                reward = abs(entry_price - tp_price)

            risk_reward_ratio = reward / risk if risk > 0 else 0
            risk_usdt = risk * self.quantity
            reward_usdt = reward * self.quantity

            result = {
                'risk': risk,
                'reward': reward,
                'risk_reward_ratio': risk_reward_ratio,
                'risk_usdt': risk_usdt,
                'reward_usdt': reward_usdt
            }

            LoggingHelper.log_action_success("CÁLCULO DE RIESGO/RECOMPENSA",
                                             f"Ratio: 1:{risk_reward_ratio:.2f} | Riesgo: ${risk_usdt:.2f} | Recompensa: ${reward_usdt:.2f}")

            return result
        except Exception as e:
            LoggingHelper.log_action_error("CÁLCULO DE RIESGO/RECOMPENSA", str(e))
            return {}

    def execute_buy_signal(self, current_price: float):
        """Ejecuta señal de compra"""
        if self.position is not None:
            logger.info("⚠️ Ya hay una posición abierta, ignorando señal de compra")
            return

        LoggingHelper.log_section_header("EJECUTANDO SEÑAL DE COMPRA (LONG)", "🚀")

        # Abrir posición long
        order = self.place_market_order('BUY', self.quantity)
        if order:
            # Calcular precios de stop loss y take profit
            stop_price = self.round_price(current_price * (1 - self.stop_loss_pct / 100))
            tp_price = self.round_price(current_price * (1 + self.take_profit_pct / 100))

            # Calcular métricas de riesgo/recompensa
            rr_metrics = self.calculate_risk_reward(current_price, stop_price, tp_price, 'LONG')

            # Log detallado de la operación
            LoggingHelper.log_subsection_header("DETALLES DE LA POSICIÓN LONG")
            logger.info(f"   💰 Precio de entrada: ${current_price:.4f}")
            logger.info(f"   🛡️ Stop Loss: ${stop_price:.4f} (-{self.stop_loss_pct}%)")
            logger.info(f"   🎯 Take Profit: ${tp_price:.4f} (+{self.take_profit_pct}%)")
            logger.info(f"   📊 Cantidad: {self.quantity} {self.symbol.replace('USDT', '')}")
            logger.info(f"   💵 Capital en riesgo: ${rr_metrics['risk_usdt']:.2f}")
            logger.info(f"   💎 Ganancia potencial: ${rr_metrics['reward_usdt']:.2f}")
            logger.info(f"   ⚖️ Ratio Riesgo/Recompensa: 1:{rr_metrics['risk_reward_ratio']:.2f}")

            # Colocar órdenes de protección
            self.place_stop_order('SELL', self.quantity, stop_price)
            self.place_take_profit_order('SELL', self.quantity, tp_price)

            self.position = {
                'side': 'LONG',
                'entry_price': current_price,
                'quantity': self.quantity,
                'stop_price': stop_price,
                'tp_price': tp_price,
                'entry_time': datetime.now(),
                'risk_reward_metrics': rr_metrics
            }

            LoggingHelper.log_section_header("POSICIÓN LONG ESTABLECIDA EXITOSAMENTE", "✅")

    def execute_sell_signal(self, current_price: float):
        """Ejecuta señal de venta"""
        if self.position is not None:
            logger.info("⚠️ Ya hay una posición abierta, ignorando señal de venta")
            return

        LoggingHelper.log_section_header("EJECUTANDO SEÑAL DE VENTA (SHORT)", "🔻")

        # Abrir posición short
        order = self.place_market_order('SELL', self.quantity)
        if order:
            # Calcular precios de stop loss y take profit
            stop_price = self.round_price(current_price * (1 + self.stop_loss_pct / 100))
            tp_price = self.round_price(current_price * (1 - self.take_profit_pct / 100))

            # Calcular métricas de riesgo/recompensa
            rr_metrics = self.calculate_risk_reward(current_price, stop_price, tp_price, 'SHORT')

            # Log detallado de la operación
            LoggingHelper.log_subsection_header("DETALLES DE LA POSICIÓN SHORT")
            logger.info(f"   💰 Precio de entrada: ${current_price:.4f}")
            logger.info(f"   🛡️ Stop Loss: ${stop_price:.4f} (+{self.stop_loss_pct}%)")
            logger.info(f"   🎯 Take Profit: ${tp_price:.4f} (-{self.take_profit_pct}%)")
            logger.info(f"   📊 Cantidad: {self.quantity} {self.symbol.replace('USDT', '')}")
            logger.info(f"   💵 Capital en riesgo: ${rr_metrics['risk_usdt']:.2f}")
            logger.info(f"   💎 Ganancia potencial: ${rr_metrics['reward_usdt']:.2f}")
            logger.info(f"   ⚖️ Ratio Riesgo/Recompensa: 1:{rr_metrics['risk_reward_ratio']:.2f}")

            # Colocar órdenes de protección
            self.place_stop_order('BUY', self.quantity, stop_price)
            self.place_take_profit_order('BUY', self.quantity, tp_price)

            self.position = {
                'side': 'SHORT',
                'entry_price': current_price,
                'quantity': self.quantity,
                'stop_price': stop_price,
                'tp_price': tp_price,
                'entry_time': datetime.now(),
                'risk_reward_metrics': rr_metrics
            }

            LoggingHelper.log_section_header("POSICIÓN SHORT ESTABLECIDA EXITOSAMENTE", "✅")

    def log_position_status(self, current_price: float):
        """Log del estado actual de la posición con objetivos"""
        if not self.position:
            return

        entry_price = self.position['entry_price']
        stop_price = self.position['stop_price']
        tp_price = self.position['tp_price']
        side = self.position['side']

        # Calcular PnL actual
        if side == 'LONG':
            unrealized_pnl = (current_price - entry_price) * self.quantity
            distance_to_sl = ((current_price - stop_price) / stop_price) * 100
            distance_to_tp = ((tp_price - current_price) / current_price) * 100
        else:  # SHORT
            unrealized_pnl = (entry_price - current_price) * self.quantity
            distance_to_sl = ((stop_price - current_price) / current_price) * 100
            distance_to_tp = ((current_price - tp_price) / current_price) * 100

        # Tiempo en posición
        time_in_position = datetime.now() - self.position['entry_time']

        LoggingHelper.log_subsection_header("ESTADO DE POSICIÓN ACTIVA")
        logger.info(f"   🔄 Dirección: {side}")
        logger.info(f"   💰 Precio entrada: ${entry_price:.4f}")
        logger.info(f"   📈 Precio actual: ${current_price:.4f}")
        logger.info(f"   💵 PnL no realizado: ${unrealized_pnl:.2f} USDT")
        logger.info(f"   🛡️ Stop Loss: ${stop_price:.4f} ({distance_to_sl:+.2f}%)")
        logger.info(f"   🎯 Take Profit: ${tp_price:.4f} ({distance_to_tp:+.2f}%)")
        logger.info(f"   ⏱️ Tiempo en posición: {str(time_in_position).split('.')[0]}")
        logger.info("-" * 60)

    def handle_socket_message(self, msg):
        """Maneja mensajes del WebSocket de klines"""
        try:
            # Solo procesar klines cerradas
            if msg['k']['x']:  # kline is closed
                self.process_new_kline(msg['k'])

        except Exception as e:
            LoggingHelper.log_action_error("PROCESAMIENTO DE MENSAJE WEBSOCKET", str(e))

    def process_new_kline(self, kline_data: Dict):
        """Procesa una nueva kline y verifica señales"""
        LoggingHelper.log_action_start("PROCESAMIENTO DE NUEVA VELA")

        try:
            # Obtener datos históricos actualizados
            df = self.get_historical_data(limit=800)  # Suficientes datos para indicadores

            if df.empty:
                logger.info("⚠️ No se pudieron obtener datos históricos")
                return

            # Calcular indicadores
            df = self.strategy.calculate_indicators(df)

            # Detectar señales
            buy_signal, sell_signal = self.strategy.detect_pullback_signals(df)

            current_price = df['close'].iloc[-1]

            # Log de estado actual del mercado
            current_trend = df['trend_direction'].iloc[-1]
            trend_text = "🟢 ALCISTA" if current_trend == 1 else "🔴 BAJISTA" if current_trend == -1 else "🟡 NEUTRAL"

            # Usar el helper para logging estructurado del mercado
            LoggingHelper.log_market_status(
                price=current_price,
                trend=trend_text,
                pac_price=df['pac_close'].iloc[-1],
                buy_signal=buy_signal,
                sell_signal=sell_signal
            )

            # Mostrar estado de posición si existe
            if self.position:
                self.log_position_status(current_price)

            # Ejecutar señales
            if buy_signal:
                LoggingHelper.log_section_header("🚨 SEÑAL DE COMPRA DETECTADA 🚨", "🚀")
                self.execute_buy_signal(current_price)
            elif sell_signal:
                LoggingHelper.log_section_header("🚨 SEÑAL DE VENTA DETECTADA 🚨", "🔻")
                self.execute_sell_signal(current_price)

            # Verificar estado de posición actual
            current_position = self.get_current_position()
            if current_position is None and self.position is not None:
                # La posición se cerró, determinar si fue por SL o TP
                self.log_position_closed(current_price)
                self.position = None
                self.active_stop_order_id = None
                self.active_tp_order_id = None

            LoggingHelper.log_action_success("PROCESAMIENTO DE NUEVA VELA", "Análisis completado")

        except Exception as e:
            LoggingHelper.log_action_error("PROCESAMIENTO DE NUEVA VELA", str(e))

    def log_position_closed(self, exit_price: float):
        """Log cuando se cierra una posición"""
        if not self.position:
            return

        entry_price = self.position['entry_price']
        stop_price = self.position['stop_price']
        tp_price = self.position['tp_price']
        side = self.position['side']

        # Determinar si se cerró por SL o TP
        if side == 'LONG':
            closed_by_sl = exit_price <= stop_price * 1.001  # Pequeño margen de tolerancia
            closed_by_tp = exit_price >= tp_price * 0.999
            pnl = (exit_price - entry_price) * self.quantity
        else:  # SHORT
            closed_by_sl = exit_price >= stop_price * 0.999
            closed_by_tp = exit_price <= tp_price * 1.001
            pnl = (entry_price - exit_price) * self.quantity

        time_in_position = datetime.now() - self.position['entry_time']

        # Log estructurado del cierre de posición
        if closed_by_tp:
            LoggingHelper.log_section_header("🎉 POSICIÓN CERRADA POR TAKE PROFIT 🎉", "✅")
        elif closed_by_sl:
            LoggingHelper.log_section_header("🛡️ POSICIÓN CERRADA POR STOP LOSS", "⚠️")
        else:
            LoggingHelper.log_section_header("🔄 POSICIÓN CERRADA", "ℹ️")

        LoggingHelper.log_subsection_header("RESUMEN DE OPERACIÓN")
        logger.info(f"   🔄 Dirección: {side}")
        logger.info(f"   💰 Precio entrada: ${entry_price:.4f}")
        logger.info(f"   🚪 Precio salida: ${exit_price:.4f}")
        logger.info(f"   💵 PnL realizado: ${pnl:.2f} USDT")
        logger.info(f"   📈 Rendimiento: {((pnl / (entry_price * self.quantity)) * 100):+.2f}%")
        logger.info(f"   ⏱️ Duración: {str(time_in_position).split('.')[0]}")

        # Estadísticas adicionales
        rr_metrics = self.position.get('risk_reward_metrics', {})
        if rr_metrics:
            logger.info(f"   ⚖️ Ratio R/R planeado: 1:{rr_metrics.get('risk_reward_ratio', 0):.2f}")
            actual_rr = abs(pnl) / rr_metrics.get('risk_usdt', 1) if rr_metrics.get('risk_usdt', 0) > 0 else 0
            logger.info(f"   📊 Ratio R/R real: 1:{actual_rr:.2f}")

        LoggingHelper.log_section_header("OPERACIÓN FINALIZADA", "🏁")

    def start_websocket(self):
        """Inicia el WebSocket usando ThreadedWebSocketManager"""
        LoggingHelper.log_action_start("INICIALIZACIÓN DE WEBSOCKET")

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

            LoggingHelper.log_action_success("INICIALIZACIÓN DE WEBSOCKET",
                                             f"Conectado a {self.symbol} en timeframe {self.timeframe}")

        except Exception as e:
            LoggingHelper.log_action_error("INICIALIZACIÓN DE WEBSOCKET", str(e))

    def run(self):
        """Inicia el bot de scalping"""
        LoggingHelper.log_section_header(f"🤖 BOT DE SCALPING INICIADO - {self.bot_name}", "🚀")

        # Log de configuración inicial
        LoggingHelper.log_subsection_header("CONFIGURACIÓN DEL BOT")
        logger.info(f"   📍 Par de trading: {self.symbol}")
        logger.info(f"   ⚡ Timeframe: {self.timeframe}")
        logger.info(f"   📊 Apalancamiento: {self.leverage}x")
        logger.info(f"   🛡️ Stop Loss: {self.stop_loss_pct}%")
        logger.info(f"   🎯 Take Profit: {self.take_profit_pct}%")
        logger.info(f"   💰 Cantidad por trade: {self.quantity} {self.symbol.replace('USDT', '')}")
        logger.info(f"   🌐 Modo: {'TESTNET' if self.testnet else 'MAINNET'}")

        LoggingHelper.log_subsection_header("CONFIGURACIÓN DE ESTRATEGIA")
        logger.info(f"   📈 EMA Rápida: {self.strategy.fast_ema}")
        logger.info(f"   📊 EMA Media: {self.strategy.medium_ema}")
        logger.info(f"   📉 EMA Lenta: {self.strategy.slow_ema}")
        logger.info(f"   🎯 PAC Length: {self.strategy.pac_length}")
        logger.info(f"   🔍 Lookback: {self.strategy.lookback}")

        self.is_running = True

        # Inicializar datos históricos
        df = self.get_historical_data()
        if df.empty:
            LoggingHelper.log_action_error("INICIALIZACIÓN DE DATOS", "No se pudieron obtener datos históricos")
            return

        # Iniciar WebSocket
        self.start_websocket()

        LoggingHelper.log_section_header("BOT EN FUNCIONAMIENTO - MONITOREANDO MERCADO", "👁️")

        try:
            while self.is_running:
                # Verificar conexión y estado cada minuto
                time.sleep(60)

                LoggingHelper.log_subsection_header("MONITOREO POSICIONES ABIERTAS")

                # Verificar posición actual desde la API
                current_pos = self.get_current_position()
                if current_pos:
                    # Usar 'unRealizedProfit' en lugar de 'unrealizedPnl'
                    pnl_key = 'unRealizedProfit' if 'unRealizedProfit' in current_pos else 'unrealizedPnl'
                    if pnl_key in current_pos:
                        pnl = float(current_pos[pnl_key])
                        position_amt = float(current_pos['positionAmt'])
                        entry_price = float(current_pos['entryPrice'])

                        # Determinar dirección de la posición
                        position_direction = "SHORT" if position_amt < 0 else "LONG"
                        position_size = abs(position_amt)

                        logger.info(f"   💼 Posición {position_direction}: {position_size} SOL")
                        logger.info(f"   💰 Precio entrada: ${entry_price:.4f}")
                        logger.info(f"   💵 PnL: ${pnl:.4f} USDT")
                        logger.info(f"   📊 Modo: {current_pos['positionSide']}")

                        # Mostrar información adicional de SL/TP si hay posición registrada
                        if self.position:
                            logger.info(f"   🛡️ Stop Loss objetivo: ${self.position['stop_price']:.4f}")
                            logger.info(f"   🎯 Take Profit objetivo: ${self.position['tp_price']:.4f}")
                    else:
                        logger.info(f"   💼 Posición detectada pero sin PnL disponible")
                else:
                    logger.info("   📊 Sin posiciones activas")

                logger.info("-" * 60)

        except KeyboardInterrupt:
            LoggingHelper.log_section_header("⏹️ BOT DETENIDO POR EL USUARIO", "🛑")
        except Exception as e:
            LoggingHelper.log_action_error("EJECUCIÓN DEL BOT", str(e))
        finally:
            self.stop()

    def stop(self):
        """Detiene el bot"""
        LoggingHelper.log_section_header("⏹️ DETENIENDO BOT DE SCALPING", "🛑")

        self.is_running = False
        if self.twm:
            try:
                self.twm.stop()
                LoggingHelper.log_action_success("DESCONEXIÓN DE WEBSOCKET", "WebSocket cerrado correctamente")
            except Exception as e:
                LoggingHelper.log_action_error("DESCONEXIÓN DE WEBSOCKET", str(e))

        # Mostrar resumen si hay posición activa
        if self.position:
            LoggingHelper.log_subsection_header("⚠️ ATENCIÓN: POSICIÓN ACTIVA AL DETENER BOT")
            logger.info(f"   📊 Posición: {self.position['side']}")
            logger.info(f"   💰 Precio entrada: ${self.position['entry_price']:.4f}")
            logger.info(f"   🛡️ Stop Loss: ${self.position['stop_price']:.4f}")
            logger.info(f"   🎯 Take Profit: ${self.position['tp_price']:.4f}")
            logger.info(f"   ⚠️ Las órdenes de SL/TP permanecen activas")
            logger.info("-" * 60)

        LoggingHelper.log_section_header("✅ BOT DETENIDO EXITOSAMENTE", "🏁")


# Configuración y ejecución
if __name__ == "__main__":
    # IMPORTANTE: Reemplaza con tus credenciales reales
    API_KEY = 'bPT08S5D4JijR0qJDAZh2ILB35yjQ1s5ozHxmVcY0qGL4eywVeVHPy1PjfgCkkwK'
    API_SECRET = '2xIUXyeIA82A367HPrHXklozZl6DafBQJjT9sVfsv6TfdOoxMCt7LTKRc2BPsagj'

    # Usar testnet para pruebas (cambiar a False para trading real)
    USE_TESTNET = False

    LoggingHelper.log_section_header("INICIANDO PROGRAMA DE TRADING BOT", "🚀")

    try:
        # Crear y ejecutar el bot
        bot = BinanceScalpingBot(
            api_key=API_KEY,
            api_secret=API_SECRET,
            testnet=USE_TESTNET
        )

        # Configuraciones adicionales del bot (opcional)
        # bot.quantity = 0.1  # Ajustar según tu capital
        # bot.leverage = 10  # Apalancamiento
        # bot.stop_loss_pct = 0.5  # 0.5% stop loss
        # bot.take_profit_pct = 1.0  # 1% take profit

        # Ejecutar el bot
        bot.run()

    except Exception as e:
        LoggingHelper.log_action_error("EJECUCIÓN PRINCIPAL DEL PROGRAMA", str(e))
    finally:
        LoggingHelper.log_section_header("🔚 PROGRAMA FINALIZADO", "🏁")