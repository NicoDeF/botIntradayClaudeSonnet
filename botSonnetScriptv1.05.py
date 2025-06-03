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

        ##################################################
        #
        # Configuración de trading
        #
        ##################################################

        self.symbol = 'SOLUSDT'
        self.timeframe = '1m'  # 1 minuto para scalping
        self.quantity = 0.2 # Cantidad por trade (ajustar según capital)
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
        try:
            self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
            logger.info(f"🔧 Apalancamiento configurado a {self.leverage}x para {self.symbol}")
        except Exception as e:
            logger.error(f"❌ Error configurando apalancamiento: {e}")

    def get_symbol_info(self):
        """Obtiene información del símbolo para configurar precisión"""
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
            logger.info(f"📊 Precisión configurada - Precio: {self.price_precision}, Cantidad: {self.qty_precision}")
        except Exception as e:
            logger.error(f"❌ Error obteniendo información del símbolo: {e}")

    def round_price(self, price: float) -> float:
        """Redondea el precio según la precisión del símbolo"""
        return round(price, self.price_precision)

    def round_quantity(self, quantity: float) -> float:
        """Redondea la cantidad según la precisión del símbolo"""
        return round(quantity, self.qty_precision)

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
            logger.error(f"❌ Error obteniendo datos históricos: {e}")
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
        if order:
            order_id = order.get('orderId', 'N/A')
            side = order.get('side', 'N/A')
            quantity = order.get('origQty', order.get('quantity', 'N/A'))
            price = order.get('price', order.get('stopPrice', 'N/A'))

            logger.info(
                f"📋 {order_type} | ID: {order_id} | {side} {quantity} {self.symbol} | Precio: {price} | {additional_info}")

    def place_market_order(self, side: str, quantity: float) -> Optional[Dict]:
        """Coloca una orden de mercado"""
        try:
            quantity = self.round_quantity(quantity)
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            self.log_order_details("ORDEN MERCADO EJECUTADA", order)
            return order
        except BinanceAPIException as e:
            logger.error(f"❌ Error en orden {side}: {e}")
            return None

    def place_stop_order(self, side: str, quantity: float, stop_price: float) -> Optional[Dict]:
        """Coloca una orden stop loss"""
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
                self.log_order_details("STOP LOSS COLOCADO", order, f"🛡️ Protección activada")

            return order
        except BinanceAPIException as e:
            logger.error(f"❌ Error en stop loss: {e}")
            return None

    def place_take_profit_order(self, side: str, quantity: float, price: float) -> Optional[Dict]:
        """Coloca una orden take profit"""
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
                self.log_order_details("TAKE PROFIT COLOCADO", order, f"🎯 Objetivo establecido")

            return order
        except BinanceAPIException as e:
            logger.error(f"❌ Error en take profit: {e}")
            return None

    def calculate_risk_reward(self, entry_price: float, stop_price: float, tp_price: float, side: str) -> Dict:
        """Calcula el riesgo/recompensa de la operación"""
        if side == 'LONG':
            risk = abs(entry_price - stop_price)
            reward = abs(tp_price - entry_price)
        else:  # SHORT
            risk = abs(stop_price - entry_price)
            reward = abs(entry_price - tp_price)

        risk_reward_ratio = reward / risk if risk > 0 else 0
        risk_usdt = risk * self.quantity
        reward_usdt = reward * self.quantity

        return {
            'risk': risk,
            'reward': reward,
            'risk_reward_ratio': risk_reward_ratio,
            'risk_usdt': risk_usdt,
            'reward_usdt': reward_usdt
        }

    def execute_buy_signal(self, current_price: float):
        """Ejecuta señal de compra"""
        if self.position is not None:
            logger.info("⚠️ Ya hay una posición abierta, ignorando señal de compra")
            return

        logger.info("=" * 60)
        logger.info("🚀 EJECUTANDO SEÑAL DE COMPRA (LONG)")
        logger.info("=" * 60)

        # Abrir posición long
        order = self.place_market_order('BUY', self.quantity)
        if order:
            # Calcular precios de stop loss y take profit
            stop_price = self.round_price(current_price * (1 - self.stop_loss_pct / 100))
            tp_price = self.round_price(current_price * (1 + self.take_profit_pct / 100))

            # Calcular métricas de riesgo/recompensa
            rr_metrics = self.calculate_risk_reward(current_price, stop_price, tp_price, 'LONG')

            # Log detallado de la operación
            logger.info(f"📈 POSICIÓN LONG ABIERTA")
            logger.info(f"💰 Precio de entrada: ${current_price:.4f}")
            logger.info(f"🛡️ Stop Loss: ${stop_price:.4f} (-{self.stop_loss_pct}%)")
            logger.info(f"🎯 Take Profit: ${tp_price:.4f} (+{self.take_profit_pct}%)")
            logger.info(f"📊 Cantidad: {self.quantity} {self.symbol.replace('USDT', '')}")
            logger.info(f"💵 Capital en riesgo: ${rr_metrics['risk_usdt']:.2f}")
            logger.info(f"💎 Ganancia potencial: ${rr_metrics['reward_usdt']:.2f}")
            logger.info(f"⚖️ Ratio Riesgo/Recompensa: 1:{rr_metrics['risk_reward_ratio']:.2f}")

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

            logger.info("✅ Órdenes de protección colocadas exitosamente")
            logger.info("=" * 60)

    def execute_sell_signal(self, current_price: float):
        """Ejecuta señal de venta"""
        if self.position is not None:
            logger.info("⚠️ Ya hay una posición abierta, ignorando señal de venta")
            return

        logger.info("=" * 60)
        logger.info("🔻 EJECUTANDO SEÑAL DE VENTA (SHORT)")
        logger.info("=" * 60)

        # Abrir posición short
        order = self.place_market_order('SELL', self.quantity)
        if order:
            # Calcular precios de stop loss y take profit
            stop_price = self.round_price(current_price * (1 + self.stop_loss_pct / 100))
            tp_price = self.round_price(current_price * (1 - self.take_profit_pct / 100))

            # Calcular métricas de riesgo/recompensa
            rr_metrics = self.calculate_risk_reward(current_price, stop_price, tp_price, 'SHORT')

            # Log detallado de la operación
            logger.info(f"📉 POSICIÓN SHORT ABIERTA")
            logger.info(f"💰 Precio de entrada: ${current_price:.4f}")
            logger.info(f"🛡️ Stop Loss: ${stop_price:.4f} (+{self.stop_loss_pct}%)")
            logger.info(f"🎯 Take Profit: ${tp_price:.4f} (-{self.take_profit_pct}%)")
            logger.info(f"📊 Cantidad: {self.quantity} {self.symbol.replace('USDT', '')}")
            logger.info(f"💵 Capital en riesgo: ${rr_metrics['risk_usdt']:.2f}")
            logger.info(f"💎 Ganancia potencial: ${rr_metrics['reward_usdt']:.2f}")
            logger.info(f"⚖️ Ratio Riesgo/Recompensa: 1:{rr_metrics['risk_reward_ratio']:.2f}")

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

            logger.info("✅ Órdenes de protección colocadas exitosamente")
            logger.info("=" * 60)

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

        logger.info("📊 ESTADO DE POSICIÓN ACTIVA:")
        logger.info(f"   🔄 Dirección: {side}")
        logger.info(f"   💰 Precio entrada: ${entry_price:.4f}")
        logger.info(f"   📈 Precio actual: ${current_price:.4f}")
        logger.info(f"   💵 PnL no realizado: ${unrealized_pnl:.2f} USDT")
        logger.info(f"   🛡️ Stop Loss: ${stop_price:.4f} ({distance_to_sl:+.2f}%)")
        logger.info(f"   🎯 Take Profit: ${tp_price:.4f} ({distance_to_tp:+.2f}%)")
        logger.info(f"   ⏱️ Tiempo en posición: {str(time_in_position).split('.')[0]}")

    def handle_socket_message(self, msg):
        """Maneja mensajes del WebSocket de klines"""
        try:
            # Solo procesar klines cerradas
            if msg['k']['x']:  # kline is closed
                self.process_new_kline(msg['k'])

        except Exception as e:
            logger.error(f"❌ Error procesando mensaje WebSocket: {e}")

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

            # Log de estado actual del mercado
            current_trend = df['trend_direction'].iloc[-1]
            trend_text = "🟢 ALCISTA" if current_trend == 1 else "🔴 BAJISTA" if current_trend == -1 else "🟡 NEUTRAL"

            logger.info(f"📊 Precio: ${current_price:.4f} | Trend: {trend_text} | "
                        f"PAC: ${df['pac_close'].iloc[-1]:.4f} | "
                        f"Señales - Buy: {'✅' if buy_signal else '❌'} | Sell: {'✅' if sell_signal else '❌'}")

            # Mostrar estado de posición si existe
            if self.position:
                self.log_position_status(current_price)

            # Ejecutar señales
            if buy_signal:
                logger.info("🚨 SEÑAL DE COMPRA DETECTADA 🚨")
                self.execute_buy_signal(current_price)
            elif sell_signal:
                logger.info("🚨 SEÑAL DE VENTA DETECTADA 🚨")
                self.execute_sell_signal(current_price)

            # Verificar estado de posición actual
            current_position = self.get_current_position()
            if current_position is None and self.position is not None:
                # La posición se cerró, determinar si fue por SL o TP
                self.log_position_closed(current_price)
                self.position = None
                self.active_stop_order_id = None
                self.active_tp_order_id = None

        except Exception as e:
            logger.error(f"❌ Error procesando nueva kline: {e}")

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

        logger.info("=" * 60)
        if closed_by_tp:
            logger.info("🎉 POSICIÓN CERRADA POR TAKE PROFIT 🎉")
        elif closed_by_sl:
            logger.info("🛡️ POSICIÓN CERRADA POR STOP LOSS")
        else:
            logger.info("🔄 POSICIÓN CERRADA")

        logger.info(f"📊 Resumen de operación:")
        logger.info(f"   🔄 Dirección: {side}")
        logger.info(f"   💰 Precio entrada: ${entry_price:.4f}")
        logger.info(f"   🚪 Precio salida: ${exit_price:.4f}")
        logger.info(f"   💵 PnL realizado: ${pnl:.2f} USDT")
        logger.info(f"   📈 Rendimiento: {((pnl / (entry_price * self.quantity)) * 100):+.2f}%")
        logger.info(f"   ⏱️ Duración: {str(time_in_position).split('.')[0]}")
        logger.info("=" * 60)

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

            logger.info("🌐 WebSocket iniciado correctamente")

        except Exception as e:
            logger.error(f"❌ Error iniciando WebSocket: {e}")

    def run(self):
        """Inicia el bot de scalping"""
        logger.info("=" * 80)
        logger.info(f"🤖 BOT DE SCALPING INICIADO - {self.bot_name}")
        logger.info("=" * 80)
        logger.info(f"📍 Par de trading: {self.symbol}")
        logger.info(f"⚡ Timeframe: {self.timeframe}")
        logger.info(f"📊 Apalancamiento: {self.leverage}x")
        logger.info(f"🛡️ Stop Loss: {self.stop_loss_pct}%")
        logger.info(f"🎯 Take Profit: {self.take_profit_pct}%")
        logger.info(f"💰 Cantidad por trade: {self.quantity} {self.symbol.replace('USDT', '')}")
        logger.info(f"🌐 Modo: {'TESTNET' if self.testnet else 'MAINNET'}")
        logger.info("=" * 80)

        self.is_running = True

        # Inicializar datos históricos
        df = self.get_historical_data()
        if df.empty:
            logger.error("❌ No se pudieron obtener datos históricos")
            return

        # Iniciar WebSocket
        self.start_websocket()

        try:
            while self.is_running:
                # Verificar conexión y estado cada minuto
                time.sleep(60)

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

                        logger.info(f"💼 Posición {position_direction}: {position_size} SOL | "
                                    f"💰 Precio entrada: ${entry_price:.4f} | "
                                    f"💵 PnL: ${pnl:.4f} USDT | "
                                    f"📊 Modo: {current_pos['positionSide']}")

                        # Mostrar información adicional de SL/TP si hay posición registrada
                        if self.position:
                            logger.info(f"🛡️ Stop Loss objetivo: ${self.position['stop_price']:.4f}")
                            logger.info(f"🎯 Take Profit objetivo: ${self.position['tp_price']:.4f}")
                    else:
                        logger.info(f"💼 Posición detectada pero sin PnL disponible: {current_pos}")
                else:
                    logger.info("📊 Sin posiciones activas")

        except KeyboardInterrupt:
            logger.info("⏹️ Bot detenido por el usuario")
        except Exception as e:
            logger.error(f"❌ Error durante la ejecución: {e}")
        finally:
            self.stop()

    def stop(self):
        """Detiene el bot"""
        logger.info("=" * 60)
        logger.info("⏹️ DETENIENDO BOT DE SCALPING")
        logger.info("=" * 60)

        self.is_running = False
        if self.twm:
            try:
                self.twm.stop()
                logger.info("🌐 WebSocket desconectado")
            except Exception as e:
                logger.error(f"❌ Error deteniendo WebSocket: {e}")

        # Mostrar resumen si hay posición activa
        if self.position:
            logger.info("⚠️ ATENCIÓN: Hay una posición activa al detener el bot")
            logger.info(f"   📊 Posición: {self.position['side']}")
            logger.info(f"   💰 Precio entrada: ${self.position['entry_price']:.4f}")
            logger.info(f"   🛡️ Stop Loss: ${self.position['stop_price']:.4f}")
            logger.info(f"   🎯 Take Profit: ${self.position['tp_price']:.4f}")
            logger.info("   ⚠️ Las órdenes de SL/TP permanecen activas")

        logger.info("✅ Bot detenido exitosamente")
        logger.info("=" * 60)


# Configuración y ejecución
if __name__ == "__main__":
    # IMPORTANTE: Reemplaza con tus credenciales reales
    API_KEY = 'bPT08S5D4JijR0qJDAZh2ILB35yjQ1s5ozHxmVcY0qGL4eywVeVHPy1PjfgCkkwK'
    API_SECRET = '2xIUXyeIA82A367HPrHXklozZl6DafBQJjT9sVfsv6TfdOoxMCt7LTKRc2BPsagj'

    # Usar testnet para pruebas (cambiar a False para trading real)
    USE_TESTNET = False

    try:
        # Crear y ejecutar el bot
        bot = BinanceScalpingBot(
            api_key=API_KEY,
            api_secret=API_SECRET,
            testnet=USE_TESTNET
        )

        # Configuraciones adicionales del bot
        # bot.quantity = 0.1  # Ajustar según tu capital
        # bot.leverage = 10  # Apalancamiento
        # bot.stop_loss_pct = 0.5  # 0.5% stop loss
        # bot.take_profit_pct = 1.0  # 1% take profit
        # Ejecutar el bot
        bot.run()

    except Exception as e:
        logger.error(f"❌ Error crítico en la ejecución del bot: {e}")
    finally:
        logger.info("🔚 Programa finalizado")