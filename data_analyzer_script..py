import pandas as pd
import numpy as np
import talib
from binance.client import Client
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class TradingDataAnalyzer:
    """
    Analizador de datos de trading para entender el DataFrame y las señales
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        # Configuración del cliente
        if testnet:
            self.client = Client(api_key, api_secret, testnet=True)
            self.client.API_URL = 'https://testnet.binancefuture.com/fapi'
        else:
            self.client = Client(api_key, api_secret)

        # Configuración de estrategia (igual que el bot)
        self.symbol = 'SOLUSDT'
        self.timeframe = '1m'
        self.pac_length = 34
        self.fast_ema = 89
        self.medium_ema = 200
        self.slow_ema = 600
        self.lookback = 3

    def get_historical_data(self, limit: int = 1000) -> pd.DataFrame:
        """
        Obtiene datos históricos de Binance y explica cada columna
        """
        print("🔄 Obteniendo datos históricos de Binance...")
        print(f"   📍 Símbolo: {self.symbol}")
        print(f"   ⏰ Timeframe: {self.timeframe}")
        print(f"   📊 Velas: {limit}")
        print()

        try:
            # Obtener datos de Binance
            klines = self.client.futures_klines(
                symbol=self.symbol,
                interval=self.timeframe,
                limit=limit
            )

            # Crear DataFrame con todas las columnas que retorna Binance
            df = pd.DataFrame(klines, columns=[
                'timestamp',  # Timestamp de apertura
                'open',  # Precio de apertura
                'high',  # Precio máximo
                'low',  # Precio mínimo
                'close',  # Precio de cierre
                'volume',  # Volumen en cantidad base
                'close_time',  # Timestamp de cierre
                'quote_asset_volume',  # Volumen en quote asset (USDT)
                'number_of_trades',  # Número de trades
                'taker_buy_base_asset_volume',  # Volumen de compra taker (base)
                'taker_buy_quote_asset_volume',  # Volumen de compra taker (quote)
                'ignore'  # Campo ignorado
            ])

            # Convertir a tipos numéricos las columnas que usamos
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])

            # Convertir timestamp a datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            print("✅ Datos obtenidos exitosamente!")
            print(f"   📈 Rango: {df.index[0]} a {df.index[-1]}")
            print(f"   💰 Precio actual: ${df['close'].iloc[-1]:.4f}")
            print()

            # Retornar solo las columnas que usamos en el trading
            return df[numeric_columns]

        except Exception as e:
            print(f"❌ Error obteniendo datos: {e}")
            return pd.DataFrame()

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula TODOS los indicadores que usa el bot y explica cada uno
        """
        print("🧮 Calculando indicadores técnicos...")

        # 1. EMAs (Exponential Moving Averages)
        print(f"   📈 EMA Rápida ({self.fast_ema} períodos)")
        df['fast_ema'] = talib.EMA(df['close'], timeperiod=self.fast_ema)

        print(f"   📊 EMA Media ({self.medium_ema} períodos)")
        df['medium_ema'] = talib.EMA(df['close'], timeperiod=self.medium_ema)

        print(f"   📉 EMA Lenta ({self.slow_ema} períodos)")
        df['slow_ema'] = talib.EMA(df['close'], timeperiod=self.slow_ema)

        # 2. PAC Channel (Price Action Channel)
        print(f"   🎯 PAC Channel ({self.pac_length} períodos)")
        print("      • PAC Close: EMA del precio de cierre")
        df['pac_close'] = talib.EMA(df['close'], timeperiod=self.pac_length)

        print("      • PAC High: EMA del precio máximo")
        df['pac_high'] = talib.EMA(df['high'], timeperiod=self.pac_length)

        print("      • PAC Low: EMA del precio mínimo")
        df['pac_low'] = talib.EMA(df['low'], timeperiod=self.pac_length)

        # 3. Trend Direction (Dirección de tendencia)
        print("   🔄 Calculando dirección de tendencia...")
        print("      • Alcista: fast_ema > medium_ema Y pac_low > medium_ema")
        print("      • Bajista: fast_ema < medium_ema Y pac_high < medium_ema")
        print("      • Neutral: Otras combinaciones")

        df['trend_direction'] = np.where(
            (df['fast_ema'] > df['medium_ema']) & (df['pac_low'] > df['medium_ema']), 1,
            np.where(
                (df['fast_ema'] < df['medium_ema']) & (df['pac_high'] < df['medium_ema']), -1, 0
            )
        )

        # 4. Bar Colors (Posición relativa al PAC)
        print("   🎨 Calculando colores de velas...")
        print("      • Azul: Precio > PAC High (arriba del canal)")
        print("      • Rojo: Precio < PAC Low (abajo del canal)")
        print("      • Gris: Precio dentro del PAC")

        df['bar_color'] = np.where(
            df['close'] > df['pac_high'], 'blue',
            np.where(df['close'] < df['pac_low'], 'red', 'gray')
        )

        # 5. Señales de Trading
        print("   🚨 Detectando señales de trading...")
        df['buy_signal'] = False
        df['sell_signal'] = False

        for i in range(len(df)):
            if i < max(self.fast_ema, self.medium_ema, self.pac_length) + self.lookback:
                continue

            current = df.iloc[i]

            # Buy Signal Logic
            buy_conditions = []
            buy_conditions.append(current['trend_direction'] == 1)

            # Verificar pullback
            recent_below_pac = False
            for j in range(1, self.lookback + 1):
                if i - j >= 0 and df.iloc[i - j]['close'] < df.iloc[i - j]['pac_close']:
                    recent_below_pac = True
                    break

            pac_exit_up = (current['open'] < current['pac_high'] < current['close'] and recent_below_pac)
            buy_conditions.append(pac_exit_up)

            df.iloc[i, df.columns.get_loc('buy_signal')] = all(buy_conditions)

            # Sell Signal Logic
            sell_conditions = []
            sell_conditions.append(current['trend_direction'] == -1)

            recent_above_pac = False
            for j in range(1, self.lookback + 1):
                if i - j >= 0 and df.iloc[i - j]['close'] > df.iloc[i - j]['pac_close']:
                    recent_above_pac = True
                    break

            pac_exit_down = (current['open'] > current['pac_low'] and
                             current['close'] < current['pac_low'] and recent_above_pac)
            sell_conditions.append(pac_exit_down)

            df.iloc[i, df.columns.get_loc('sell_signal')] = all(sell_conditions)

        print("✅ Todos los indicadores calculados!")
        print()

        return df

    def analyze_signals(self, df: pd.DataFrame):
        """
        Analiza las señales generadas
        """
        print("🔍 ANÁLISIS DE SEÑALES GENERADAS:")
        print("=" * 50)

        # Estadísticas generales
        total_candles = len(df)
        buy_signals = df['buy_signal'].sum()
        sell_signals = df['sell_signal'].sum()

        print(f"📊 Total de velas analizadas: {total_candles}")
        print(f"🚀 Señales de compra: {buy_signals}")
        print(f"🔻 Señales de venta: {sell_signals}")
        print(f"📈 Frecuencia compra: {(buy_signals / total_candles * 100):.2f}%")
        print(f"📉 Frecuencia venta: {(sell_signals / total_candles * 100):.2f}%")
        print()

        # Últimas señales
        recent_data = df.tail(50)  # Últimas 50 velas
        recent_buys = recent_data[recent_data['buy_signal'] == True]
        recent_sells = recent_data[recent_data['sell_signal'] == True]

        print("🕒 SEÑALES RECIENTES (Últimas 50 velas):")
        print(f"🚀 Compras: {len(recent_buys)}")
        if len(recent_buys) > 0:
            last_buy = recent_buys.index[-1]
            last_buy_price = recent_buys['close'].iloc[-1]
            print(f"   Última compra: {last_buy} @ ${last_buy_price:.4f}")

        print(f"🔻 Ventas: {len(recent_sells)}")
        if len(recent_sells) > 0:
            last_sell = recent_sells.index[-1]
            last_sell_price = recent_sells['close'].iloc[-1]
            print(f"   Última venta: {last_sell} @ ${last_sell_price:.4f}")
        print()

        # Estado actual
        current = df.iloc[-1]
        trend_text = "🟢 Alcista" if current['trend_direction'] == 1 else "🔴 Bajista" if current[
                                                                                            'trend_direction'] == -1 else "🟡 Neutral"

        print("📊 ESTADO ACTUAL DEL MERCADO:")
        print(f"💰 Precio: ${current['close']:.4f}")
        print(f"📈 Tendencia: {trend_text}")
        print(f"🎯 PAC Center: ${current['pac_close']:.4f}")
        print(f"🔺 PAC High: ${current['pac_high']:.4f}")
        print(f"🔻 PAC Low: ${current['pac_low']:.4f}")
        print(f"🎨 Color vela: {current['bar_color']}")
        print(
            f"🚨 Señal actual: {'🚀 COMPRA' if current['buy_signal'] else '🔻 VENTA' if current['sell_signal'] else '⏸️ Ninguna'}")
        print()

    def export_to_excel(self, df: pd.DataFrame, filename: str = None):
        """
        Exporta todos los datos a Excel con formato
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_analysis_{self.symbol}_{timestamp}.xlsx"

        print(f"📁 Exportando datos a Excel: {filename}")

        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Hoja 1: Datos completos
                df_export = df.copy()
                df_export.reset_index(inplace=True)
                df_export.to_excel(writer, sheet_name='Datos_Completos', index=False)

                # Hoja 2: Solo señales
                signals_df = df[df['buy_signal'] | df['sell_signal']].copy()
                signals_df['signal_type'] = signals_df.apply(
                    lambda x: 'BUY' if x['buy_signal'] else 'SELL', axis=1
                )
                signals_df = signals_df[['close', 'signal_type', 'trend_direction', 'pac_close', 'pac_high', 'pac_low']]
                signals_df.reset_index(inplace=True)
                signals_df.to_excel(writer, sheet_name='Señales', index=False)

                # Hoja 3: Estadísticas
                stats_data = {
                    'Métrica': [
                        'Total Velas',
                        'Señales Compra',
                        'Señales Venta',
                        'Frecuencia Compra (%)',
                        'Frecuencia Venta (%)',
                        'Precio Actual',
                        'PAC Center Actual',
                        'Tendencia Actual',
                        'EMA Rápida Actual',
                        'EMA Media Actual',
                        'EMA Lenta Actual'
                    ],
                    'Valor': [
                        len(df),
                        df['buy_signal'].sum(),
                        df['sell_signal'].sum(),
                        round(df['buy_signal'].sum() / len(df) * 100, 2),
                        round(df['sell_signal'].sum() / len(df) * 100, 2),
                        round(df['close'].iloc[-1], 4),
                        round(df['pac_close'].iloc[-1], 4),
                        'Alcista' if df['trend_direction'].iloc[-1] == 1 else 'Bajista' if df['trend_direction'].iloc[
                                                                                               -1] == -1 else 'Neutral',
                        round(df['fast_ema'].iloc[-1], 4),
                        round(df['medium_ema'].iloc[-1], 4),
                        round(df['slow_ema'].iloc[-1], 4)
                    ]
                }
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Estadísticas', index=False)

                # Hoja 4: Configuración de la estrategia
                config_data = {
                    'Parámetro': [
                        'Símbolo',
                        'Timeframe',
                        'PAC Length',
                        'EMA Rápida',
                        'EMA Media',
                        'EMA Lenta',
                        'Lookback',
                        'Fecha Análisis'
                    ],
                    'Valor': [
                        self.symbol,
                        self.timeframe,
                        self.pac_length,
                        self.fast_ema,
                        self.medium_ema,
                        self.slow_ema,
                        self.lookback,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ]
                }
                config_df = pd.DataFrame(config_data)
                config_df.to_excel(writer, sheet_name='Configuración', index=False)

            print(f"✅ Datos exportados exitosamente!")
            print(f"📊 Hojas creadas:")
            print(f"   • Datos_Completos: Todos los datos e indicadores")
            print(f"   • Señales: Solo las velas con señales de trading")
            print(f"   • Estadísticas: Resumen de métricas")
            print(f"   • Configuración: Parámetros de la estrategia")
            print()

        except Exception as e:
            print(f"❌ Error exportando a Excel: {e}")

    def create_visualization(self, df: pd.DataFrame):
        """
        Crea gráficos para visualizar los datos
        """
        print("📈 Creando visualizaciones...")

        try:
            # Configurar el estilo
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))

            # Gráfico 1: Precio y EMAs
            recent_data = df.tail(200)  # Últimas 200 velas

            axes[0].plot(recent_data.index, recent_data['close'], label='Precio', linewidth=2, color='black')
            axes[0].plot(recent_data.index, recent_data['fast_ema'], label=f'EMA {self.fast_ema}', alpha=0.7)
            axes[0].plot(recent_data.index, recent_data['medium_ema'], label=f'EMA {self.medium_ema}', alpha=0.7)
            axes[0].plot(recent_data.index, recent_data['slow_ema'], label=f'EMA {self.slow_ema}', alpha=0.7)
            axes[0].set_title(f'{self.symbol} - Precio y EMAs')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Gráfico 2: PAC Channel
            axes[1].plot(recent_data.index, recent_data['close'], label='Precio', linewidth=2, color='black')
            axes[1].plot(recent_data.index, recent_data['pac_high'], label='PAC High', color='green', alpha=0.7)
            axes[1].plot(recent_data.index, recent_data['pac_close'], label='PAC Center', color='blue', alpha=0.7)
            axes[1].plot(recent_data.index, recent_data['pac_low'], label='PAC Low', color='red', alpha=0.7)
            axes[1].fill_between(recent_data.index, recent_data['pac_high'], recent_data['pac_low'], alpha=0.1,
                                 color='gray')
            axes[1].set_title('PAC Channel')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # Gráfico 3: Señales
            axes[2].plot(recent_data.index, recent_data['close'], label='Precio', linewidth=2, color='black')

            # Marcar señales de compra
            buy_signals = recent_data[recent_data['buy_signal']]
            if len(buy_signals) > 0:
                axes[2].scatter(buy_signals.index, buy_signals['close'],
                                color='green', marker='^', s=100, label='Señal Compra', zorder=5)

            # Marcar señales de venta
            sell_signals = recent_data[recent_data['sell_signal']]
            if len(sell_signals) > 0:
                axes[2].scatter(sell_signals.index, sell_signals['close'],
                                color='red', marker='v', s=100, label='Señal Venta', zorder=5)

            axes[2].set_title('Señales de Trading')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()

            # Guardar gráfico
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_filename = f"trading_chart_{self.symbol}_{timestamp}.png"
            plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico guardado: {chart_filename}")

            plt.show()

        except Exception as e:
            print(f"❌ Error creando visualizaciones: {e}")

    def run_complete_analysis(self, limit: int = 1000):
        """
        Ejecuta el análisis completo
        """
        print("🚀 INICIANDO ANÁLISIS COMPLETO DE DATOS DE TRADING")
        print("=" * 60)
        print()

        # 1. Obtener datos
        df = self.get_historical_data(limit)
        if df.empty:
            print("❌ No se pudieron obtener datos")
            return

        # 2. Calcular indicadores
        df = self.calculate_all_indicators(df)

        # 3. Analizar señales
        self.analyze_signals(df)

        # 4. Exportar a Excel
        self.export_to_excel(df)

        # 5. Crear visualizaciones
        self.create_visualization(df)

        print("🎉 ¡ANÁLISIS COMPLETO FINALIZADO!")
        print("📁 Archivos generados:")
        print("   • Excel con todos los datos")
        print("   • Gráfico PNG con visualizaciones")
        print()

        return df


# Script principal para ejecutar
if __name__ == "__main__":
    # Configurar credenciales
    API_KEY = 'bPT08S5D4JijR0qJDAZh2ILB35yjQ1s5ozHxmVcY0qGL4eywVeVHPy1PjfgCkkwK'
    API_SECRET = '2xIUXyeIA82A367HPrHXklozZl6DafBQJjT9sVfsv6TfdOoxMCt7LTKRc2BPsagj'
    USE_TESTNET = False

    print("🔍 ANALIZADOR DE DATOS DE TRADING")
    print("=" * 50)
    print("Este script analiza los mismos datos que usa el bot de trading")
    print("y genera un Excel con toda la información detallada.")
    print()

    try:
        # Crear analizador
        analyzer = TradingDataAnalyzer(
            api_key=API_KEY,
            api_secret=API_SECRET,
            testnet=USE_TESTNET
        )

        # Ejecutar análisis completo
        df = analyzer.run_complete_analysis(limit=1000)

        if df is not None:
            print("✅ Puedes revisar el archivo Excel generado para ver:")
            print("   📊 Todos los datos del DataFrame")
            print("   🧮 Todos los indicadores calculados")
            print("   🚨 Todas las señales generadas")
            print("   📈 Estadísticas completas")
            print("   ⚙️ Configuración de la estrategia")

    except Exception as e:
        print(f"❌ Error en el análisis: {e}")

    input("Presiona Enter para salir...")