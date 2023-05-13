import ccxt
import pandas as pd
from ta import momentum, volume, volatility, trend, others

StrSeriesPairs = list[tuple[str, pd.Series]]

def retrieve_data():
	exchange = ccxt.kraken()
	ohlc = exchange.fetch_ohlcv(symbol='BTC/AUD', timeframe='1d')
	df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
	df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
	df.set_index('timestamp', inplace=True)

	return df

def add_all_indicators(df: pd.DataFrame, indicators_and_candle_values: StrSeriesPairs):
	for col, value in indicators_and_candle_values[:-5]:
		df[col] = value

	# Normalise by the first non-nan value
	for col in df.columns:
		first_value = df[col].loc[~df[col].isnull()].iloc[0]
		df[col] /= first_value

	return df

def list_indicators_and_candle_values(df: pd.DataFrame) -> StrSeriesPairs:
	return [
		('awesome_oscillator', momentum.awesome_oscillator(df['high'], df['low'])),
		('kama', momentum.kama(df['close'])),
		('ppo', momentum.ppo(df['close'])),
		('pvo', momentum.pvo(df['volume'])),
		('roc', momentum.roc(df['close'])),
		('rsi', momentum.rsi(df['close'])),
		('stochrsi', momentum.stochrsi(df['close'])),
		('stoch', momentum.stoch(df['high'], df['low'], df['close'])),
		('tsi', momentum.tsi(df['close'])),
		('ultimate_oscillator', momentum.ultimate_oscillator(df['high'], df['low'], df['close'])),
		('williams_r', momentum.williams_r(df['high'], df['low'], df['close'])),
		('acc_dist_index', volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])),
		('chaikin_money_flow', volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])),
		('ease_of_movement', volume.ease_of_movement(df['high'], df['low'], df['volume'])),
		('force_index', volume.force_index(df['close'], df['volume'])),
		('mfi', volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])),
		('negative_volume_index', volume.negative_volume_index(df['close'], df['volume'])),
		('on_balance_volume', volume.on_balance_volume(df['close'], df['volume'])),
		('volume_price_trend', volume.volume_price_trend(df['close'], df['volume'])),
		('volume_weighted_average_price', volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])),
		('roc', volatility.average_true_range(df['high'], df['low'], df['close'])),
		('bollinger_mavg', volatility.bollinger_mavg(df['close'])),
		('bollinger_hband', volatility.bollinger_hband(df['close'])),
		('bollinger_lband', volatility.bollinger_lband(df['close'])),
		('bollinger_pband', volatility.bollinger_pband(df['close'])),
		('bollinger_wband', volatility.bollinger_wband(df['close'])),
		('donchian_channel_hband', volatility.donchian_channel_hband(df['high'], df['low'], df['close'])),
		('donchian_channel_lband', volatility.donchian_channel_lband(df['high'], df['low'], df['close'])),
		('donchian_channel_pband', volatility.donchian_channel_pband(df['high'], df['low'], df['close'])),
		('donchian_channel_wband', volatility.donchian_channel_wband(df['high'], df['low'], df['close'])),
		('donchian_channel_mband', volatility.donchian_channel_mband(df['high'], df['low'], df['close'])),
		('keltner_channel_hband', volatility.keltner_channel_hband(df['high'], df['low'], df['close'])),
		('keltner_channel_lband', volatility.keltner_channel_lband(df['high'], df['low'], df['close'])),
		('keltner_channel_mband', volatility.keltner_channel_mband(df['high'], df['low'], df['close'])),
		('keltner_channel_pband', volatility.keltner_channel_pband(df['high'], df['low'], df['close'])),
		('keltner_channel_wband', volatility.keltner_channel_wband(df['high'], df['low'], df['close'])),
		('ulcer_index', volatility.ulcer_index(df['close'])),
		('adx_pos', trend.adx_pos(df['high'], df['low'], df['close'])),
		('adx_neg', trend.adx_neg(df['high'], df['low'], df['close'])),
		('aroon_up', trend.aroon_up(df['close'])),
		('aroon_down', trend.aroon_down(df['close'])),
		('cci', trend.cci(df['high'], df['low'], df['close'])),
		('dpo', trend.dpo(df['close'])),
		('ema', trend.ema_indicator(df['close'])),
		('ichimoku_a', trend.ichimoku_a(df['high'], df['low'])),
		('ichimoku_b', trend.ichimoku_b(df['high'], df['low'])),
		('ichimoku_base_line', trend.ichimoku_base_line(df['high'], df['low'])),
		('kst', trend.kst(df['close'])),
		('mass_index', trend.mass_index(df['high'], df['low'])),
		('macd', trend.macd(df['close'])),
		('psar_up', trend.psar_up(df['high'], df['low'], df['close'])),
		('psar_down', trend.psar_down(df['high'], df['low'], df['close'])),
		('sma', trend.sma_indicator(df['close'])),
		('stc', trend.stc(df['close'])),
		('trix', trend.trix(df['close'])),
		('vortex_indicator_pos', trend.vortex_indicator_pos(df['high'], df['low'], df['close'])),
		('vortex_indicator_neg', trend.vortex_indicator_neg(df['high'], df['low'], df['close'])),
		('wma', trend.wma_indicator(df['close'])),
		('cumulative_return', others.cumulative_return(df['close'])),
		('daily_return', others.daily_return(df['close'])),
		('daily_log_return', others.daily_log_return(df['close'])),
		('open', df['close']),
		('high', df['high']),
		('low', df['low']),
		('close', df['close']),
		('volume', df['volume']),
	]