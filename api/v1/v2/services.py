# service.py
from datetime import datetime, timedelta
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import json
import numpy as np

def add_vwap_signals(data, do_arrows=True, slope_degree=45, volumeLength=50):
    
    
    data['ema5'] = data['Close'].ewm(span=5, adjust=False).mean()
    data['ema12'] = data['Close'].ewm(span=12, adjust=False).mean()

    data['avgVol'] = ta.sma(data['Volume'], volumeLength)
    data['high_volume'] = data['Volume'] > data['avgVol']

    data['isRedCandle'] = data['Open'] > data['Close'] 
    data['isGreenCandle'] = data['Close'] > data['Open'] 

    # Green 
    data['ripsterSignalXup'] = (data['ema5'] > data['ema12']) & (data['ema5'].shift(1) <= data['ema12'].shift(1))
    data['ripsterSignalXdn'] = (data['ema5'] < data['ema12']) & (data['ema5'].shift(1) >= data['ema12'].shift(1))

    # magenta 
    data['priceAboveEMA5'] = (data['ema5'] > data['ema12']) & (data['Low'] < data['ema12']) & (data['Low'].shift(1) >= data['ema12'].shift(1)) & data['isRedCandle']
    data['priceBelowEMA5'] = (data['ema5'] < data['ema12']) & (data['High'] > data['ema12']) & (data['High'].shift(1) <= data['ema12'].shift(1)) & data['isGreenCandle']

    # Signal 
    if do_arrows:
        data['signal_up'] = data['high_volume'] & data['priceBelowEMA5'] 
        data['signal_down'] = data['high_volume'] & data['priceAboveEMA5'] 
        data['ripster_signal_up'] = data['ripsterSignalXup']
        data['ripster_signal_down'] = data['ripsterSignalXdn']
    else:
        data['signal_up'] = False
        data['signal_down'] = False
        data['ripster_signal_up'] = False
        data['ripster_signal_down'] = False
    
    data['debug_high_volume'] = data['high_volume'] 
    data['debug_ema5_gt_ema12'] = data['ema5'] > data['ema12']
    data['debug_ema5_lt_ema12'] = data['ema5'] < data['ema12']
    data['debug_low_cross_ema12'] = (data['Low'] < data['ema12']) & (data['Low'].shift(1) >= data['ema12'].shift(1))
    data['debug_high_cross_ema12'] = (data['High'] > data['ema12']) & (data['High'].shift(1) <= data['ema12'].shift(1))

    data['RSI'] = ta.rsi(data['Close'], length=14)
   
    resetUpper = 1
    resetLower = 1
    upperVwapCrossYellow = 0
    lowerVwapCrossYellow = 0
    for i in range(0, len(data)):
        if data['RSI'].iloc[i] > 70:
            upperVwapCrossYellow = 1
        elif (upperVwapCrossYellow == 1) and resetUpper == 1 :
            upperVwapCrossYellow = 1
        else:
            upperVwapCrossYellow = 0

        upperBand_vwap_cross_green = upperVwapCrossYellow == 1 and data['Close'].iloc[i] < data['Open'].iloc[i] and data['MACDh_12_26_9'].iloc[i] < data['MACDh_12_26_9'].iloc[i-1]
        if(upperBand_vwap_cross_green):
            data.loc[data.index[i], 'upperVwapCrossGreen'] = 1
        else:
            data.loc[data.index[i], 'upperVwapCrossGreen'] = 0

        if data['upperVwapCrossGreen'].iloc[i] == 1:
            resetUpper = 0
        else:
            resetUpper = 1

        if data['RSI'].iloc[i] < 30:
            lowerVwapCrossYellow = 1
        elif (lowerVwapCrossYellow == 1) and resetLower == 1 :
            lowerVwapCrossYellow = 1
        else:
            lowerVwapCrossYellow = 0
        
        lowerBand_vwap_cross_green = lowerVwapCrossYellow == 1 and data['Close'].iloc[i] > data['Open'].iloc[i] and data['MACDh_12_26_9'].iloc[i] > data['MACDh_12_26_9'].iloc[i-1]
        if(lowerBand_vwap_cross_green):
            data.loc[data.index[i], 'lowerVwapCrossGreen'] = 1
        else:
            data.loc[data.index[i], 'lowerVwapCrossGreen'] = 0

        if data['lowerVwapCrossGreen'].iloc[i] == 1:
            resetLower = 0
        else:
            resetLower = 1

    return data
def calculate_ttm_waves(data):

    data['macd_a_slow'] = ta.macd(data['Close'], fastperiod=8, slowperiod=55, signalperiod=55).iloc[:, 2] 
    data['macd_a_fast'] = ta.macd(data['Close'], fastperiod=8, slowperiod=34, signalperiod=34).iloc[:, 2]

     # ATR Calculation
    atr_length = 14
    atr1_multiplier = 1.5

    data['atr'] = ta.atr(data['High'], data['Low'], data['Close'], length=atr_length)
    data['atr1'] = data['atr'] * atr1_multiplier

    length = 20
    bb_mult = 2.0
    kc_mult_high = 1.0
    kc_mult_mid = 1.5
    kc_mult_low = 2.0
    
    # Bollinger Bands
    bb = ta.bbands(data['Close'], length=length, std=bb_mult)
    data['bb_basis'] = bb['BBM_20_2.0']
    data['bb_upper'] = bb['BBU_20_2.0']
    data['bb_lower'] = bb['BBL_20_2.0']
    
    # Keltner Channels
    data['tr'] = ta.true_range(data['High'], data['Low'], data['Close'])
    data['atr'] = ta.sma(data['tr'], length)
    data['kc_basis'] = ta.sma(data['Close'], length)
    
    # KC Bands
    for mult in [kc_mult_high, kc_mult_mid, kc_mult_low]:
        suffix = f"_{str(mult).replace('.', '_')}"
        data[f'kc_upper{suffix}'] = data['kc_basis'] + data['atr'] * mult
        data[f'kc_lower{suffix}'] = data['kc_basis'] - data['atr'] * mult
    
    # Squeeze Conditions
    data['no_sqz'] = (data['bb_lower'] < data['kc_lower_2_0']) | (data['bb_upper'] > data['kc_upper_2_0'])
    data['low_sqz'] = (data['bb_lower'] >= data['kc_lower_2_0']) | (data['bb_upper'] <= data['kc_upper_2_0'])
    data['mid_sqz'] = (data['bb_lower'] >= data['kc_lower_1_5']) | (data['bb_upper'] <= data['kc_upper_1_5'])
    data['high_sqz'] = (data['bb_lower'] >= data['kc_lower_1_0']) | (data['bb_upper'] <= data['kc_upper_1_0'])
    
    # Momentum
    highest_high = data['High'].rolling(window=length).max()
    lowest_low = data['Low'].rolling(window=length).min()
    avg_hl = (highest_high + lowest_low) / 2
    avg_hl_close = (avg_hl + data['Close'].rolling(window=length).mean()) / 2
    data['momentum'] = ta.linreg(data['Close'] - avg_hl_close, length=length)
    
    # AO (Awesome Oscillator)
    data['ao'] = ta.sma((data['High'] + data['Low']) / 2, 5) - ta.sma((data['High'] + data['Low']) / 2, 34)    
    
    return data

def fetch_yahoo_data(ticker, interval, ema_period=20, macd_fast=12, macd_slow=26, macd_signal=9, vwap_period=20, vwap_std_dev=2):
    end_date = datetime.now()
    if interval in ['1m', '5m']:
        start_date = end_date - timedelta(days=7)       
    elif interval in ['15m', '60m']:
        start_date = end_date - timedelta(days=60)
    elif interval == '1d':
        start_date = end_date - timedelta(days=365*5)
    elif interval == '1wk':
        start_date = end_date - timedelta(weeks=365*5)
    elif interval == '1mo':
        start_date = end_date - timedelta(days=365*5)

    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    
    data['EMA'] = ta.ema(data['Close'], length=int(ema_period))

    macd = ta.macd(data['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    data = pd.concat([data, macd], axis=1)

    data = pd.DataFrame(data)
    data.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
    print(data.head())
    
    data['Volume_MA'] = data['Volume'].rolling(window=20).mean()

    data['VWAP'] = ta.vwap(data['High'], data['Low'], data['Close'], data['Volume'])

    # Calculate VWAP bands
    data['VWAP_Std'] = data['VWAP'].rolling(window=vwap_period).std()
    data['VWAP_Upper'] = data['VWAP'] + (vwap_std_dev * data['VWAP_Std'])
    data['VWAP_Lower'] = data['VWAP'] - (vwap_std_dev * data['VWAP_Std'])
    data = add_vwap_signals(data)
    data = calculate_ttm_waves(data)

    candlestick_data = [
        {
            'time': int(row.Index.timestamp()),
            'open': row.Open,
            'high': row.High,
            'low': row.Low,
            'close': row.Close
        }
        for row in data.itertuples()
    ]

    ema_data = [
        {
            'time': int(row.Index.timestamp()),
            'value': row.EMA
        }
        for row in data.itertuples() if not pd.isna(row.EMA)
    ]

    macd_data = [
        {
            'time': int(row.Index.timestamp()),
            'macd': row.MACD_12_26_9 if not pd.isna(row.MACD_12_26_9) else 0,
            'signal': row.MACDs_12_26_9 if not pd.isna(row.MACDs_12_26_9) else 0,
            'histogram': row.MACDh_12_26_9 if not pd.isna(row.MACDh_12_26_9) else 0
        }
        for row in data.itertuples()
    ]

    vwap_data = [
        {
            'time': int(row.Index.timestamp()),
            'vwap': row.VWAP if not pd.isna(row.VWAP) else 0,
            'upper': row.VWAP_Upper if not pd.isna(row.VWAP_Upper) else 0,
            'lower': row.VWAP_Lower if not pd.isna(row.VWAP_Lower) else 0
        }
        for row in data.itertuples()
    ]

    arrow_signals = [
        {
        'time': int(row.Index.timestamp()),
        'signal_up': bool(row.signal_up),
        'signal_down': bool(row.signal_down),
        'ripster_signal_up': bool(row.ripster_signal_up),
        'ripster_signal_down': bool(row.ripster_signal_down),
        'debug_high_volume': bool(row.Volume > row.avgVol),
        'debug_ema5_gt_ema12': bool(row.ema5 > row.ema12),
        'debug_low_cross_ema12': bool((row.Low < row.ema12) and (data['Low'].shift(1).iloc[i] >= data['ema12'].shift(1).iloc[i])),
        'isRedCandle': bool(row.isRedCandle),
        'debug_ema5_lt_ema12': bool(row.ema5 < row.ema12),
        'debug_high_cross_ema12': bool((row.High > row.ema12) and (data['High'].shift(1).iloc[i] <= data['ema12'].shift(1).iloc[i])),
        'isGreenCandle': bool(row.isGreenCandle),
        'ema5': row.ema5,
        'ema12': row.ema12,
        'low': row.Low,
        'high': row.High,
        'yellow_signal_up': bool(row.lowerVwapCrossGreen),
        'yellow_signal_down': bool(row.upperVwapCrossGreen)        
    }
    for i, row in enumerate(data.itertuples())
    ]
    ttm_waves_data = [
        {
            'time': int(index.timestamp()),
            'wave_a_slow': row['macd_a_slow'] if not pd.isna(row['macd_a_slow']) else 0,
            'wave_a_fast': row['macd_a_fast'] if not pd.isna(row['macd_a_fast']) else 0,
            'momentum': row['momentum'] if not pd.isna(row['momentum']) else 0,
            'ao': row['ao'] if not pd.isna(row['ao']) else 0,
            'squeeze': 'high' if row['high_sqz'] else 'mid' if row['mid_sqz'] else 'low' if row['low_sqz'] else 'none',
            'atr1': row['atr1'] if not pd.isna(row['atr1']) else 0
        }
        for index, row in data.iterrows()
    ]
    
    return candlestick_data, ema_data, macd_data, vwap_data, arrow_signals, ttm_waves_data
    