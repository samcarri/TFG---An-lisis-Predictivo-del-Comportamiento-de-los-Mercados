#!/usr/bin/env python3
"""
Backend API para servir datos reales de NVDA desde el Módulo 3.
"""

from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
import pandas as pd
from pathlib import Path
import sys
import requests

app = Flask(__name__)
CORS(app)

# Paths a los datos
BASE_DIR = Path(__file__).parent.parent.parent / "Modulo3_NewsPredictor"
UNIFIED_CSV = BASE_DIR / "collected_news" / "processed_unified_news" / "merged_financial_news_features.csv"


@app.route('/api/market/calendar', methods=['GET'])
def get_market_calendar():
    """Consulta si una fecha es día de trading y cuál es el anterior/siguiente."""
    try:
        date_str = request.args.get('date')
        if not date_str:
            from datetime import date as _date
            date_str = _date.today().strftime('%Y-%m-%d')
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from tools.mcp.mcp_market_agent import get_market_calendar_info
        result = get_market_calendar_info(date=date_str)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/refresh', methods=['POST'])
def refresh_all_data():
    """
    Recopila datos faltantes para:
    1. financial_data.csv (gráfica principal) — solo fechas de trading no presentes
    2. dataset_nvda_news.csv (ensemble) — solo si hay fechas de trading sin features de noticias
    Nunca sobreescribe datos existentes.
    """
    try:
        import yfinance as yf
        import pandas_market_calendars as mcal
        from datetime import date, timedelta

        ROOT = Path(__file__).parent.parent.parent
        fin_csv    = ROOT / 'data' / 'financial_data.csv'
        news_csv   = ROOT / 'entrenamiento_agentes' / 'data' / 'dataset_nvda_news.csv'

        nyse  = mcal.get_calendar('NYSE')
        today = date.today()
        results = {}

        # ── 1. financial_data.csv ─────────────────────────────────────────────
        schedule_1y = nyse.schedule(
            start_date=(today - timedelta(days=365)).strftime('%Y-%m-%d'),
            end_date=today.strftime('%Y-%m-%d')
        )
        expected = set(schedule_1y.index.strftime('%Y-%m-%d').tolist())

        existing = set()
        df_fin = pd.DataFrame()
        if fin_csv.exists() and fin_csv.stat().st_size > 0:
            df_fin = pd.read_csv(fin_csv)
            df_fin['date'] = pd.to_datetime(df_fin['date']).dt.strftime('%Y-%m-%d')
            existing = set(df_fin['date'].tolist())

        missing_fin = sorted(expected - existing)

        if missing_fin:
            fetch_start = missing_fin[0]
            fetch_end   = (date.fromisoformat(missing_fin[-1]) + timedelta(days=1)).strftime('%Y-%m-%d')
            new_df = yf.download('NVDA', start=fetch_start, end=fetch_end, progress=False)
            if not new_df.empty:
                new_df = new_df.reset_index()
                if isinstance(new_df.columns, pd.MultiIndex):
                    new_df.columns = [c[0] for c in new_df.columns]
                new_df = new_df.rename(columns={'Date':'date','Open':'open','High':'high',
                                                 'Low':'low','Close':'close','Volume':'volume','Adj Close':'adj_close'})
                new_df['date'] = pd.to_datetime(new_df['date']).dt.strftime('%Y-%m-%d')
                new_df['ticker'] = 'NVDA'
                if 'adj_close' not in new_df.columns:
                    new_df['adj_close'] = new_df['close']
                cols = ['date','open','high','low','close','volume','adj_close','ticker']
                new_df = new_df[[c for c in cols if c in new_df.columns]]
                # Solo añadir fechas realmente faltantes
                new_df = new_df[new_df['date'].isin(set(missing_fin))]
                if not df_fin.empty:
                    combined = pd.concat([df_fin, new_df], ignore_index=True)
                    combined = combined.drop_duplicates(subset='date').sort_values('date')
                else:
                    combined = new_df.sort_values('date')
                combined.to_csv(fin_csv, index=False)
                results['financial_data'] = {'added': len(new_df), 'missing_before': len(missing_fin)}
            else:
                results['financial_data'] = {'added': 0, 'missing_before': len(missing_fin), 'note': 'yfinance sin datos'}
        else:
            results['financial_data'] = {'added': 0, 'up_to_date': True}

        # ── 2. dataset_nvda_news.csv (ensemble) ───────────────────────────────
        if news_csv.exists() and news_csv.stat().st_size > 0:
            df_news = pd.read_csv(news_csv)
            last_news_date = pd.to_datetime(df_news['date']).max().date()
            schedule_news = nyse.schedule(
                start_date=(last_news_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                end_date=today.strftime('%Y-%m-%d')
            )
            missing_news = len(schedule_news)
            if missing_news > 0:
                results['ensemble_news'] = {
                    'missing_trading_days': missing_news,
                    'last_date': str(last_news_date),
                    'note': 'Ejecuta build_news_dataset.py para actualizar features de noticias'
                }
            else:
                results['ensemble_news'] = {'up_to_date': True, 'last_date': str(last_news_date)}
        else:
            results['ensemble_news'] = {'error': 'Dataset no encontrado'}

        return jsonify({'success': True, 'results': results})

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()[-400:]}), 500


@app.route('/api/stock/current', methods=['GET'])
def get_current_stock():
    """Obtiene el precio actual y estadísticas desde financial_data.csv"""
    try:
        fin_csv = Path(__file__).parent.parent.parent / 'data' / 'financial_data.csv'
        df = pd.read_csv(fin_csv)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        price_change = latest['close'] - previous['close']
        price_change_pct = (price_change / previous['close']) * 100

        last_year = df.tail(252)

        return jsonify({
            'price': float(latest['close']),
            'change': float(price_change),
            'change_percent': float(price_change_pct),
            'volume_24h': float(latest['volume']),
            'high_52w': float(last_year['high'].max()),
            'low_52w': float(last_year['low'].min()),
            'date': latest['date'].strftime('%Y-%m-%d')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stock/history', methods=['GET'])
def get_stock_history():
    """Obtiene histórico de precios para el gráfico desde financial_data.csv, actualizando si faltan días."""
    try:
        days = request.args.get('days', 30, type=int)
        specific_date = request.args.get('date')  # para 1D: fecha exacta

        fin_csv = Path(__file__).parent.parent.parent / 'data' / 'financial_data.csv'

        # --- Actualizar CSV si faltan días recientes ---
        from datetime import date, timedelta
        import pandas_market_calendars as mcal

        nyse = mcal.get_calendar('NYSE')
        today = date.today()
        # Últimos N días de trading esperados
        schedule = nyse.schedule(
            start_date=(today - timedelta(days=days * 2)).strftime('%Y-%m-%d'),
            end_date=today.strftime('%Y-%m-%d')
        )
        expected_days = set(schedule.index.strftime('%Y-%m-%d').tolist())

        existing_days = set()
        if fin_csv.exists() and fin_csv.stat().st_size > 0:
            df_existing = pd.read_csv(fin_csv)
            df_existing['date'] = pd.to_datetime(df_existing['date']).dt.strftime('%Y-%m-%d')
            existing_days = set(df_existing['date'].tolist())

        missing = sorted(expected_days - existing_days)

        if missing:
            try:
                import yfinance as yf
                fetch_start = missing[0]
                fetch_end = (date.fromisoformat(missing[-1]) + timedelta(days=1)).strftime('%Y-%m-%d')
                new_df = yf.download('NVDA', start=fetch_start, end=fetch_end, progress=False)
                if not new_df.empty:
                    new_df = new_df.reset_index()
                    # Aplanar MultiIndex si existe
                    if isinstance(new_df.columns, pd.MultiIndex):
                        new_df.columns = [c[0] if c[1] == '' else c[0] for c in new_df.columns]
                    new_df = new_df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
                                                     'Low': 'low', 'Close': 'close', 'Volume': 'volume',
                                                     'Adj Close': 'adj_close'})
                    new_df['date'] = pd.to_datetime(new_df['date']).dt.strftime('%Y-%m-%d')
                    new_df['ticker'] = 'NVDA'
                    if 'adj_close' not in new_df.columns:
                        new_df['adj_close'] = new_df['close']
                    cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'ticker']
                    new_df = new_df[[c for c in cols if c in new_df.columns]]

                    if fin_csv.exists() and fin_csv.stat().st_size > 0:
                        df_existing = pd.read_csv(fin_csv)
                        df_existing['date'] = pd.to_datetime(df_existing['date']).dt.strftime('%Y-%m-%d')
                        combined = pd.concat([df_existing, new_df], ignore_index=True)
                        combined = combined.drop_duplicates(subset='date').sort_values('date')
                    else:
                        combined = new_df.sort_values('date')

                    combined.to_csv(fin_csv, index=False)
            except Exception as e:
                print(f"[history] yfinance update failed: {e}")

        # --- Leer y devolver ---
        df = pd.read_csv(fin_csv)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        if specific_date:
            # Modo 1D: devolver los últimos `days` registros hasta la fecha indicada
            target = pd.to_datetime(specific_date)
            data = df[df['date'] <= target].tail(days if days > 1 else 2)
        else:
            data = df.tail(days)

        candles = []
        for _, row in data.iterrows():
            candles.append({
                'x': row['date'].strftime('%Y-%m-%d'),
                'o': float(row['open']),
                'h': float(row['high']),
                'l': float(row['low']),
                'c': float(row['close'])
            })

        return jsonify(candles)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/news', methods=['GET'])
def get_news():
    """Obtiene noticias financieras desde nvidia_news_cache.csv"""
    try:
        limit = request.args.get('limit', 10, type=int)

        news_csv = Path(__file__).parent.parent.parent / 'data' / 'nvidia_news_cache.csv'
        if not news_csv.exists():
            return jsonify({'error': 'News data not found'}), 404

        df = pd.read_csv(news_csv)
        # Columna de fecha puede llamarse 'date' o 'created_at'
        date_col = 'date' if 'date' in df.columns else 'created_at'
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col]).sort_values(date_col, ascending=False)

        news_list = []
        for _, row in df.head(limit).iterrows():
            title_col = 'title' if 'title' in row else 'headline'
            title = str(row.get(title_col, ''))
            summary = str(row.get('summary', '') or row.get('description', '') or title)
            if pd.isna(summary) or summary == 'nan':
                summary = title

            label = str(row.get('financialbert_label', '') or '').lower()
            prob = row.get('financialbert_probability', 0.5)
            try:
                prob = float(prob) if not pd.isna(prob) else 0.5
            except:
                prob = 0.5
            # Convertir label + probabilidad a score centrado en 0
            if label == 'positive':
                score = round(prob, 3)
            elif label == 'negative':
                score = round(-prob, 3)
            else:
                score = 0.0

            url = str(row.get('url', '') or '')
            if url == 'nan':
                url = ''

            news_list.append({
                'headline': title,
                'summary': summary,
                'source': str(row.get('source', 'Alpaca')),
                'date': row[date_col].strftime('%Y-%m-%d'),
                'url': url,
                'sentiment_score': score
            })

        return jsonify(news_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/news/refresh', methods=['POST'])
def refresh_news():
    """Descarga noticias desde la última fecha del caché hasta hoy usando Alpaca API."""
    try:
        from datetime import date as _date
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        news_csv = Path(__file__).parent.parent.parent / 'data' / 'nvidia_news_cache.csv'
        today = _date.today().strftime('%Y-%m-%d')

        # Determinar fecha de inicio: día siguiente a la última noticia en caché
        if news_csv.exists() and news_csv.stat().st_size > 0:
            df_existing = pd.read_csv(news_csv)
            df_existing['date'] = pd.to_datetime(df_existing['date'], errors='coerce')
            last_date = df_existing['date'].max()
            if pd.isna(last_date):
                start_date = (pd.Timestamp.now() - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
            else:
                start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=7)).strftime('%Y-%m-%d')

        if start_date > today:
            return jsonify({'success': True, 'message': 'Las noticias ya están al día', 'added': 0})

        from tools.mcp.mcp_news_agent import _fetch_alpaca_and_update_cache
        df_updated = _fetch_alpaca_and_update_cache('NVDA', start_date, today)

        added = 0
        if not df_updated.empty:
            df_updated['date'] = pd.to_datetime(df_updated['date'], errors='coerce')
            added = int((df_updated['date'] >= pd.Timestamp(start_date)).sum())

        return jsonify({
            'success': True,
            'message': f'{added} noticias nuevas añadidas ({start_date} → {today})',
            'added': added,
            'start_date': start_date,
            'end_date': today,
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()[-400:]}), 500


@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    """Forecast basado en momentum de los últimos 5 días desde financial_data.csv"""
    try:
        fin_csv = Path(__file__).parent.parent.parent / 'data' / 'financial_data.csv'
        df = pd.read_csv(fin_csv)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        closes = df['close'].tail(14).tolist()
        current_price = closes[-1]

        # RSI simple 14 días
        gains, losses = [], []
        for i in range(1, len(closes)):
            d = closes[i] - closes[i-1]
            (gains if d > 0 else losses).append(abs(d))
        avg_gain = sum(gains) / 14 if gains else 0
        avg_loss = sum(losses) / 14 if losses else 0.001
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))

        # Momentum: retorno últimos 5 días
        momentum = (closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0

        if rsi < 45 or momentum < -2:
            direction = 'down'
            forecast_change = round(momentum * 0.3, 2)
        elif rsi > 55 or momentum > 2:
            direction = 'up'
            forecast_change = round(momentum * 0.3, 2)
        else:
            direction = 'up' if momentum >= 0 else 'down'
            forecast_change = round(momentum * 0.2, 2)

        forecast_price = current_price * (1 + forecast_change / 100)
        from datetime import date
        tomorrow = date.today().strftime('%d %b %Y')

        return jsonify({
            'forecast_change_percent': forecast_change,
            'forecast_price': round(forecast_price, 2),
            'direction': direction,
            'confidence': round(min(0.5 + abs(momentum) * 0.05, 0.85), 2),
            'timestamp': f'{tomorrow}, 4:30 PM'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast/ensemble', methods=['GET'])
def get_ensemble_forecast():
    """Predicción del Voting Ensemble (RF_News + LightGBM_News + XGBoost_Reddit)."""
    try:
        import joblib
        import numpy as np
        ROOT = Path(__file__).resolve().parent.parent.parent
        MODELS_DIR = ROOT / 'entrenamiento_agentes' / 'trained_models'
        NEWS_CSV   = ROOT / 'entrenamiento_agentes' / 'data' / 'dataset_nvda_news.csv'
        REDDIT_CSV = ROOT / 'entrenamiento_agentes' / 'data' / 'dataset_nvda_reddit.csv'

        model_files = {
            'rf_news':     MODELS_DIR / 'rf_news_model.joblib',
            'rf_scaler':   MODELS_DIR / 'rf_news_scaler.joblib',
            'lgbm_news':   MODELS_DIR / 'lgbm_news_model.joblib',
            'lgbm_scaler': MODELS_DIR / 'lgbm_news_scaler.joblib',
            'xgb_reddit':  MODELS_DIR / 'xgb_reddit2_model.joblib',
            'xgb_scaler':  MODELS_DIR / 'xgb_reddit2_scaler.joblib',
        }
        for name, path in model_files.items():
            if not path.exists():
                return jsonify({'error': f'Modelo no encontrado: {name} en {path}', 'needs_refresh': True}), 404

        # Verificar que los datasets existen
        for csv_path in [NEWS_CSV, REDDIT_CSV]:
            if not csv_path.exists():
                return jsonify({'error': f'Dataset no encontrado: {csv_path.name}', 'needs_refresh': True}), 404

        rf_model    = joblib.load(model_files['rf_news'])
        rf_scaler   = joblib.load(model_files['rf_scaler'])
        lgbm_model  = joblib.load(model_files['lgbm_news'])
        lgbm_scaler = joblib.load(model_files['lgbm_scaler'])
        xgb_model   = joblib.load(model_files['xgb_reddit'])
        xgb_scaler  = joblib.load(model_files['xgb_scaler'])

        df_news   = pd.read_csv(NEWS_CSV).sort_values('date')
        df_reddit = pd.read_csv(REDDIT_CSV).sort_values('date')

        # Obtener última fecha común entre ambos datasets
        last_news_date = pd.to_datetime(df_news['date']).max()
        last_reddit_date = pd.to_datetime(df_reddit['date']).max()
        last_common_date = min(last_news_date, last_reddit_date)
        
        # Filtrar ambos datasets por la fecha común
        common_date_str = last_common_date.strftime('%Y-%m-%d')
        df_news_filtered = df_news[df_news['date'] == common_date_str]
        df_reddit_filtered = df_reddit[df_reddit['date'] == common_date_str]
        
        # Verificar que ambos tienen datos para esa fecha
        if df_news_filtered.empty or df_reddit_filtered.empty:
            return jsonify({
                'error': 'No hay datos comunes para la última fecha disponible',
                'last_news_date': str(last_news_date.date()),
                'last_reddit_date': str(last_reddit_date.date()),
                'needs_refresh': True
            }), 404

        news_features   = [c for c in df_news.columns   if c not in ('date', 'target')]
        reddit_features = [c for c in df_reddit.columns if c not in ('date', 'target')]

        X_news_last   = df_news_filtered[news_features].values
        X_reddit_last = df_reddit_filtered[reddit_features].values

        p_rf   = float(rf_model.predict_proba(rf_scaler.transform(X_news_last))[0, 1])
        p_lgbm = float(lgbm_model.predict_proba(lgbm_scaler.transform(X_news_last))[0, 1])
        p_xgb  = float(xgb_model.predict_proba(xgb_scaler.transform(X_reddit_last))[0, 1])

        avg_proba  = (p_rf + p_lgbm + p_xgb) / 3.0
        prediction = 'SUBE' if avg_proba >= 0.5 else 'BAJA'
        confidence = round(abs(avg_proba - 0.5) * 2, 3)

        fin_csv = ROOT / 'data' / 'financial_data.csv'
        current_price = None
        if fin_csv.exists():
            df_fin = pd.read_csv(fin_csv).sort_values('date')
            current_price = float(df_fin['close'].iloc[-1])

        return jsonify({
            'needs_refresh': False,
            'prediction': prediction,
            'probability': round(avg_proba, 4),
            'confidence': confidence,
            'model_votes': {
                'RF_News':        round(p_rf, 4),
                'LightGBM_News':  round(p_lgbm, 4),
                'XGBoost_Reddit': round(p_xgb, 4),
            },
            'current_price': current_price,
            'prediction_date': common_date_str,
            'last_news_date':   str(last_news_date.date()),
            'last_reddit_date': str(last_reddit_date.date()),
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()[-300:]}), 500


@app.route('/api/forecast/ensemble/refresh', methods=['POST'])
def refresh_ensemble_data():
    """Recopila datos recientes para actualizar los datasets del ensemble."""
    try:
        import subprocess
        ROOT = Path(__file__).parent.parent.parent
        TRAIN_DIR = ROOT / 'entrenamiento_agentes'
        results = []
        for script in ['scripts/preprocessing/build_news_dataset.py',
                       'scripts/preprocessing/build_reddit_dataset.py']:
            script_path = TRAIN_DIR / script
            if script_path.exists():
                result = subprocess.run(
                    ['python3', str(script_path)],
                    cwd=str(TRAIN_DIR),
                    capture_output=True, text=True, timeout=180
                )
                results.append({
                    'script': script,
                    'ok': result.returncode == 0,
                    'output': result.stdout[-300:] if result.stdout else '',
                    'error':  result.stderr[-300:] if result.stderr else '',
                })
            else:
                results.append({'script': script, 'ok': False, 'error': 'Script no encontrado'})
        return jsonify({'success': all(r['ok'] for r in results), 'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast/weekly', methods=['GET'])
def get_weekly_forecast():
    """Predicción Weekly usando Random Forest (5 días) - misma fecha que ensemble."""
    try:
        import joblib
        import numpy as np
        ROOT = Path(__file__).resolve().parent.parent.parent
        MODELS_DIR = ROOT / 'entrenamiento_agentes' / 'trained_models'
        FIN_CSV = ROOT / 'data' / 'financial_data.csv'
        NEWS_CSV = ROOT / 'entrenamiento_agentes' / 'data' / 'dataset_nvda_news.csv'
        REDDIT_CSV = ROOT / 'entrenamiento_agentes' / 'data' / 'dataset_nvda_reddit.csv'
        
        model_path = MODELS_DIR / 'rf_weekly_model.joblib'
        scaler_path = MODELS_DIR / 'rf_weekly_scaler.joblib'
        
        if not model_path.exists() or not scaler_path.exists():
            return jsonify({'error': 'Modelo weekly no encontrado', 'needs_refresh': True}), 404
        
        # Verificar que existen todos los datasets para obtener fecha común
        for csv_path in [FIN_CSV, NEWS_CSV, REDDIT_CSV]:
            if not csv_path.exists():
                return jsonify({'error': f'Dataset no encontrado: {csv_path.name}', 'needs_refresh': True}), 404
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Obtener última fecha común entre todos los datasets (igual que ensemble)
        df_news = pd.read_csv(NEWS_CSV).sort_values('date')
        df_reddit = pd.read_csv(REDDIT_CSV).sort_values('date')
        last_news_date = pd.to_datetime(df_news['date']).max()
        last_reddit_date = pd.to_datetime(df_reddit['date']).max()
        last_common_date = min(last_news_date, last_reddit_date)
        common_date_str = last_common_date.strftime('%Y-%m-%d')
        
        # Cargar datos financieros hasta esa fecha
        df = pd.read_csv(FIN_CSV).sort_values('date')
        df = df[df['date'] <= common_date_str]
        df['return'] = df['close'].pct_change()
        df['volatility_7d'] = df['return'].rolling(7).std()
        df['rsi_14'] = calculate_rsi(df['close'], 14)
        df['ema_7'] = df['close'].ewm(span=7).mean()
        df['ema_14'] = df['close'].ewm(span=14).mean()
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        df['bb_width'] = (df['close'].rolling(20).std() * 2) / df['close'].rolling(20).mean()
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df = df.dropna()
        
        if df.empty:
            return jsonify({'error': 'No hay suficientes datos para calcular features'}), 404
        
        # Features esperadas por el modelo (22 features)
        feature_cols = ['close', 'high', 'low', 'open', 'volume', 'return', 
                       'volatility_7d', 'rsi_14', 'ema_7', 'ema_14', 
                       'macd', 'macd_signal', 'macd_diff', 'bb_width', 'momentum_10']
        
        # Agregar lags si es necesario
        for i in range(1, 8):
            df[f'return_lag_{i}'] = df['return'].shift(i)
            feature_cols.append(f'return_lag_{i}')
        
        df = df.dropna()
        X_last = df[feature_cols].iloc[-1:].values
        
        proba = float(model.predict_proba(scaler.transform(X_last))[0, 1])
        prediction = 'SUBE' if proba >= 0.5 else 'BAJA'
        confidence = round(abs(proba - 0.5) * 2, 3)
        
        last_date = df['date'].iloc[-1]
        current_price = float(df['close'].iloc[-1])
        
        return jsonify({
            'needs_refresh': False,
            'prediction': prediction,
            'probability': round(proba, 4),
            'confidence': confidence,
            'model_votes': {
                'RF_Weekly': round(proba, 4),
            },
            'current_price': current_price,
            'prediction_date': last_date,
            'horizon': '5 días',
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()[-300:]}), 500


@app.route('/api/forecast/monthly', methods=['GET'])
def get_monthly_forecast():
    """Predicción Monthly usando XGBoost (21 días) - misma fecha que ensemble."""
    try:
        import joblib
        import numpy as np
        ROOT = Path(__file__).resolve().parent.parent.parent
        MODELS_DIR = ROOT / 'entrenamiento_agentes' / 'trained_models'
        FIN_CSV = ROOT / 'data' / 'financial_data.csv'
        NEWS_CSV = ROOT / 'entrenamiento_agentes' / 'data' / 'dataset_nvda_news.csv'
        REDDIT_CSV = ROOT / 'entrenamiento_agentes' / 'data' / 'dataset_nvda_reddit.csv'
        
        model_path = MODELS_DIR / 'xgb_monthly_2022_model.joblib'
        scaler_path = MODELS_DIR / 'xgb_monthly_2022_scaler.joblib'
        
        if not model_path.exists() or not scaler_path.exists():
            return jsonify({'error': 'Modelo monthly no encontrado', 'needs_refresh': True}), 404
        
        # Verificar que existen todos los datasets para obtener fecha común
        for csv_path in [FIN_CSV, NEWS_CSV, REDDIT_CSV]:
            if not csv_path.exists():
                return jsonify({'error': f'Dataset no encontrado: {csv_path.name}', 'needs_refresh': True}), 404
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Obtener última fecha común entre todos los datasets (igual que ensemble)
        df_news = pd.read_csv(NEWS_CSV).sort_values('date')
        df_reddit = pd.read_csv(REDDIT_CSV).sort_values('date')
        last_news_date = pd.to_datetime(df_news['date']).max()
        last_reddit_date = pd.to_datetime(df_reddit['date']).max()
        last_common_date = min(last_news_date, last_reddit_date)
        common_date_str = last_common_date.strftime('%Y-%m-%d')
        
        # Cargar datos financieros hasta esa fecha
        df = pd.read_csv(FIN_CSV).sort_values('date')
        df = df[df['date'] <= common_date_str]
        df['return'] = df['close'].pct_change()
        df['volatility_7d'] = df['return'].rolling(7).std()
        df['rsi_14'] = calculate_rsi(df['close'], 14)
        df['ema_7'] = df['close'].ewm(span=7).mean()
        df['ema_14'] = df['close'].ewm(span=14).mean()
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        df['bb_width'] = (df['close'].rolling(20).std() * 2) / df['close'].rolling(20).mean()
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df = df.dropna()
        
        if df.empty:
            return jsonify({'error': 'No hay suficientes datos para calcular features'}), 404
        
        # Features esperadas por el modelo
        feature_cols = ['close', 'high', 'low', 'open', 'volume', 'return', 
                       'volatility_7d', 'rsi_14', 'ema_7', 'ema_14', 
                       'macd', 'macd_signal', 'macd_diff', 'bb_width', 'momentum_10']
        
        # Agregar lags
        for i in range(1, 8):
            df[f'return_lag_{i}'] = df['return'].shift(i)
            feature_cols.append(f'return_lag_{i}')
        
        df = df.dropna()
        X_last = df[feature_cols].iloc[-1:].values
        
        proba = float(model.predict_proba(scaler.transform(X_last))[0, 1])
        prediction = 'SUBE' if proba >= 0.5 else 'BAJA'
        confidence = round(abs(proba - 0.5) * 2, 3)
        
        last_date = df['date'].iloc[-1]
        current_price = float(df['close'].iloc[-1])
        
        return jsonify({
            'needs_refresh': False,
            'prediction': prediction,
            'probability': round(proba, 4),
            'confidence': confidence,
            'model_votes': {
                'XGB_Monthly': round(proba, 4),
            },
            'current_price': current_price,
            'prediction_date': last_date,
            'horizon': '21 días',
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()[-300:]}), 500


def calculate_rsi(series, period=14):
    """Calcula el RSI (Relative Strength Index)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Endpoint SSE para el chat individual con streaming de tool calls"""
    data = request.json
    message = data.get('message', '')
    agent_type = data.get('agent', 'market')

    def generate():
        import json as _json
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        try:
            from agents.tool_tracking_handler import ToolTrackingHandler

            # Crear handler que emite eventos SSE
            tool_events = []

            class StreamingHandler(ToolTrackingHandler):
                def __init__(self, queue, **kwargs):
                    super().__init__(**kwargs)
                    self.queue = queue
                    self._emitted_steps = set()

                def __call__(self, **kwargs):
                    import json as _j
                    event = kwargs.get('event', {})
                    prev_count = len(self.tool_calls)

                    super().__call__(**kwargs)

                    data = kwargs.get('data', '')
                    if data:
                        self.queue.put({'type': 'text_chunk', 'text': data, 'complete': kwargs.get('complete', False)})

                    # Inicio de tool call
                    tool_use = event.get('contentBlockStart', {}).get('start', {}).get('toolUse')
                    if tool_use:
                        self.queue.put({'type': 'tool_call', 'name': tool_use.get('name', ''), 'agent': agent_type})

                    # Tool completada → emitir _steps del output
                    if len(self.tool_calls) > prev_count:
                        output = self.tool_calls[-1].get('output', {})
                        steps = []
                        if isinstance(output, dict):
                            steps = output.get('_steps', [])
                        elif isinstance(output, str):
                            try:
                                steps = _j.loads(output).get('_steps', [])
                            except Exception:
                                pass
                        for step in steps:
                            if step not in self._emitted_steps:
                                self._emitted_steps.add(step)
                                self.queue.put({'type': 'tool_step', 'step': step})

            import queue as _queue
            import threading
            import sys as _sys
            import io

            msg_queue = _queue.Queue()
            handler = StreamingHandler(queue=msg_queue, verbose=False, agent_name=agent_type)

            # Interceptar stdout para capturar los print() de las tools
            class StepCapture(io.StringIO):
                def write(self, s):
                    stripped = s.strip()
                    if stripped and any(stripped.startswith(p) for p in ('📂', '✅', '⚠️', '❌', '🔬', '🌐', '📊', '⚡')):
                        msg_queue.put({'type': 'tool_step', 'step': stripped})
                    # También escribir al stdout real para el log del backend
                    _original_stdout.write(s)
                def flush(self):
                    _original_stdout.flush()

            _original_stdout = _sys.stdout

            if agent_type == 'news':
                from agents.news_agent import create_news_agent
                agent = create_news_agent(callback_handler=handler)
            elif agent_type == 'reddit':
                from agents.reddit_agent import create_reddit_agent
                agent = create_reddit_agent(callback_handler=handler)
            else:
                from agents.market_agent import create_market_agent
                agent = create_market_agent(callback_handler=handler)

            if agent is None:
                yield f"data: {_json.dumps({'type': 'error', 'message': 'Ollama no disponible'})}\n\n"
                return

            yield f"data: {_json.dumps({'type': 'thinking', 'agent': agent_type})}\n\n"

            from datetime import date as _date
            _today = _date.today().strftime('%Y-%m-%d')
            _yesterday = (_date.today() - __import__('datetime').timedelta(days=1)).strftime('%Y-%m-%d')

            full_message = (
                f"[Contexto: Ticker activo = NVDA. Fecha actual = {_today}. Ayer = {_yesterday}. "
                f"Responde siempre sobre NVIDIA (NVDA). "
                f"IMPORTANTE: SIEMPRE llama a tus herramientas para obtener datos reales — "
                f"nunca respondas desde conocimiento interno sobre precios o indicadores.]\n\n{message}"
            )

            response_text = [None]
            error_text = [None]

            def run_agent():
                try:
                    _sys.stdout = StepCapture()
                    response_text[0] = str(agent(full_message))
                except Exception as e:
                    error_text[0] = str(e)
                finally:
                    _sys.stdout = _original_stdout
                    msg_queue.put({'type': 'done'})

            thread = threading.Thread(target=run_agent)
            thread.start()

            # Emitir eventos conforme llegan
            while True:
                try:
                    event = msg_queue.get(timeout=300)
                    yield f"data: {_json.dumps(event)}\n\n"
                    if event['type'] == 'done':
                        break
                except _queue.Empty:
                    yield f"data: {_json.dumps({'type': 'error', 'message': 'Timeout'})}\n\n"
                    break

            if error_text[0]:
                yield f"data: {_json.dumps({'type': 'error', 'message': error_text[0]})}\n\n"
            elif response_text[0]:
                yield f"data: {_json.dumps({'type': 'response', 'text': response_text[0]})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {_json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return app.response_class(stream_with_context(generate()), mimetype='text/event-stream',
                              headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint para el chatbot con Ollama — agente individual"""
    try:
        data = request.json
        message = data.get('message', '')
        agent_type = data.get('agent', 'market')  # market | news | reddit

        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Importar el agente correcto
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        if agent_type == 'news':
            from agents.news_agent import create_news_agent
            agent = create_news_agent()
        elif agent_type == 'reddit':
            from agents.reddit_agent import create_reddit_agent
            agent = create_reddit_agent()
        else:
            from agents.market_agent import create_market_agent
            agent = create_market_agent()

        if agent is None:
            return jsonify({'response': 'Error: Ollama no disponible', 'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')})

        from datetime import date as _date
        _today = _date.today().strftime('%Y-%m-%d')
        _yesterday = (_date.today() - __import__('datetime').timedelta(days=1)).strftime('%Y-%m-%d')
        response_text = str(agent(
            f"[Contexto: Ticker activo = NVDA. Fecha actual = {_today}. Ayer = {_yesterday}. "
            f"IMPORTANTE: SIEMPRE llama a tus herramientas para obtener datos reales.]\n\n{message}"
        ))

        return jsonify({
            'response': response_text,
            'agent': agent_type,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    except Exception as e:
        return jsonify({
            'response': f'Error: {str(e)}',
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        })


@app.route('/api/debate/stream', methods=['POST'])
def debate_stream():
    """Endpoint SSE para el debate multi-agente con streaming por agente"""
    data = request.json
    ticker = data.get('ticker', 'NVDA')
    start_date = data.get('start_date') or None
    end_date = data.get('end_date') or None
    debate_rounds = int(data.get('rounds', 1))

    # Definición de agentes con sus herramientas
    AGENT_DEFS = {
        'market': {
            'name': '💹 Market Agent', 'color': '#6366f1', 'icon': '💹',
            'description': 'Análisis técnico de mercado',
            'tools': ['get_current_date', 'get_market_data', 'get_technical_analysis', 'get_market_calendar_info']
        },
        'news': {
            'name': '📰 News Agent', 'color': '#f59e0b', 'icon': '📰',
            'description': 'Análisis de noticias y sentimiento',
            'tools': ['get_current_date', 'get_news_sentiment_summary', 'get_alpaca_news_signals', 'get_gdelt_signals', 'collect_gdelt_data']
        },
        'reddit': {
            'name': '💬 Reddit Agent', 'color': '#ec4899', 'icon': '💬',
            'description': 'Sentimiento de comunidades Reddit',
            'tools': ['get_current_date', 'get_reddit_sentiment_summary', 'get_reddit_signals']
        },
    }

    def generate():
        import json as _json
        import threading
        import queue as _queue
        from concurrent.futures import ThreadPoolExecutor, as_completed
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        try:
            from agents.debate_system import DebateSystem
            from agents.output_parser import parse_agent_output

            yield f"data: {_json.dumps({'type': 'status', 'message': '⚙️ Inicializando agentes...', 'phase': 'init'})}\n\n"

            for key, ag in AGENT_DEFS.items():
                yield f"data: {_json.dumps({'type': 'agent_init', 'agent': key, 'name': ag['name'], 'color': ag['color'], 'icon': ag['icon'], 'description': ag['description'], 'tools': ag['tools']})}\n\n"

            system = DebateSystem(verbose=False)
            system._init_agents()
            ctx = system._build_date_context(ticker, start_date, end_date)

            # ── Ronda 1: paralela con SSE por agente ──────────────────────────
            yield f"data: {_json.dumps({'type': 'status', 'message': f'🚀 Ronda 1 — Análisis inicial ({ticker})', 'phase': 'round1'})}\n\n"

            # Emitir typing para los 3 agentes simultáneamente
            for key in ('market', 'news', 'reddit'):
                ag = AGENT_DEFS[key]
                yield f"data: {_json.dumps({'type': 'typing', 'agent': key, 'name': ag['name'], 'color': ag['color']})}\n\n"

            # Ejecutar en paralelo
            from agents.debate_system import ROUND1_PROMPTS
            result_queue = _queue.Queue()

            def run_and_enqueue(key):
                agent_obj = getattr(system, f'_{key}_agent')
                prompt = ctx + ROUND1_PROMPTS[key].format(ticker=ticker)
                text = system._run_agent_safe(agent_obj, prompt, key)
                result_queue.put((key, text))

            threads = [threading.Thread(target=run_and_enqueue, args=(k,)) for k in ('market', 'news', 'reddit')]
            for t in threads: t.start()

            round1_raw = {}
            for _ in range(3):
                key, text = result_queue.get(timeout=200)
                round1_raw[key] = text
                ag = AGENT_DEFS[key]
                yield f"data: {_json.dumps({'type': 'agent_result', 'agent': key, 'name': ag['name'], 'color': ag['color'], 'icon': ag['icon'], 'text': text, 'round': 1})}\n\n"

            for t in threads: t.join()

            round1_parsed = {k: parse_agent_output(round1_raw[k], k) for k in ('market', 'news', 'reddit')}
            round1 = {'raw': round1_raw, 'parsed': round1_parsed}

            # ── Rondas de debate ──────────────────────────────────────────────
            prev_round = round1
            for r in range(debate_rounds):
                round_num = r + 2
                yield f"data: {_json.dumps({'type': 'status', 'message': f'🗣️ Ronda {round_num} — Debate entre agentes', 'phase': f'debate_{r+1}'})}\n\n"
                prev_round = system._round2_debate(ticker, ctx, prev_round)
                for key in ('market', 'news', 'reddit'):
                    ag = AGENT_DEFS[key]
                    yield f"data: {_json.dumps({'type': 'agent_result', 'agent': key, 'name': ag['name'], 'color': ag['color'], 'icon': ag['icon'], 'text': prev_round['raw'][key], 'round': round_num})}\n\n"

            # ── Consenso ──────────────────────────────────────────────────────
            yield f"data: {_json.dumps({'type': 'status', 'message': '🤝 Alcanzando consenso...', 'phase': 'consensus'})}\n\n"
            consensus = system._round3_consensus(ticker, ctx, round1, prev_round)

            # ── Veredicto ─────────────────────────────────────────────────────
            yield f"data: {_json.dumps({'type': 'status', 'message': '⚖️ El juez está deliberando...', 'phase': 'judge'})}\n\n"
            verdict = system._judge_verdict(ticker, start_date, end_date, round1, prev_round, consensus)

            yield f"data: {_json.dumps({'type': 'verdict', 'text': verdict})}\n\n"
            yield f"data: {_json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            import traceback
            yield f"data: {_json.dumps({'type': 'error', 'message': str(e), 'trace': traceback.format_exc()[-500:]})}\n\n"

    return app.response_class(stream_with_context(generate()), mimetype='text/event-stream',
                              headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/debate', methods=['POST'])
def debate():
    """Endpoint para el sistema de debate multi-agente"""
    try:
        data = request.json
        ticker = data.get('ticker', 'NVDA')
        start_date = data.get('start_date') or None
        end_date = data.get('end_date') or None
        debate_rounds = data.get('rounds', 1)

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from agents.debate_system import DebateSystem

        system = DebateSystem(verbose=False)
        result = system.run(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            debate_rounds=debate_rounds,
        )

        return jsonify({
            'ticker': result['ticker'],
            'verdict': result['verdict'],
            'round1': result['round1_analysis'],
            'round2': result['round2_debate'],
            'consensus': result['consensus'],
            'started_at': str(result['started_at']),
            'finished_at': str(result['finished_at']),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Obtiene estadísticas calculadas desde financial_data.csv"""
    try:
        fin_csv = Path(__file__).parent.parent.parent / 'data' / 'financial_data.csv'
        df = pd.read_csv(fin_csv)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        closes = df['close'].tail(20).tolist()
        volumes = df['volume'].tail(20).tolist()

        # RSI 14
        gains, losses = [], []
        for i in range(1, min(15, len(closes))):
            d = closes[i] - closes[i-1]
            (gains if d > 0 else losses).append(abs(d))
        avg_gain = sum(gains) / 14 if gains else 0
        avg_loss = sum(losses) / 14 if losses else 0.001
        rsi = round(100 - (100 / (1 + avg_gain / avg_loss)), 2)

        # MACD simplificado (EMA12 - EMA26 sobre últimos 26 días)
        c26 = df['close'].tail(26).tolist()
        def ema(data, n):
            k = 2 / (n + 1)
            e = data[0]
            for v in data[1:]:
                e = v * k + e * (1 - k)
            return e
        macd = round(ema(c26, 12) - ema(c26, 26), 4) if len(c26) >= 26 else 0.0

        # Volume ratio (último vs media 20d)
        vol_ratio = round(volumes[-1] / (sum(volumes[:-1]) / max(len(volumes)-1, 1)), 3) if len(volumes) > 1 else 1.0

        # Volatilidad 20d (std de retornos diarios)
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        import math
        volatility = round(math.sqrt(sum(r**2 for r in returns) / len(returns)) * 100, 4) if returns else 0.0

        return jsonify({
            'rsi': rsi,
            'macd': macd,
            'volume_ratio': vol_ratio,
            'volatility': volatility
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/backtest/strategies', methods=['GET'])
def get_backtest_strategies():
    """Obtiene resultados de las estrategias de backtesting"""
    try:
        import json
        
        # Path al archivo de comparación de estrategias
        backtest_dir = Path(__file__).parent.parent.parent / "Modulo7_Backtesting" / "results"
        comparison_file = backtest_dir / "strategies_comparison.json"
        
        if not comparison_file.exists():
            return jsonify({'error': 'Backtest results not found. Run backtester.py first.'}), 404
        
        # Leer resultados
        with open(comparison_file, 'r') as f:
            data = json.load(f)
        
        strategies = data.get('strategies', [])
        
        # Formatear para el frontend
        formatted_strategies = []
        for strategy in strategies:
            formatted_strategies.append({
                'name': strategy['strategy_name'],
                'initial_cash': strategy['initial_cash'],
                'final_value': strategy['final_value'],
                'total_return': strategy['total_return'],
                'sharpe_ratio': strategy['sharpe_ratio'],
                'max_drawdown': strategy['max_drawdown'],
                'total_trades': strategy['total_trades'],
                'won_trades': strategy['won_trades'],
                'lost_trades': strategy['lost_trades'],
                'win_rate': strategy['win_rate'],
                'avg_win': strategy.get('avg_win', 0),
                'avg_loss': strategy.get('avg_loss', 0),
                'params': strategy.get('params', {})
            })
        
        return jsonify({
            'timestamp': data.get('timestamp'),
            'strategies': formatted_strategies
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/backtest/trades/<strategy_name>', methods=['GET'])
def get_strategy_trades(strategy_name):
    """Obtiene los trades de una estrategia específica"""
    try:
        backtest_dir = Path(__file__).parent.parent.parent / "Modulo7_Backtesting" / "results"
        trades_file = backtest_dir / f"{strategy_name}_trades.csv"
        
        if not trades_file.exists():
            return jsonify({'error': f'Trades file for {strategy_name} not found'}), 404
        
        df = pd.read_csv(trades_file)
        trades = df.to_dict('records')
        
        return jsonify({'trades': trades})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/user/trades', methods=['GET'])
def get_user_trades():
    """Obtiene los trades del usuario desde user_trades.csv"""
    try:
        frontend_dir = Path(__file__).parent.parent
        trades_file = frontend_dir / "user_trades.csv"
        
        if not trades_file.exists():
            return jsonify({'error': 'User trades file not found'}), 404
        
        df = pd.read_csv(trades_file)
        df['date'] = pd.to_datetime(df['date'])
        # IMPORTANTE: procesar en orden cronológico para que el cálculo sea correcto
        df = df.sort_values('date', ascending=True)

        # Calcular estadísticas con método FIFO / coste medio
        cash = 0.0          # cash neto: empieza en 0, sube con ventas, baja con compras
        current_shares = 0
        total_cost_basis = 0.0  # coste total de las acciones en cartera
        realized_profit = 0.0

        for _, trade in df.iterrows():
            qty = int(trade['quantity'])
            total_val = float(trade['total'])
            if trade['type'] == 'BUY':
                cash -= total_val
                current_shares += qty
                total_cost_basis += total_val
            else:  # SELL
                avg_cost = total_cost_basis / current_shares if current_shares > 0 else 0
                cost_of_sold = avg_cost * qty
                realized_profit += total_val - cost_of_sold
                total_cost_basis -= cost_of_sold
                current_shares -= qty
                cash += total_val

        # Obtener precio actual de NVDA
        try:
            fin_csv = Path(__file__).parent.parent.parent / 'data' / 'financial_data.csv'
            price_df = pd.read_csv(fin_csv)
            price_df['date'] = pd.to_datetime(price_df['date'])
            current_price = float(price_df.sort_values('date').iloc[-1]['close'])
        except:
            current_price = 110.25

        current_value = current_shares * current_price
        # total_invested = coste de las acciones que aún tenemos en cartera
        total_invested = max(total_cost_basis, 0.0)
        # Capital inicial implícito = total comprado históricamente
        total_spent = float(df[df['type'] == 'BUY']['total'].sum())
        total_received = float(df[df['type'] == 'SELL']['total'].sum())
        # available_cash = lo que queda en efectivo tras todas las operaciones
        # = capital_inicial + ventas - compras, donde capital_inicial = total_spent (asumimos que
        # el usuario siempre tenía suficiente para cada compra)
        available_cash = max(total_received - (total_spent - total_invested), 0.0)
        total_portfolio_value = current_value + available_cash
        total_profit = realized_profit + (current_value - total_cost_basis)

        # Reordenar para mostrar al frontend (más recientes primero)
        df = df.sort_values('date', ascending=False)
        
        trades_list = []
        for _, trade in df.iterrows():
            trades_list.append({
                'date': trade['date'].strftime('%Y-%m-%d'),
                'type': trade['type'],
                'ticker': trade['ticker'],
                'price': float(trade['price']),
                'quantity': int(trade['quantity']),
                'total': float(trade['total']),
                'notes': str(trade['notes'])
            })
        
        return jsonify({
            'trades': trades_list,
            'stats': {
                'total_invested': round(total_invested, 2),
                'available_cash': round(available_cash, 2),
                'current_shares': int(current_shares),
                'current_price': round(current_price, 2),
                'current_value': round(current_value, 2),
                'total_profit': round(total_profit, 2),
                'total_portfolio_value': round(total_portfolio_value, 2),
                'return_percent': round((total_profit / total_spent) * 100, 2) if total_spent > 0 else 0
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print()
    print("Endpoints disponibles:")
    print("  GET  /api/stock/current   - Precio actual")
    print("  GET  /api/stock/history   - Histórico (gráfico)")
    print("  GET  /api/news            - Noticias")
    print("  GET  /api/forecast        - Predicción")
    print("  POST /api/chat            - Chatbot")
    print("  GET  /api/stats           - Estadísticas")
    print()
    print("Servidor corriendo en: http://localhost:5000")
    print()
    app.run(debug=True, port=5000, use_reloader=False)
