import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from openai import OpenAI
import os
from supabase import create_client
import pytz
import json #GPTでの分析の際にJson化させるため記載

# ページ設定
st.set_page_config(
    page_title="ベビーケア ダッシュボード",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="collapsed" #collapsed:折りたたみ expanded:展開
)

# カスタムCSS（レスポンシブ対応 + デスクトップ1画面表示）
st.markdown("""
<style>
    /* デスクトップで1画面表示のためのメインコンテナ */
    .stApp {
        height: 100vh;
        overflow: hidden;
    }
    
    .main .block-container{
        padding-top: 1rem; 
        padding-bottom: 1rem; 
        max-height: 100vh; 
        overflow-y: auto;
    }

    .main-header {
        font-size: clamp(1.5rem, 3vh, 2.2rem);
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: clamp(1rem, 2vh, 1.5rem);
    }
    
    /* ヘッダーと水平線の間隔を調整 */
    /*.stApp h1 + hr {
        margin-top: -1.5rem !important;
    }:*/

    /* デスクトップ用（769px以上）：1画面表示 */
    @media (min-width: 769px) {
        .metric-card, .chart-card, .log-card, {
            
            padding: 1rem;
            border-radius: 15px;
            
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 0.8vh;
            height: calc(40vh - 2rem);
            display: flex;
            flex-direction: column;
            justify-content: center;
            
        }
        
        
        
        .card-title {
            
            font-size: 1.2rem;
            margin-bottom: 0.5rem;
            text-align: center;
            font-weight: bold;
            flex-shrink: 0;
        }
        
        .chart-container {
            flex: 1;
            display: flex;
            align-items: center;
            min-height: 0;
        }
        
        .log-content {
            flex: 1;
            overflow-y: auto;
            padding-right: 0.5rem;
        }
        
        .chart-label {
            text-align: center;
            font-size: 11px;
            opacity: 0.8;
            margin-top: 0.3rem;
            flex-shrink: 0;
        }
    }
    
    .log-item {
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
        padding: 0.4rem;
        background: rgba(255,255,255,0.1);
        border-radius: 6px;
        line-height: 1.3;
        text-align: center;
    }

    /* タブレット対応 */
    @media (max-width: 768px) and (min-width: 481px) {
        .stApp {
            height: auto;
            overflow: visible;
        }
        
        .main .block-container {
            max-height: none;
            overflow-y: visible;
            padding: 1rem 0.5rem;
        }
        
        .metric-card, .chart-card, .log-card, {
            
            padding: 1.2rem;
            border-radius: 18px;
            
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            min-height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            
        }
        
        
        
        .card-title {
            
            font-size: 1.3rem;
            margin-bottom: 1rem;
            text-align: center;
            font-weight: bold;
        }
        
        .chart-label {
            font-size: 12px;
        }
        
        .log-item {
            font-size: 0.9rem;
            padding: 0.5rem;
            margin-bottom: 0.7rem;
            text-align: center;
        }
    }
    
    /* スマートフォン対応 */
    @media (max-width: 480px) {
        .stApp {
            height: auto;
            overflow: visible;
        }
        
        .main .block-container {
            max-height: none;
            overflow-y: visible;
            padding: 0.5rem;
        }
        
        .main-header {
            font-size: 1.8rem;
            margin-bottom: 1rem;
        }
        
        .metric-card, .chart-card, .log-card, {
           
            padding: 1rem;
            border-radius: 15px;
            
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 0.8rem;
            min-height: 160px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            
        }
        
        
        
        .card-title {
           
            font-size: 1.1rem;
            margin-bottom: 0.8rem;
            text-align: center;
            font-weight: bold;
        }
        
        .chart-label {
            font-size: 10px;
        }
        
        .log-item {
            font-size: 0.75rem;
            padding: 0.4rem;
            margin-bottom: 0.5rem;
            text-align: center;
        }
    }
</style>
""", unsafe_allow_html=True)

#---------------------------------------------------------
# セキュリティ対応
#---------------------------------------------------------
#.env 読み込み（無ければ何もしない）
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

#---------------------------------------------------------
# OpenAI APIキー関連
#---------------------------------------------------------
# OpenAI APIキーの取得
def get_api_key(env_key: str = "OPENAI_API_KEY") -> str | None:
    key = os.getenv(env_key)
    if key:
        return key
    try:
        return st.secrets[env_key]  # secrets.toml が無い環境でも例外安全に
    except Exception:
        return None

API_KEY = get_api_key()
if not API_KEY:
    st.error(
        "OpenAI APIキーが見つかりません。\n\n"
        "■ 推奨（ローカル学習向け）\n"
        "  1) .env を作成し OPENAI_API_KEY=sk-xxxx を記載\n"
        "  2) このアプリを再実行\n\n"
        "■ 参考（secrets を使う場合）\n"
        "  .streamlit/secrets.toml に OPENAI_API_KEY を記載（※リポジトリにコミットしない）\n"
        "  公式: st.secrets / secrets.toml の使い方はドキュメント参照"
    )
    st.stop()

client = OpenAI(api_key=API_KEY)

#---------------------------------------------------------
#ChatGPTによる回答生成
#---------------------------------------------------------
#2025.9.22関
#SYSTEM_PROMPTを記載してsystemメッセージ(役割設定)を一元管理
SYSTEM_PROMPT = """\
あなたは小児看護・育児の実務知識を持つアシスタントです。
- 口調: 親身でやさしい丁寧語。断定は避け、根拠を短く添える。
- 安全: 危険兆候・受診目安はあれば必ず明示（不安を煽りすぎない）。
- 出力: 見出し→箇条書き→最後に「次の一歩」を1~3個。
- 制約: 医療行為・診断はしない。専門家の診断を促す。
"""
#2025.9.22関
#FORMAT_HINTを指定し、後続でユーザー質問と一緒にモデルに渡すことで回答の構造を誘導
FORMAT_HINT = """\
# 形式
## 要点
- 3~5点で簡潔に
## 補足
- 各項目を1~2文で
## 次の一歩
- 1~3個の具体的行動
"""
def get_chat_response(
    user_query: str,
    system_prompt: str = SYSTEM_PROMPT,
    format_hint: str = FORMAT_HINT,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> str:
    if not client.api_key:
        return "APIキーが設定されていません。"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_query}\n\n{format_hint}"},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"エラーが発生しました: {e}"

#---------------------------------------------------------
# Supabase APIキー関連
#---------------------------------------------------------
# SupabaseのURLとAPIキーの取得
def get_supabase_info():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return url, key

# Supabaseの情報を取得し、存在しない場合はエラーを表示して停止
supabase_url, supabase_key = get_supabase_info()
if not supabase_url or not supabase_key:
    st.error(
        "SupabaseのURLとキーが見つかりません。"
        "\n\n.envファイルに SUPABASE_URL=\"...\" と SUPABASE_KEY=\"...\" を記載してください。"
    )
    st.stop()

#supabaseクライアントの初期化
supabase_client = create_client(supabase_url, supabase_key)


# ---------------------------------------------------------
# spabaseからおむつ経過時間計算＜カード1＞
# ---------------------------------------------------------
@st.cache_data(ttl=60) # 1分間キャッシュ
def get_diaper_elapsed_time(table_name="baby_events"):
    """
    Supabaseから最新の「おしっこ」または「うんち」のイベント時刻を取得し、
    現在時刻からの経過時間（分）を計算する。
    """
    try:
        # type_slugが 'diaper_pee' (おしっこ) または 'diaper_poo' (うんち) の最新ログを1件取得
        response = supabase_client.table(table_name).select("datetime, type_slug").in_('type_slug', ['diaper_pee', 'diaper_poo']).order("datetime", desc=True).limit(1).execute()
        
        if response.data:
            latest_diaper_log = response.data[0]
            # 1. ログ時刻をタイムゾーン付きで読み込み、JSTに変換する
            log_time_utc = datetime.fromisoformat(latest_diaper_log['datetime'])
            # タイムゾーン情報がない場合は、ここでUTCを付与する必要があるが、
            # Supabaseから返されるデータは通常タイムゾーン付きなので、tz_convertのみでOK
            log_time_jst = log_time_utc.astimezone(pytz.timezone('Asia/Tokyo'))
            
            # 2. 現在時刻をJSTで取得する (pytzが必要)
            current_time_jst = datetime.now(pytz.timezone('Asia/Tokyo'))
            
            # 3. JST同士で経過時間を計算
            delta = current_time_jst - log_time_jst
            minutes_passed = int(delta.total_seconds() / 60)
            
            # 経過時間を返す
            return minutes_passed
        else:
            # データがない場合は0分を返す
            return 0
    except Exception as e:
        st.error(f"おむつデータの読み込み中にエラーが発生しました: {e}")
        return 0

# ---------------------------------------------------------
# spabaseから睡眠時間の日ごとの累計値と前週平均の計算＜カード2＞
# ---------------------------------------------------------
@st.cache_data(ttl=60) # 1分間キャッシュ
def get_sleep_summary_data(table_name="baby_events"):
    """
    Supabaseから直近2週間分の睡眠イベントを取得し、
    日ごとの睡眠時間累計（14日間）と前週の平均値を計算して返す。
    """
    try:
        # 1. データの取得
        # 直近14日間のイベントだけだと、期間の開始前のsleep_startが欠ける可能性があるため、
        # 余裕を持って過去15日間のデータを取得します。
        fifteen_days_ago = datetime.now() - timedelta(days=15)
        
        # type_slugが 'sleep_start' または 'sleep_end' のログを取得
        response = supabase_client.table(table_name).select("datetime, type_slug").in_('type_slug', ['sleep_start', 'sleep_end']).gte('datetime', fifteen_days_ago.isoformat()).order("datetime", desc=False).execute()
        
        if not response.data:
            # データがない場合のダミーデータ（14日間）
            dates_14 = [datetime.now().date() - timedelta(days=i) for i in range(13, -1, -1)]
            df_display = pd.DataFrame({'date': dates_14, 'count': [0.0] * 14})
            return df_display, 0.0

        df = pd.DataFrame(response.data)
        # タイムゾーン変換
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['datetime'] = df['datetime'].dt.tz_convert('Asia/Tokyo')
        df['date'] = df['datetime'].dt.date 
        
        # 2. 睡眠時間の計算 (sleep_start から sleep_end までのペアを見つける)
        sleep_durations = []
        i = 0
        while i < len(df) - 1:
            start_row = df.iloc[i]
            end_row = df.iloc[i+1]
            
            # sleep_start から始まり、直後に sleep_end が続く場合のみ計算
            if start_row['type_slug'] == 'sleep_start' and end_row['type_slug'] == 'sleep_end':
                # 睡眠時間（時間単位）を計算
                duration_hours = (end_row['datetime'] - start_row['datetime']).total_seconds() / 3600
                
                # 睡眠終了時の日付をキーとして保存
                sleep_durations.append({
                    'date': end_row['datetime'].date(), 
                    'duration_hours': duration_hours
                })
                i += 2 # 次のペアへ
            else:
                # 'sleep_start' の次が 'sleep_start' (ログ抜け) または 'sleep_end' の次が 'sleep_start' ではない場合
                # start_row が 'sleep_start' ではない場合、次の行に進む
                # start_row が 'sleep_start' で end_row が 'sleep_start' の場合、start_rowをスキップして次の行に進む
                i += 1 

        df_durations = pd.DataFrame(sleep_durations)
        
        # 3. 日ごとの累計睡眠時間（時間）を計算
        if df_durations.empty:
            sleep_summary = pd.DataFrame()
        else:
            sleep_summary = df_durations.groupby('date')['duration_hours'].sum().reset_index()
            sleep_summary.columns = ['date', 'count']

        # 4. グラフ表示期間（直近14日間）を定義
        today = datetime.now().date()
        dates_14 = [today - timedelta(days=i) for i in range(13, -1, -1)]
        
        # 5. グラフ表示用DataFrameに結合し、データがない日は0とする
        df_display = pd.DataFrame({'date': dates_14})
        df_display = pd.merge(df_display, sleep_summary, on='date', how='left').fillna(0.0)
        
        # 6. 前週平均値の計算
        start_of_current_period = today - timedelta(days=6) # 直近7日間の開始日
        start_of_last_period = start_of_current_period - timedelta(days=7) # 前週7日間の開始日
        
        # 前7日間 (前週扱い) のデータのみを抽出
        df_last_period_summary = sleep_summary[(sleep_summary['date'] < start_of_current_period) & (sleep_summary['date'] >= start_of_last_period)]
        
        # 前週の平均値（日ごとの累計睡眠時間の平均）
        last_week_average = df_last_period_summary['count'].mean() if not df_last_period_summary.empty else 0.0
        
        # 7. 日付を「月/日」形式の文字列に変換 (PlotlyのX軸表示を確実にするため)
        df_display['date'] = df_display['date'].apply(lambda x: x.strftime('%m/%d'))
        
        return df_display, last_week_average
        
    except Exception as e:
        st.error(f"睡眠データの集計中にエラーが発生しました: {e}")
        # エラー発生時はダミーデータを返す (14日間)
        dates_14 = [datetime.now().date() - timedelta(days=i) for i in range(13, -1, -1)]
        return pd.DataFrame({'date': dates_14, 'count': [0.0] * 14}), 0.0

#---------------------------------------------------------
#spabaseから最新ログを取得＜カード3＞
#---------------------------------------------------------
def get_supabase_data(table_name="baby_events"):
    """Supabaseからデータを取得する"""
    try:
        response = supabase_client.table(table_name).select("datetime, type_jp").order("datetime", desc=True).limit(3).execute()
        
        # データをDataFrameに変換
        df = pd.DataFrame(response.data)
        
        # 'datetime' 列を希望の形式にフォーマット
        df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d %H:%M')
        
        # DataFrameを辞書リストに戻す（st.dataframeにそのまま渡せる）
        return df.to_dict('records')
    except Exception as e:
        st.error(f"データベースの読み込み中にエラーが発生しました: {e}")
        return []


# ---------------------------------------------------------
# spabaseから授乳経過時間計算＜カード4＞
# ---------------------------------------------------------
@st.cache_data(ttl=60) # 1分間キャッシュ
def get_feeding_elapsed_time(table_name="baby_events"):
    """
    Supabaseから最新の「授乳」イベント時刻を取得し、
    現在時刻からの経過時間（分）を計算する。
    """
    try:
        # type_slugが 'formula' (ミルク) または 'breast' (母乳)(仮) の最新ログを1件取得
        # baby_eventsテーブルログには'breast' (母乳)は無いので今後必要に応じて修正
        response = supabase_client.table(table_name).select("datetime, type_slug").in_('type_slug', ['formula', 'breast']).order("datetime", desc=True).limit(1).execute()
        
        if response.data:
            latest_feeding_log = response.data[0]
            # datetimeをISO 8601形式からタイムゾーン情報付きのdatetimeオブジェクトに変換
            log_time = datetime.fromisoformat(latest_feeding_log['datetime'].replace('Z', '+00:00'))
            
            # 現在時刻も同じタイムゾーンに合わせる
            current_time = datetime.now(log_time.tzinfo)
            
            # 経過時間を計算
            delta = current_time - log_time
            minutes_passed = int(delta.total_seconds() / 60)
            
            # 経過時間を返す
            return minutes_passed
        else:
            # データがない場合は0分を返す
            return 0
    except Exception as e:
        st.error(f"授乳データの読み込み中にエラーが発生しました: {e}")
        return 0

# ---------------------------------------------------------
# spabaseからミルク量の日ごとの累計値と前週平均の計算＜カード5＞
# ---------------------------------------------------------
@st.cache_data(ttl=60) # 1分間キャッシュ
def get_feeding_summary_data(table_name="baby_events"):
    """
    Supabaseから直近2週間分のミルク量データを取得し、
    日ごとの累計値（14日間）と前週の平均値を計算して返す。
    """
    try:
        # 直近14日間のデータを取得（今週7日 + 前週7日）
        fourteen_days_ago = datetime.now() - timedelta(days=14)
        
        # type_slugが 'formula' のログを取得
        response = supabase_client.table(table_name).select("datetime, amount_ml, type_slug").eq('type_slug', 'formula').gte('datetime', fourteen_days_ago.isoformat()).order("datetime", desc=True).execute()
        
        if not response.data:
            # グラフに表示するためのダミーデータ（14日間）を生成
            dates_14 = [datetime.now().date() - timedelta(days=i) for i in range(13, -1, -1)]
            df_display = pd.DataFrame({'date': dates_14, 'amount': [0] * 14})
            return df_display, 0

        df = pd.DataFrame(response.data)
        # タイムゾーン変換後にtz-awareを削除
        df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_convert('Asia/Tokyo').dt.tz_localize(None) 
        df['date'] = df['datetime'].dt.date
        df['amount_ml'] = pd.to_numeric(df['amount_ml'], errors='coerce').fillna(0)
        
        # 期間の定義
        today = datetime.now().date()
        
        # 1. 表示する日付（直近14日間）のリストを作成
        dates_14 = [today - timedelta(days=i) for i in range(13, -1, -1)]
        
        # 2. 直近14日間の日ごとの累計値を計算
        all_period_summary = df.groupby('date')['amount_ml'].sum().reset_index()
        all_period_summary.columns = ['date', 'amount']
        
        # 3. 直近14日間を表示用のDataFrameに結合し、データがない日は0とする
        df_display = pd.DataFrame({'date': dates_14})
        df_display = pd.merge(df_display, all_period_summary, on='date', how='left').fillna(0)
        
        # 4. 前週の平均値（前7日間）を計算
        start_of_current_period = today - timedelta(days=6) # 直近7日間の開始日
        start_of_last_period = start_of_current_period - timedelta(days=7) # 前週7日間の開始日
        
        # 前7日間 (前週扱い) のデータのみを抽出
        df_last_period = df[(df['date'] < start_of_current_period) & (df['date'] >= start_of_last_period)]
        
        # 前7日間 (前週扱い) の日ごとの累計を計算
        last_period_summary = df_last_period.groupby('date')['amount_ml'].sum()
        
        # 前週の平均値（日ごとの累計値の平均）
        last_week_average = last_period_summary.mean() if not last_period_summary.empty else 0
        
        # create_bar_chartの形式に合わせて列名を修正
        df_display.columns = ['date', 'amount']
        df_display['date'] = df_display['date'].apply(lambda x: x.strftime('%m/%d'))
        
        return df_display, last_week_average
        
    except Exception as e:
        st.error(f"ミルク量データの集計中にエラーが発生しました: {e}")
        # エラー発生時はダミーデータを返す (14日間)
        dates_14 = [datetime.now().date() - timedelta(days=i) for i in range(13, -1, -1)]
        return pd.DataFrame({'date': dates_14, 'amount': [0] * 14}), 0

# ---------------------------------------------------------
# 既存KPIから派生統計を計算 → 日常語ラベル化（色バッジは使わない）
# （DB集計関数の直後に置く：グラフ関数の前）
# ---------------------------------------------------------
def _series_stats(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "trend_slope_per_day": 0.0}
    x = np.arange(arr.size, dtype=float)
    slope = float(np.polyfit(x, arr, 1)[0]) if arr.size >= 2 else 0.0
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "trend_slope_per_day": slope,
    }

def _qualitative_labels(mean: float, std: float, slope: float, unit: str,
                        abs_threshold: float | None = None) -> dict:
    """
    統計を“日常語”に変換（色やバッジは使わない）。
    - 変動（CV=std/mean）: 『ほぼ毎日おなじ / 日によって少しちがう / 日によってかなりちがう』
      目安幅も一緒に返す（±10%/±25% を実数化）
    - 傾き: 1日あたり平均比±5%を目安に『少し増えつつある / 少し減りつつある / だいたい同じ』
    """
    eps = 1e-9
    cv = std / (abs(mean) + eps)
    band10 = abs(mean) * 0.10
    band25 = abs(mean) * 0.25

    if cv < 0.10:
        variability = "ほぼ毎日おなじ"
        variability_phrase = f"日ごとの差は小さめ（目安: ±{band10:.1f}{unit}以内）。"
    elif cv < 0.25:
        variability = "日によって少しちがう"
        variability_phrase = f"日ごとの差は中くらい（目安: ±{band10:.1f}〜±{band25:.1f}{unit}）。"
    else:
        variability = "日によってかなりちがう"
        variability_phrase = f"日ごとの差は大きめ（目安: ±{band25:.1f}{unit}以上）。"

    rel = abs(slope) / (abs(mean) + eps) if mean else 0.0
    use_abs = abs_threshold is not None and abs(slope) >= abs_threshold
    use_rel = rel >= 0.05  # 5%/日を目安

    if (slope > 0) and (use_abs or use_rel):
        trend = "少し増えつつある"
        trend_phrase = f"ここ数日は{unit}がゆるやかに増えています。"
    elif (slope < 0) and (use_abs or use_rel):
        trend = "少し減りつつある"
        trend_phrase = f"ここ数日は{unit}がゆるやかに減っています。"
    else:
        trend = "だいたい同じ"
        trend_phrase = f"ここ数日は{unit}は大きく変わっていません。"

    return {
        "variability": variability,
        "variability_phrase": variability_phrase,
        "trend": trend,
        "trend_phrase": trend_phrase,
        "guideline_band_10pct": band10,
        "guideline_band_25pct": band25,
    }
# ---------------------------------------------------------
# GPTプロンプト組み立て（KPI_JSON同梱）と質問別インストラクション・共通呼び出し
# ---------------------------------------------------------
def build_kpi_payload_for_gpt() -> dict:
    # 既存の集計関数を再利用（ダッシュボードと同条件）
    sleep_chart_data, last_week_avg_sleep = get_sleep_summary_data(table_name="baby_events")
    feeding_chart_data, last_week_avg_amount = get_feeding_summary_data(table_name="baby_events")

    diaper_elapsed = get_diaper_elapsed_time(table_name="baby_events")
    feeding_elapsed = get_feeding_elapsed_time(table_name="baby_events")
    if isinstance(diaper_elapsed, tuple): _, diaper_elapsed = diaper_elapsed
    if isinstance(feeding_elapsed, tuple): _, feeding_elapsed = feeding_elapsed

    sleep_df = pd.DataFrame(sleep_chart_data).tail(7)
    feed_df  = pd.DataFrame(feeding_chart_data).tail(7)
    sleep_val = sleep_df.columns[1] if len(sleep_df.columns) > 1 else None
    feed_val  = feed_df.columns[1]  if len(feed_df.columns)  > 1 else None

    sleep_vals = sleep_df[sleep_val].fillna(0).astype(float).tolist() if sleep_val else []
    milk_vals  = feed_df[feed_val].fillna(0).astype(float).tolist() if feed_val else []

    sleep_stats = _series_stats(sleep_vals)
    milk_stats  = _series_stats(milk_vals)

    # “日常語”ラベル（睡眠は0.2h/日、ミルクは20ml/日を絶対閾値の目安）
    sleep_labels = _qualitative_labels(
        mean=sleep_stats["mean"], std=sleep_stats["std"], slope=sleep_stats["trend_slope_per_day"],
        unit="時間/日", abs_threshold=0.2
    )
    milk_labels = _qualitative_labels(
        mean=milk_stats["mean"], std=milk_stats["std"], slope=milk_stats["trend_slope_per_day"],
        unit="ml/日", abs_threshold=20.0
    )

    def bucket_minutes(m: int) -> str:
        if m is None: return "unknown"
        return "0-90" if m < 90 else "90-180" if m < 180 else "180+"

    return {
        "units": {
            "sleep_hours_per_day": "hours",
            "milk_amount_per_day": "ml",
            "elapsed_since_diaper": "minutes",
            "elapsed_since_feeding": "minutes",
        },
        "elapsed": {
            "diaper_minutes": int(diaper_elapsed or 0),
            "feeding_minutes": int(feeding_elapsed or 0),
            "diaper_bucket": bucket_minutes(int(diaper_elapsed or 0)),
            "feeding_bucket": bucket_minutes(int(feeding_elapsed or 0)),
        },
        "sleep_last7": [
            {"date": str(r["date"]), "hours": float(r[sleep_val] or 0)}
            for _, r in sleep_df.iterrows()
        ],
        "sleep_prev_week_avg_hours": float(round(float(last_week_avg_sleep or 0), 2)),
        "sleep_last7_stats": sleep_stats,
        "sleep_last7_labels": sleep_labels,   # ← 日常語ラベル（色情報なし）
        "milk_last7": [
            {"date": str(r["date"]), "ml": float(r[feed_val] or 0)}
            for _, r in feed_df.iterrows()
        ],
        "milk_prev_week_avg_ml": float(round(float(last_week_avg_amount or 0), 2)),
        "milk_last7_stats": milk_stats,
        "milk_last7_labels": milk_labels,     # ← 日常語ラベル（色情報なし）
        "notes": "Derived stats and plain-language labels are computed on last7 only.",
    }

def build_analysis_instruction(question: str) -> str:
    # ※ 統計用語や追加ログ要求を出さない運用
    common = (
        "※ 追加のデータ収集や新たなログの提案は行わないでください。"
        "KPI_JSONの範囲だけで分析し、実行負荷の小さい『低後悔』アクションのみ提案してください。"
        "専門用語（標準偏差・相関・傾き など）は使わず、"
        "KPI_JSONに含まれる『…_labels』の短い言い回し（例: ほぼ毎日おなじ/日によって少しちがう/日によってかなりちがう、少し増えつつある/少し減りつつある/だいたい同じ）で説明してください。"
    )
    if "睡眠パターン" in question:
        return (
            "直近7日の睡眠合計（hours/day）と前週平均の差、ムラの大きさ、最近の流れを評価し、"
            "就寝時間の固定や就寝前ルーティンなど、低負荷の対策を提案してください。"
            + common
        )
    if "授乳間隔" in question:
        return (
            "厳密な間隔は算出せず、『授乳からの経過分』とミルク量の推移/ムラ/最近の流れから、"
            "保守的に過剰/不足の兆候を評価してください。"
            + common
        )
    if "ミルク量" in question:
        return (
            "直近7日のミルク量と前週平均の差、ムラの大きさ、最近の流れを評価し、"
            "配分の工夫など低負荷の提案を行ってください。"
            + common
        )
    if "おむつ替え" in question:
        return (
            "『おむつからの経過分』を主指標に替えタイミングの妥当性を評価し、"
            "外出前チェックや最大間隔の目安など低負荷の運用を示してください。"
            + common
        )
    return "KPI_JSONに基づく分析と、低負荷なNext Actionのみを提示してください。" + common

def ask_gpt_with_optional_kpi(user_question: str, include_kpi: bool = True) -> str:
    """
    SYSTEM_PROMPT / FORMAT_HINT は既存 get_chat_response のデフォルトで踏襲。
    include_kpi=True なら KPI_JSON を同梱。
    """
    parts = []
    parts.append("以下のユーザー質問に回答し、その後で与えられたKPI_JSON（あれば）を一次ソースとして事実ベースの分析と示唆を述べてください。")
    parts.append("\n[ユーザー質問]\n" + user_question)

    if include_kpi:
        kpi_json = json.dumps(build_kpi_payload_for_gpt(), ensure_ascii=False)
        instruction = build_analysis_instruction(user_question)
        parts.append("\n[分析タスク]\n" + instruction)
        parts.append("\n[KPI_JSON]\n" + kpi_json)

    parts.append("\n出力フォーマットは指定の形式（SYSTEM/FORMAT_HINT）に従ってください。")
    prompt = "\n".join(parts)
    return get_chat_response(prompt)  # 既存のSYSTEM_PROMPT/FORMAT_HINTが効く


#---------------------------------------------------------
# データ生成・グラフ作成
#---------------------------------------------------------

# 円形プログレスバーの作成（レスポンシブ対応）＜カード1・4＞
def create_circular_progress(actual_value, max_value, color="#FF6B47"):
    """
    円形プログレスバーを作成し、中央に値を表示する
    
    Args:
        actual_value (int): 実際の経過時間（分）。中央に表示される値。
        max_value (int): グラフの上限（分）。プログレスバーが一周する値。
        color (str): プログレスバーの色。
    
    Returns:
        go.Figure: PlotlyのFigureオブジェクト。
    """
    
    # グラフのオレンジ色の領域として表示する値。最大値を超えないように制限する。
    display_value = min(actual_value, max_value)

    #円形プログレスバー作成
    fig = go.Figure(data=[go.Pie(
        values=[display_value, max_value - display_value],
        hole=.7,
        marker_colors=[color, '#d3d3d3'],
        textinfo='none',
        showlegend=False,
        hoverinfo='skip',
        direction='clockwise', # 時計回り 
        sort=False, 
        #rotation=90　←最初から12時の方向に開始されるので不要
    )])
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=0, b=0, l=0, r=0),
        autosize=True,
        height=150,  # 高さを半分に設定
        annotations=[
            dict(
                # 実際の経過時間 (actual_value) を表示
                text=f'<span style="color:black; font-size:30px; font-weight:bold;">{actual_value}</span><br><span style="color:black; font-size:16px;"><br>分経過</span>',
                x=0.5, 
                y=0.5, 
                showarrow=False,
                align='center'
            )
        ]
    )
    
    return fig

# 棒グラフの作成（デスクトップ1画面対応）＜カード2・5＞
def create_bar_chart(data, title, color="#4A90E2", average_value=None): 
    df = pd.DataFrame(data)

    # DataFrameの2列目（index 1）をデータの値の列とする
    value_column = df.columns[1] 
    
    # 日付列を 'date' に統一する (get_feeding_summary_dataが出力する形式に合わせる)
    if 'date' not in df.columns:
        df.columns = ['date', value_column]
    
    num_days = len(df)
    x_range_indices = None
    
    # データが7日分以上ある場合、直近7日間の範囲を設定する
    if num_days >= 7:
        # PlotlyはX軸をカテゴリカルデータとして扱うため、インデックスで範囲を指定する。
        # 直近7日間はインデックスの (num_days - 7) から (num_days - 1) に対応。
        # グラフの棒が途切れないように、開始と終了のインデックスに +/- 0.5 の調整を加える。
        x_range_indices = [num_days - 7 - 0.5, num_days - 1 + 0.5]

    # 棒グラフの作成
    bar_fig = go.Figure(data=[
        go.Bar(
            x=df['date'],
            y=df[value_column], 
            text=df[value_column].apply(lambda x: f'{int(x)}' if x > 0 else ''), # 0は表示しない
            marker_color=color,
            textposition='inside',
            insidetextanchor='end',
            marker_cornerradius=3,
            textfont=dict(color='white', size=12),
            showlegend=False # 凡例を非表示にする
        )
    ])
    
    # 前週平均線の作成 (average_value が渡された場合にのみ実行)
    if average_value is not None and average_value > 0:
        # グラフに表示されている日付リストを取得
        dates = df['date'].tolist()
        
        # 直近7日間の日付のみを抽出 (最後の7つ)
        # 前週の平均値を今週の棒グラフ領域に表示したい
        this_week_dates = dates[-7:] # 例: 9/21〜9/27
        
        # 直近7日間の日付に対応する平均値のリストを作成
        # 前週のデータには線を引かないように、直近7日分だけ平均値、残りはNone(Plotlyは無視する)を設定
        y_line = [None] * (len(dates) - 7) + [average_value] * 7
        
        line_fig = go.Figure(data=[
            go.Scatter(
                x=df['date'],
                y=y_line, # ← 14日間のうち直近7日間にのみ平均値を設定
                mode='lines',
                line=dict(color='red', width=2),
                name='前週平均',
                showlegend=False 
            )
        ])
        # 棒グラフと折れ線グラフを統合
        final_fig = go.Figure(data=bar_fig.data + line_fig.data)
    else:
        final_fig = bar_fig # 平均値がない場合は棒グラフのみ
        

    final_fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50', size=10),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            #tickformat='%m/%d',
            title="",
            tickfont=dict(size=9),
            range=x_range_indices,# ★★★ X軸の表示範囲を適用 ★★★
            rangeslider=dict(visible=False), 
            type='category' # X軸をカテゴリカルとして扱う
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            title="",
            tickfont=dict(size=9),
            # Y軸の最大値を調整して平均線が入りやすいようにする (平均値が存在する場合)
            range=[0, df[value_column].max() * 1.1 if average_value is None or df[value_column].max() * 1.1 > average_value * 1.1 else average_value * 1.1]
        ),
        margin=dict(t=5, b=5, l=15, r=15),
        autosize=True,
        height=180
    )
    
    return final_fig


# 「今何してる」のステータスと経過時間を計算する関数＜カード6＞
def get_status_and_time(log_data):
    # 最新のログを取得
    latest_log = max(log_data, key=lambda x: x['time'])
    action = latest_log['action']
    log_time = latest_log['time']
    
    current_time = datetime.now()
    delta = current_time - log_time
    minutes_passed = int(delta.total_seconds() / 60)
    
    # 状態を決定
    status_text = ""
    if "起床" in action:
        status_text = "起床"
    elif "就寝" in action:
        status_text = "就寝"
    else:
        status_text = action.split(',')[0] 
    
    # 表示用の文字列を生成
    time_passed_str = f"{minutes_passed}分経過"
        
    return status_text, time_passed_str, log_time

#---------------------------------------------------------
# メイン画面
#---------------------------------------------------------
def main():
    # ヘッダー
    st.header("ベビーケア ダッシュボード")
    st.markdown("---")

    # カード1用データ取得: 最新のおむつ替えからの経過時間を取得
    elapsed_minutes = get_diaper_elapsed_time(table_name="baby_events")
    DIAPER_MAX_MINUTES = 180 # グラフの上限を180分に設定

    # カード2用データ取得: 睡眠時間の日ごとの累計と前週平均 (新規追加)
    sleep_chart_data, last_week_avg_sleep = get_sleep_summary_data(table_name="baby_events")

    # カード3用データ取得　Supabaseから最新ログデータを取得
    supabase_log_data = get_supabase_data(table_name="baby_events") # テーブル名を編集

    # カード4用データ取得: 最新の授乳からの経過時間を取得
    elapsed_minutes_feeding = get_feeding_elapsed_time(table_name="baby_events")
    FEEDING_MAX_MINUTES = 180 # 授乳グラフの上限を180分（3時間）に設定

    # カード5用データ取得: ミルク量の日ごとの累計と前週平均 (新規追加)
    feeding_chart_data, last_week_avg_amount = get_feeding_summary_data(table_name="baby_events")
    
    # カード6用データ取得　Supabaseから最新の起床or就寝ログを取得
    latest_sleep_log = None
    try:
        # type_slugが 'sleep_start' または 'sleep_end' の最新のログを1件取得
        response = supabase_client.table("baby_events").select("datetime, type_slug").in_('type_slug', ['sleep_start', 'sleep_end']).order("datetime", desc=True).limit(1).execute()
        if response.data:
            latest_sleep_log = response.data[0]
        else:
            st.info("睡眠に関するデータがありません。")
    except Exception as e:
        st.error(f"睡眠データの読み込み中にエラーが発生しました: {e}")
    
    
    
    
    # レスポンシブレイアウト設定
    # デスクトップ: 3列, タブレット: 2列, スマホ: 1列
    if st.session_state.get('mobile_view', False):
        # モバイル表示: 1列6行
        cols = [st.columns(1)[0] for _ in range(6)]
    else:
        # デスクトップ・タブレット表示: 2行3列
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        cols = [row1_col1, row1_col2, row1_col3, row2_col1, row2_col2, row2_col3]
    
    # カード1: おむつ替え経過時間
    with cols[0]:
        st.markdown('<div class="card-title">おむつ替え経過時間</div>', unsafe_allow_html=True)
        # 経過時間と上限値(例：180分)を渡す
        fig_diaper_progress = create_circular_progress(elapsed_minutes, DIAPER_MAX_MINUTES, "#ff8c00")
        st.plotly_chart(fig_diaper_progress, use_container_width=True, config={'displayModeBar': False}, key="diaper_progress")
    
    # カード2: 睡眠時間 前週平均比較
    with cols[1]:
        st.markdown('<div class="card-title">睡眠時間 (h) 前週平均比較</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        fig_sleep_chart = create_bar_chart(sleep_chart_data, "睡眠時間 前週平均比較", "#4A90E2", last_week_avg_sleep)
        st.plotly_chart(fig_sleep_chart, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    
    # カード3: 最新ログ
    with cols[2]:
        st.markdown('<div class="card-title">最新ログ</div>', unsafe_allow_html=True)
        st.markdown(
        """
        <div style="display: flex; flex-direction: column; align-items: center; height: 100%;">
            <div class="log-content">
        """,
        unsafe_allow_html=True
        )
        
        #Supabaseのデータベースを表示
        data = get_supabase_data()
        if data:
            st.dataframe(data)
        else:
            st.info("データがありません。テーブル名を確認してください。")

    
    # カード4: 授乳経過時間
    with cols[3]:
        st.markdown('<div class="card-title">授乳経過時間</div>', unsafe_allow_html=True)
        fig_feeding_progress = create_circular_progress(elapsed_minutes_feeding, FEEDING_MAX_MINUTES, "#ff8c00") 
        st.plotly_chart(fig_feeding_progress, use_container_width=True, config={'displayModeBar': False}, key="feeding_progress")
    
    # カード5: ミルク量 前週平均比較
    with cols[4]:
        st.markdown('<div class="card-title">　ミルク量(ml)　前週平均比較</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # === 修正点: 動的データと前週平均を渡す ===
        fig_feeding_chart = create_bar_chart(feeding_chart_data, "ミルク量  前週平均比較", "#4A90E2", last_week_avg_amount)
        st.plotly_chart(fig_feeding_chart, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
        
    
    # カード6: 現在の起床/睡眠状態
    with cols[5]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">今何してる</div>', unsafe_allow_html=True)
        if latest_sleep_log:
            # datetimeをISO 8601形式からdatetimeオブジェクトに変換
            log_time = datetime.fromisoformat(latest_sleep_log['datetime'].replace('Z', '+00:00'))
            current_time = datetime.now(log_time.tzinfo) # 現在時刻も同じタイムゾーンに合わせる

            # 経過時間を計算
            delta = current_time - log_time
            minutes_passed = int(delta.total_seconds() / 60)

            # 状態と表示テキストを決定
            status_text = ""
            emoji = ""
            if latest_sleep_log['type_slug'] == 'sleep_start':
                status_text = "就寝"
                emoji = "😴"
            elif latest_sleep_log['type_slug'] == 'sleep_end':
                status_text = "起床"
                emoji = "🌞"

            formatted_time_passed = f"{minutes_passed}分経過"

            # HTMLで表示
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div class="time-text">
                        {log_time.strftime('%H:%M')}に{status_text}
                    </div>
                    <div class="time-text">
                        {formatted_time_passed}
                    </div>
                    <div style="font-size: 3rem; margin-top: 1rem;">
                        {emoji}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("睡眠のデータがありません。")
        st.markdown('</div>', unsafe_allow_html=True)

    #質問入力時、AIによる育児アドバイス部分に遷移するようにアンカーを設置。
    # ChatGPTによる回答表示欄
    st.markdown('<div id="advice-anchor"></div>', unsafe_allow_html=True)
    st.header("AIによる育児アドバイス")
    st.markdown("---")

    if 'chat_response' in st.session_state and st.session_state.chat_response: # セッションステートに回答が保存されていれば表示
        st.info(st.session_state.chat_response)
    else:
        st.info("サイドバーから質問を入力してください。")

    if '_last_scrolled' not in st.session_state:
        st.session_state['_last_scrolled'] = 0
    #ボタン押下後にAIによる育児アドバイス部分までスクロールさせる処理
    if st.session_state.get('scroll_trigger', 0) != st.session_state.get('_last_scrolled', 0):
        trig = st.session_state['scroll_trigger']
        st.components.v1.html(
            f"""
            <script>
            (function(){{
                const doc = window.parent.document;
                function go(retry=0){{
                    const anchor = doc.querySelector('#advice-anchor');
                    if(!anchor){{
                        if(retry<100) return setTimeout(()=>go(retry+1),20);
                        return;
                    }}
                    anchor.scrollIntoView({{ block: 'start', behavior: 'auto' }});
                    requestAnimationFrame(()=>{{
                        anchor.scrollIntoView({{ block: 'start', behavior: 'smooth' }});
                    }});
                }}
                setTimeout(()=>go(0), 150);
            }})();
            </script>
            <div data-scroll-trigger="{trig}" style="display:none"></div>
            """,
            height=1

        )
        #消費したトリガーを記録
        st.session_state['_last_scrolled'] = st.session_state['scroll_trigger']

    
#---------------------------------------------------------
# サイドバー（質問・相談機能）
#---------------------------------------------------------
with st.sidebar:
    st.title("ChatGPT 育児相談")
    
    # セッションステートを初期化
    if 'chat_response' not in st.session_state:
        st.session_state.chat_response = ""
    if 'scroll_trigger' not in st.session_state: #スクロールのために追加
        st.session_state.scroll_trigger = 0 #初期化する

    # チャット入力
    user_input = st.text_area("", placeholder="入力してください...", key="chat_input", height=100)
    
    def fire_and_scroll(text: str, include_kpi: bool = True):
        st.session_state.chat_response = ask_gpt_with_optional_kpi(text, include_kpi=include_kpi)
        st.session_state.scroll_trigger = st.session_state.get("scroll_trigger", 0) + 1#毎回トリガー値が変わり、HTMLの中身が変わってJSが再実行される
        st.rerun()

    if st.button("検索 🔎", key="send_button", use_container_width=True):
        if user_input and user_input.strip():
            fire_and_scroll(user_input.strip(),include_kpi=True)
            
        else:
            st.warning("質問を入力してください。")
    
    # よく使う質問のボタン
    st.subheader("よく使う質問")
    
    questions = [
        "睡眠パターンを分析して",
        "授乳間隔を分析して", 
        "おむつ替えタイミングを分析して",
        "ミルク量を分析して"
    ]
    
    for idx, question in enumerate(questions):
        if st.button(question, key=f"quick_q_{idx}", use_container_width=True):
            fire_and_scroll(question, include_kpi=True)


    
if __name__ == "__main__":
    main()
