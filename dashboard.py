import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from openai import OpenAI
import os
from supabase import create_client
import pytz #タイムゾーンデータベースを提供するライブラリ
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
# タイムゾーン定義
# ---------------------------------------------------------
JST = pytz.timezone('Asia/Tokyo')

# ---------------------------------------------------------
# supabaseからおむつ替え経過時間計算＜カード1＞
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
            
            # 2. UTCからJSTに変換する
            log_time_jst = log_time_utc.astimezone(pytz.timezone('Asia/Tokyo'))
            
            # 3. 現在時刻をJSTで取得する (上記 1.で定義した JST を使用)
            current_time_jst = datetime.now(pytz.timezone('Asia/Tokyo'))
            
            # 4. JST同士で経過時間を計算
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
# supabaseから睡眠時間の日ごとの累計値と前週平均の計算＜カード2＞
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
        df['datetime'] = pd.to_datetime(df['datetime']) # UTC情報付きとして読み込む
        df['datetime'] = df['datetime'].dt.tz_convert('Asia/Tokyo')  # UTCからJST (Asia/Tokyo) に変換
        df['date'] = df['datetime'].dt.date  # 日付列を作成 (JSTベースの日付になる)
        
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
#supabaseから最新ログを取得＜カード3＞
#---------------------------------------------------------
@st.cache_data(ttl=60) # 1分間キャッシュ
def get_supabase_data(table_name="baby_events"):
    """Supabaseからデータを取得し、JSTに変換して返す"""
    try:
        response = supabase_client.table(table_name).select("datetime, type_jp").order("datetime", desc=True).limit(3).execute()
        
        # データをDataFrameに変換
        df = pd.DataFrame(response.data)
        
        # JST変換と表示フォーマット
        if not df.empty and 'datetime' in df.columns:
            # 1. UTC情報付きとして読み込み
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # 2. UTCからJST (Asia/Tokyo) に変換
            df['datetime'] = df['datetime'].dt.tz_convert('Asia/Tokyo')
            
            # 3. 表示用の形式にフォーマット (タイムゾーン情報を削除して見やすくする)
            df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M')
            
        # DataFrameを辞書リストに戻す（st.dataframeにそのまま渡せる）
        return df.to_dict('records')
    
    except Exception as e:
        st.error(f"データベースの読み込み中にエラーが発生しました: {e}")
        return []


# ---------------------------------------------------------
# supabaseから授乳経過時間計算＜カード4＞
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
# supabaseからミルク量の日ごとの累計値と前週平均の計算＜カード5＞
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
# supabaseから最新の睡眠ステータスログを取得・計算＜カード6用＞
# ---------------------------------------------------------
@st.cache_data(ttl=60) # 1分間キャッシュ
def get_sleep_status_log(table_name="baby_events"):
    """
    Supabaseから最新の「sleep_start」または「sleep_end」ログを1件取得する。
    status/time計算のため、datetime, type_jp, type_slugを含める。
    """
    try:
        # type_slugが 'sleep_start' または 'sleep_end' の最新ログを1件取得
        response = supabase_client.table(table_name).select("datetime, type_jp, type_slug").in_('type_slug', ['sleep_start', 'sleep_end']).order("datetime", desc=True).limit(1).execute()
        
        if response.data:
            # get_status_and_time に渡すため、辞書のリスト形式で返す
            return response.data 
        else:
            # データがない場合は空のリストを返す
            return []
    except Exception as e:
        st.error(f"睡眠ステータスログの読み込み中にエラーが発生しました: {e}")
        return []

def get_status_and_time(log_data):
    """
    Supabaseのログデータ（UTC時刻）から最新の活動とJSTでの経過時間を計算する。
    ※ ログデータが 'datetime' と 'type_jp', 'type_slug' を持つ形式を想定
    """
    if not log_data:
        # データがない場合はデフォルト値を返す
        return "ログなし", "—", None
    
    # 最新のログを取得（get_sleep_status_logは最新ログ1件をリストで返すため、[0]を取得）
    latest_log = log_data[0] 
    
    # 1. ログ時刻をタイムゾーン付きで読み込み、JSTに変換する
    log_time_utc = datetime.fromisoformat(latest_log['datetime'])
    log_time_jst = log_time_utc.astimezone(pytz.timezone('Asia/Tokyo'))
    
    # 2. 現在時刻をJSTで取得する
    current_time_jst = datetime.now(pytz.timezone('Asia/Tokyo'))
    
    # 3. JST同士で経過時間を計算
    delta = current_time_jst - log_time_jst
    total_minutes = int(delta.total_seconds() / 60)

    # 4. 経過時間を「〇時間〇分前」の文字列に変換
    hours = total_minutes // 60
    minutes = total_minutes % 60
    
    if hours == 0 and minutes == 0:
        time_passed_str = "たった今"
    elif hours == 0:
        time_passed_str = f"{minutes}分前"
    else:
        time_passed_str = f"{hours}時間{minutes}分前"
    
    # 5. ステータスを決定 (type_slug / type_jp に基づく)
    action = latest_log.get('type_jp', '不明な活動')
    status_text = "活動中" # デフォルト

    # sleep_start と sleep_end の判定に特化
    if latest_log.get('type_slug') == 'sleep_start' or "就寝" in action:
        status_text = "就寝中"
    elif latest_log.get('type_slug') == 'sleep_end' or "起床" in action:
        status_text = "起床中"
    
    # その他の活動も表示したい場合は、ここにロジックを追加できます
    # 例：elif "授乳" in action: status_text = "授乳後"

    return status_text, time_passed_str, log_time_jst

# ---------------------------------------------------------
# 既存KPIから派生統計を計算 → 日常語ラベル化（色バッジは使わない）
# （DB集計関数の直後に置く：グラフ関数の前）
# ---------------------------------------------------------
# get_sleep_summary_data / get_feeding_summary_data / get_diaper_elapsed_time / get_feeding_elapsed_time
# get_chat_response は既存実装を利用

def _series_stats(values: list[float]) -> dict:
    """
    目的:
        数値系列(list[float])から「平均」「標準偏差」「１日あたりの直線的傾き」を計算する。
    引数:
        values:日単位の値(例:睡眠時間[h/日])、ミルク量[ml/日]
    戻り値(dict):
        {
            "mean": 平均値(float),
            "std": 標準偏差（float, 不偏ではなく母標準偏差 ddof=0）,
            "trend_slope_per_day": 直線回帰で推定した1日あたりの傾き（float）
        }
    実装メモ:
        - 配列化（np.array）して計算を安定化。
        - データが空ならすべて0で返す（ダッシュボード側の表示を安全にするため）。
        - 傾きはX=0..n-1を説明変数にpolyfit(1次)で取得（要素2以上の時のみ）。
    """
    #以下は上記戻り値（dict）の補足説明
    #ddof=0 とは今あるデータ集合そのもののばらつき”**をそのまま測る、という意味。今週の実測データの事実を示すならddof=0がいいとのこと。ddof=1にすると分母がn-1になる。
    #trend_slope_per_day 1日あたりに平均してどれくらい増減しているかを示す数値（単位は/日）

    arr = np.array(values, dtype=float)#floatで少数を表せる数値型にすることで、平均・標準偏差・回帰の傾きなどの小数点計算を正確にする。
    if arr.size == 0:#配列が空（要素数が０）かどうかチェック。データが１つもないと計算できないため使用。
        return {"mean": 0.0, "std": 0.0, "trend_slope_per_day": 0.0}#空データの場合の安全な初期値を返す。0.0にすることでダッシュボードや後続処理でのエラーを防ぐ。
    #以下のコードでやりたいこと
    #日ごとのデータ（arr）があるとき、「最近1日あたりどのくらい増えている？減っている？」＝傾きをざっくり出したい
    #そのために日数の番号（0日目、1日目、2日目…）を説明変数として使って直線を当てはめる（一次回帰）→直線の傾きを取り出す。
    x = np.arange(arr.size, dtype=float)#データの本数分 x = 0,1,2という連番を作って「日数の流れを横軸にしている」 
    #np.polyfit(x, arr, 1) は「x と arr の点群に、一次式（直線）を一番いい感じにフィットさせる」関数
    #戻り値は [傾き, 切片] の2つ。[0] で傾き（slope）だけ取り出している。
    #if文：ただし、データが1点しかないと直線の傾きは決められないので、その場合は 0.0 としている。
    slope = float(np.polyfit(x, arr, 1)[0]) if arr.size >= 2 else 0.0
    return {
        "mean": float(arr.mean()), #平均=全体の基準
        "std": float(arr.std(ddof=0)), #標準偏差＝日々のムラ
        "trend_slope_per_day": slope, #傾き＝最近の流れ（増えている？減っている？横這い？）
        #上記3点セットがそろっていると状況の要約がやりやすい。
    }

def _qualitative_labels(mean: float, std: float, slope: float, unit: str,
                        abs_threshold: float | None = None) -> dict:
    """
    目的:
        数字の統計量（平均/標準偏差/傾き）を「日常語の短い表現」に変換する。
    
    引数:
        mean:   平均
        std:    標準偏差
        slope:  1日あたりの傾き（_series_statsのtrend_slope_per_day）
        unit:   単位の短い表記（例:"時間/日"、"ml/日"）
        abs_threshold: 絶対値で傾きを有意とみなす下限（例:睡眠0.2h/日、ミルク20ml/日）
                       （「傾きがこれ以上なら"増えている/減っている"と言い切ろう」という最低ラインのこと）
                       Noneの場合は、絶対閾値を使わず、相対判定（±5%/日）だけで判定
                       （絶対量ではなく「平均と比べて1日あたり±5% 以上なら増減とみなす」という相対的な目安だけで判定）
    判定ロジック（概略）:
    - 変動の大きさ(平均と比べて、どれくらい日々の差があるか): 変動係数 CV=std/|mean|を用い、閾値10%/25%で3段階に言語化: 『10%未満：ほぼ毎日おなじ / 10~25%：日によって少しちがう / 25%以上:日によってかなりちがう』
      目安幅として±10%/±25% を実数化にして同梱
      例）平均６時間なら
      ・±10%~±0.6時間（この範囲内のブレなら小さめ）
      ・±25%~±1.5時間（ここを超えるブレは大きめ）

    - 傾き: slope（1日あたりの増減）が
        ・プラスで十分大きい→「少し増えつつある」
        ・マイナスで十分大きい→「少し減りつつある」
        ・どちらでもない→「だいたい同じ」
    　※「十分大きい」の判断は2つのどちらかを満たしたとき：
        1.abs_threshold(絶対ライン)以上
        　例：睡眠で+0.25h/日は0.2h/日を超えるので「増えてる」と言いやすい
        2.平均と比べて±5%/日以上（相対ライン）
        　例：平均6hで+0.4h/日は0.4/6~6.7%/日→増えてる判定
    
    戻り値（dict）:
        {
        "variability": 変動の大きさ,
        "variability_phrase": 変動の説明（1行）,
        "trend": 傾向
        "trend_phrase": 傾向の説明（1行）
        "guideline_band_10pct": 目安幅（±10%の実数(平均×0.10)）,
        "guideline_band_25pct": 目安幅（±25%の実数(平均×0.25)）
        }
    
    実装メモ:
    - meanが0近傍で割り算が不安定にならないようepsを加算。
        「変動の大きさ」を出すときにCV=標準偏差÷平均という計算をしている。
        平均値が0に近いと分母が小さすぎて結果が「異常に大きな数字」になってしまう。
        さらに平均が完全に0.0ならゼロ割エラーが発生する。
        そこでeps（ごく小さい数。例:1e-8）を足すことで分母が完全に0になることや計算が極端に跳ね上がるのを防ぐ。
    - "日常語"のみで返す要件のため、専門用語は返却値に含めない。
    
    具体例（数値でイメージ）
    直近7日の睡眠：だいたい 6.0時間/日
    日々のバラつき：0.6時間（平均の10%）
    傾き：+0.25時間/日（ここ数日で少しずつ増えている）
    単位：「時間/日」
    絶対の目安（abs_threshold）：0.2時間/日

    この場合の出力イメージ：
    変動：10% → 「ほぼ毎日おなじ」
    説明：「日ごとの差は小さめ（目安：±0.6時間/日以内）。」
    傾向：+0.25h/日 は 0.2h/日 を超える → 「少し増えつつある」
    説明：「ここ数日は時間/日がゆるやかに増えています。」
    目安幅：
    10% → ±0.6時間
    25% → ±1.5時間
    """
    eps = 1e-9 #ゼロ割を回避
    cv = std / (abs(mean) + eps) #平均に対してどれくらいブレているか
    band10 = abs(mean) * 0.10 #平均の10%を実際の単位(時間/日、ml/日)の数値に直す。abs(mean)を使うのは幅が必ず正の値になるようにするため。
    band25 = abs(mean) * 0.25

    #統計用語を使わず、一目でニュアンスが伝わる日本語に落とし込むための閾値設計。
    #具体例:平均6.0時間/日、標準偏差:0.9時間
    #cv = 0.9/6.0 = 0.15(15%)→日によって少し違う
    #UI/説明向け：単位を含む自然な一文(variability_phrase)をそのまま画面やプロンプトに出せる。

    if cv < 0.10: #cv(=ブレの割合)に応じて3つのラベルのどれかを選ぶ
        variability = "ほぼ毎日おなじ"
        variability_phrase = f"日ごとの差は小さめ（目安: ±{band10:.1f}{unit}以内）。" #.1fは小数点1桁で丸めて見やすくする工夫
    elif cv < 0.25:
        variability = "日によって少しちがう"
        variability_phrase = f"日ごとの差は中くらい（目安: ±{band10:.1f}〜±{band25:.1f}{unit}）。"
    else:
        variability = "日によってかなりちがう"
        variability_phrase = f"日ごとの差は大きめ（目安: ±{band25:.1f}{unit}以上）。"

    #二段構えの有意性チェック（絶対・相対）で言い過ぎを防止
    #相対しきい値（5%/日）や絶対しきい値（0.2h/日、20ml/日）は対象に合わせて調整可能。
    #平均が0のときは%判定を切り離し、絶対量で判断

    #1日あたりの変化量slopeが平均meanに対してどれくらいの割合かを出している。
    #同じ「+0.3/日」でも平均6なら+5%/日、平均12なら+2.5%/日。平均に対する割合でみると大小の比較がフェアになる。
    rel = abs(slope) / (abs(mean) + eps) if mean else 0.0
    #相対基準だけだと平均が極端に小さい/大きいと判定がブレる、絶対基準だけだと指標のスケール以前になり比較がしにくい。
    #両方用意して、どちらかを満たせば有意とすることで現実的で過剰反応しない判定にしている。
    use_abs = abs_threshold is not None and abs(slope) >= abs_threshold #絶対的な基準を超えたか（睡眠時間0.2時間/日、ミルクなら20ml/日）
    use_rel = rel >= 0.05  # 相対的な基準(5%/日)を超えたか

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
    """
    目的:
        ダッシュボードと同じ集計条件でKPI(直近7日+前週平均など)を取得し、
        "派生統計"と"日常語ラベル"を付けたJSONを作る。
        GPTに渡す一次ソース(KPI_JSON)として使用。
    
    処理の流れ:
        1)既存の集計関数から睡眠/授乳の日次データと前週平均を取得
        2)直近7日分だけを抽出(tail(7))
        3)値列（2列目）を安全に特定→欠損は0で穴埋め
        4)_series_statsで平均・標準偏差・傾きを計算
        5)_qualitative_labelsで"日常語ラベル"を付与
        　（睡眠の傾きは0.2h/日、ミルクは20ml/日を絶対閾値の目安として利用）
        6)おむつ/授乳の「前回からの経過分（分）」も付け、バケット化（0-90/90-180/180+）
        7)GPTに渡しやすいフラットな辞書構造（JSON）で返す
    """
    # 既存の集計関数から睡眠と授乳のグラフ用データと前週平均を取得。
    #table_name="baby_events" は、データの置き場所（テーブル名）を明示。
    #ダッシュボードと同じ条件で集計し、数字の整合性を保つ。
    sleep_chart_data, last_week_avg_sleep = get_sleep_summary_data(table_name="baby_events")
    feeding_chart_data, last_week_avg_amount = get_feeding_summary_data(table_name="baby_events")
    #おむつ・授乳の最終イベントからの経過分を取得。関数が (ラベル, 分) で返す場合に備え、分だけにそろえる。
    #呼び出し元の差異（戻り値がタプル/単値）を吸収し、あとで扱いやすい整数 minutesへ統一。
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

    # カード2用データ取得: 睡眠時間の日ごとの累計と前週平均 
    sleep_chart_data, last_week_avg_sleep = get_sleep_summary_data(table_name="baby_events")

    # カード3用データ取得　Supabaseから最新ログデータを取得
    supabase_log_data = get_supabase_data(table_name="baby_events") # テーブル名を編集

    # カード4用データ取得: 最新の授乳からの経過時間を取得
    elapsed_minutes_feeding = get_feeding_elapsed_time(table_name="baby_events")
    FEEDING_MAX_MINUTES = 180 # 授乳グラフの上限を180分（3時間）に設定

    # カード5用データ取得: ミルク量の日ごとの累計と前週平均 
    feeding_chart_data, last_week_avg_amount = get_feeding_summary_data(table_name="baby_events")
    
    # カード6用データ取得　Supabaseから最新の起床or就寝ログを取得
    supabase_log_data = get_sleep_status_log(table_name="baby_events")  
    latest_sleep_log = supabase_log_data[0] if supabase_log_data else None 
    
    
    
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
        
        # latest_sleep_logがmain()で定義されていることを前提とする
        if latest_sleep_log:
            
            # 1. UTC時刻をJSTに変換
            log_time_utc = datetime.fromisoformat(latest_sleep_log['datetime'])
            log_time_jst = log_time_utc.astimezone(pytz.timezone('Asia/Tokyo'))
            current_time_jst = datetime.now(pytz.timezone('Asia/Tokyo'))
            
            # 2. 経過時間（分）を計算
            delta = current_time_jst - log_time_jst
            total_minutes = int(delta.total_seconds() / 60)
            
            # 3. 状態、絵文字、表示テキストを決定
            status_text_verb = ""
            status_text_current = "" # 「起きています」/「寝ています」 
            emoji = ""
            
            if latest_sleep_log.get('type_slug') == 'sleep_start':
                status_text_verb = "就寝" # 表示文言を「就寝中」から「就寝」に変更
                status_text_current = "寝ています"
                emoji = "😴"
            elif latest_sleep_log.get('type_slug') == 'sleep_end':
                status_text_verb = "起床" # 表示文言を「起床中」から「起床」に変更
                status_text_current = "起きています"
                emoji = "🌞"
            else:
                status_text_verb = "不明"
                status_text_current = "不明な状態"
                emoji = "❓"

            # 4. 経過時間を「〇時間〇分」形式に変換
            hours = total_minutes // 60
            minutes = total_minutes % 60
            
            if total_minutes < 1:
                formatted_time_passed = "たった今"
            elif hours == 0:
                formatted_time_passed = f"{minutes}分経過"
            else:
                formatted_time_passed = f"{hours}時間{minutes}分経過" # 「経過」を削除
                
            # 5. HTML表示
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="font-size: 3.0rem; margin-bottom: 0.5rem;">
                        {emoji}
                    </div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: #3498db; margin-bottom: 0.5rem;">
                        {status_text_current}
                    </div>
                    <div style="font-size: 1.0rem; color: #2c3e50;">
                        {log_time_jst.strftime('%H:%M')}に{status_text_verb} &nbsp; | &nbsp; {formatted_time_passed}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("就寝/起床ログがありません。")
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
