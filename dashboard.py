import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from openai import OpenAI
import os

# ページ設定
st.set_page_config(
    page_title="ベビーケア ダッシュボード",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
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

#.env 読み込み（無ければ何もしない）
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


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

#ChatGPTによる回答生成
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


# サンプルデータの生成
def generate_sample_data():
    # 過去7日間のデータ
    dates = [datetime.now() - timedelta(days=i) for i in range(6, -1, -1)]
    
    # おむつ替え回数のデータ
    diaper_data = {
        'date': dates,
        'count': [12.5, 10, 9.5, 13, 10, 10.5, 12]
    }
    
    # 授乳量のデータ  
    feeding_data = {
        'date': dates,
        'amount': [600, 750, 800, 1000, 1100, 700, 900]
    }
    
    # 最新ログデータ
    log_data = [
        {'time': datetime.now() - timedelta(minutes=180), 'action': '起床'},
        {'time': datetime.now() - timedelta(minutes=240), 'action': '就寝'},
        {'time': datetime.now() - timedelta(minutes=300), 'action': '授乳、180ml'}
    
    ]
    
    return diaper_data, feeding_data, log_data

# 円形プログレスバーの作成（レスポンシブ対応）
def create_circular_progress(value, max_value, title, color="#FF6B47"):
    fig = go.Figure(data=[go.Pie(
        values=[value, max_value - value],
        hole=.7,
        marker_colors=[color, '#333333'],
        textinfo='none',
        showlegend=False,
        hoverinfo='skip'
    )])
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=0, b=0, l=0, r=0),
        autosize=True,
        height=150,  # 高さを半分に設定
        annotations=[
            dict(text=f'<b style="color:white; font-size:clamp(18px, 4vw, 24px)">{value}</b>', 
                 x=0.5, y=0.58, font_size=20, showarrow=False),
            dict(text=f'<span style="color:#ccc; font-size:clamp(10px, 2vw, 14px)">{title}</span>', 
                 x=0.5, y=0.42, font_size=14, showarrow=False)
        ]
    )
    
    return fig

# 棒グラフの作成（デスクトップ1画面対応）
def create_bar_chart(data, title, color="#4A90E2"):
    df = pd.DataFrame(data)
    
    # 棒グラフの作成
    bar_fig = go.Figure(data=[
        go.Bar(
            x=df['date'],
            y=df[df.columns[1]],
            text=df[df.columns[1]],
            marker_color=color,
            textposition='inside',
            insidetextanchor='end',
            marker_cornerradius=3,
            textfont=dict(color='white', size=12),
            showlegend=False # 凡例を非表示にする
        )
    ])
    
    # 前週平均線の作成
    weekly_average = df[df.columns[1]].mean()
    line_fig = go.Figure(data=[
        go.Scatter(
            x=df['date'],
            y=[weekly_average] * len(df),
            mode='lines',
            line=dict(color='red', width=2),
            name='前週平均',
            showlegend=False # 凡例を非表示にする
        )
    ])
    
    # 棒グラフと折れ線グラフを統合
    final_fig = go.Figure(data=bar_fig.data + line_fig.data)

    final_fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50', size=10),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickformat='%m/%d',
            title="",
            tickfont=dict(size=9)
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            title="",
            tickfont=dict(size=9)
        ),
        margin=dict(t=5, b=5, l=15, r=15),
        autosize=True,
        height=180
    )
    
    return final_fig


# 6つ目のカード「今何してる」のステータスと経過時間を計算する関数
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

# メイン画面
def main():
    # ヘッダー
    st.header("ベビーケア ダッシュボード")
    st.markdown("---")

    
    # サンプルデータの取得
    diaper_data, feeding_data, log_data = generate_sample_data()
    
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
    
    # カード1: おむつ経過時間
    with cols[0]:
        st.markdown('<div class="card-title">おむつ経過時間</div>', unsafe_allow_html=True)
        fig_diaper_progress = create_circular_progress(124, 180, "XX分", "#FF6B47")
        st.plotly_chart(fig_diaper_progress, use_container_width=True, config={'displayModeBar': False}, key="diaper_progress")
    
    # カード2: 睡眠時間 前週平均比較
    with cols[1]:
        st.markdown('<div class="card-title">　睡眠時間　前週平均比較</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_diaper_chart = create_bar_chart(diaper_data, "睡眠時間  前週平均比較", "#4A90E2")
        st.plotly_chart(fig_diaper_chart, use_container_width=True, config={'displayModeBar': False})
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
        
        #時間とアクションだけ表示するように制御
        sorted_logs = sorted(log_data, key=lambda x: x['time'], reverse=True)
        for i, log in enumerate(sorted_logs, 1):
            formatted_time = log['time'].strftime('%H:%M')
            st.markdown(f'<div class="log-item">{i}. {formatted_time} {log["action"].split(",")[0]}</div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    
    # カード4: 授乳経過時間
    with cols[3]:
        st.markdown('<div class="card-title">授乳経過時間</div>', unsafe_allow_html=True)
        fig_feeding_progress = create_circular_progress(124, 180, "XX分", "#FF6B47")
        st.plotly_chart(fig_feeding_progress, use_container_width=True, config={'displayModeBar': False}, key="feeding_progress")
    
    # カード5: ミルク量 前週平均比較
    with cols[4]:
        st.markdown('<div class="card-title">　ミルク量(ml)　前週平均比較</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_feeding_chart = create_bar_chart(feeding_data, "ミルク量  前週平均比較", "#4A90E2")
        st.plotly_chart(fig_feeding_chart, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
        
    
    # カード6: 今何してる
    with cols[5]:
        status_text, time_passed_str, log_time = get_status_and_time(log_data)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">今何してる</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="text-align: center;">
                <div class="time-text">
                    {log_time.strftime('%H:%M')}に{status_text}
                </div>
                <div class="time-text">
                    {time_passed_str}
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
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
                        if(retry<100) return setTimeout(()=>go(retry+1), 20);
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
        # ← 消費したトリガーを記録（ここが大事）
        st.session_state['_last_scrolled'] = st.session_state['scroll_trigger']
# サイドバー（質問・相談機能）
with st.sidebar:
    st.title("ChatGPT 育児相談")
    
    # セッションステートを初期化
    if 'chat_response' not in st.session_state:
        st.session_state.chat_response = ""
    if 'scroll_trigger' not in st.session_state: #スクロールのために追加
        st.session_state.scroll_trigger = 0 #初期化する

    # チャット入力
    user_input = st.text_area("", placeholder="入力してください...", key="chat_input", height=100)
    
    def fire_and_scroll(text: str):
        st.session_state.chat_response = get_chat_response(text)
        st.session_state.scroll_trigger += 1 #毎回トリガー値が変わり、HTMLの中身が変わってJSが再実行される
        st.rerun()

    if st.button("検索 🔎", key="send_button", use_container_width=True):
        if user_input:
            fire_and_scroll(user_input)
            
        else:
            st.warning("質問を入力してください。")
    
    # よく使う質問のボタン
    st.subheader("よく使う質問")
    
    questions = [
        "睡眠パターンを分析して",
        "授乳間隔を分析して", 
        "おむつ替えタイミングを分析して",
        "夜泣きの対処法を教えて"
    ]
    
    for question in questions:
        if st.button(question, key=question, use_container_width=True):
            fire_and_scroll(question)


    
if __name__ == "__main__":
    main()
