import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# ページ設定
st.set_page_config(
    page_title="ベビーケア ダッシュボード",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# カスタムCSS（レスポンシブ対応 + デスクトップ1画面表示）
st.markdown("""
<style>
    /* デスクトップで1画面表示のためのメインコンテナ */
    .stApp {
        height: 100vh;
        overflow: hidden;
    }
    
    .main .block-container {
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
    
    /* デスクトップ用（769px以上）：1画面表示 */
    @media (min-width: 769px) {
        .metric-card {
            background: linear-gradient(135deg, #74b9a0 0%, #5fa892 100%);
            padding: 1rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 0.8vh;
            height: calc(40vh - 2rem);
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .chart-card {
            background: linear-gradient(135deg, #74b9a0 0%, #5fa892 100%);
            padding: 1rem;
            border-radius: 15px;
            color: white;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 0.8vh;
            height: calc(40vh - 2rem);
            display: flex;
            flex-direction: column;
        }
        
        .log-card {
            background: linear-gradient(135deg, #74b9a0 0%, #5fa892 100%);
            padding: 1rem;
            border-radius: 15px;
            color: white;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 0.8vh;
            height: calc(40vh - 2rem);
            display: flex;
            flex-direction: column;
        }
        
        .progress-card {
            background: #333;
            border-radius: 15px;
            padding: 1rem;
            margin-bottom: 0.8vh;
            height: calc(40vh - 2rem);
            display: flex;
            align-items: center;
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
        
        .metric-card, .chart-card, .log-card, .progress-card {
            background: linear-gradient(135deg, #74b9a0 0%, #5fa892 100%);
            padding: 1.2rem;
            border-radius: 18px;
            color: white;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            min-height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .progress-card {
            background: #333;
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
        
        .metric-card, .chart-card, .log-card, .progress-card {
            background: linear-gradient(135deg, #74b9a0 0%, #5fa892 100%);
            padding: 1rem;
            border-radius: 15px;
            color: white;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 0.8rem;
            min-height: 160px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .progress-card {
            background: #333;
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
        }
</style>
""", unsafe_allow_html=True)

# サンプルデータの生成
def generate_sample_data():
    # 過去7日間のデータ
    dates = [datetime.now() - timedelta(days=i) for i in range(6, -1, -1)]
    
    # おむつ替え回数のデータ
    diaper_data = {
        'date': dates,
        'count': [8, 6, 7, 9, 8, 7, 8]
    }
    
    # 授乳量のデータ  
    feeding_data = {
        'date': dates,
        'amount': [180, 160, 170, 190, 175, 165, 185]
    }
    
    # 最新ログデータ
    log_data = [
        {'time': '14:30', 'action': 'おむつ替え完了'},
        {'time': '13:45', 'action': '授乳, 200ml'},
        {'time': '12:30', 'action': 'お昼寝開始'},
        {'time': '11:15', 'action': 'おむつ替え完了'},
        {'time': '10:30', 'action': '授乳, 180ml'}
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
    
    fig = px.bar(
        df, 
        x='date', 
        y=df.columns[1],
        title="",
        color_discrete_sequence=[color]
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=10),
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
        margin=dict(t=5, b=25, l=15, r=15),
        autosize=True,
        height=150  # 高さを半分に設定
    )
    
    fig.update_traces(marker=dict(cornerradius=3))
    
    return fig

# メイン画面
def main():
    # ヘッダー
    st.markdown('<div class="main-header">ベビーケア ダッシュボード</div>', unsafe_allow_html=True)
    
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
        st.markdown('<div class="card-title">睡眠時間 前週平均比較</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_diaper_chart = create_bar_chart(diaper_data, "睡眠時間 前週平均比較", "#4A90E2")
        st.plotly_chart(fig_diaper_chart, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-label">睡眠時間（時間）/日</div>', unsafe_allow_html=True)
    
    # カード3: 最新ログ
    with cols[2]:
        st.markdown('<div class="card-title">最新ログ</div>', unsafe_allow_html=True)
        st.markdown('<div class="log-content">', unsafe_allow_html=True)
        for i, log in enumerate(log_data, 1):
            st.markdown(f'<div class="log-item">{i}. {log["time"]} {log["action"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # カード4: 授乳経過時間
    with cols[3]:
        st.markdown('<div class="card-title">授乳経過時間</div>', unsafe_allow_html=True)
        fig_feeding_progress = create_circular_progress(124, 180, "XX分", "#FF6B47")
        st.plotly_chart(fig_feeding_progress, use_container_width=True, config={'displayModeBar': False}, key="feeding_progress")
    
    # カード5: ミルク量 前週平均比較
    with cols[4]:
        st.markdown('<div class="card-title">ミルク量 前週平均比較</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_feeding_chart = create_bar_chart(feeding_data, "ミルク量 前週平均比較", "#4A90E2")
        st.plotly_chart(fig_feeding_chart, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-label">ミルク量(ml)/日</div>', unsafe_allow_html=True)
    
    # カード6: 追加機能
    with cols[5]:
        st.markdown('<div class="card-title">6個目</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; font-size: clamp(0.9rem, 2vw, 1.1rem); opacity: 0.8; line-height: 1.6;">何か入れる<br><br>・XXX<br>・XXX<br>・XXX</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# サイドバー（質問・相談機能）
with st.sidebar:
    st.title("ChatGPT")
    
    # チャット入力
    user_input = st.text_area("", placeholder="ChatGPTに質問や相談をしてください...", key="chat_input", height=100)
    
    if st.button("✈ 送信", key="send_button", use_container_width=True):
        if user_input:
            st.success("回答を生成中です・・・")
        else:
            st.warning("質問を入力してください。")
    
    # よく使う質問のボタン
    st.subheader("よく使う質問")
    
    questions = [
        "睡眠パターンについて",
        "授乳間隔について", 
        "おむつ替えタイミング",
        "夜泣きの対処法"
    ]
    
    for question in questions:
        if st.button(question, key=question, use_container_width=True):
            st.info(f"「{question}」についての情報を表示します")