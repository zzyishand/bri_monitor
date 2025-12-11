"""
BRI Bubble Risk Indicator Monitoring Dashboard
Interactive Streamlit application for visualizing BRI indicators across multiple assets
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="BRI Monitor Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Asset information with Chinese names
ASSET_INFO = {
    'DOW_JONES': {'name_en': 'Dow Jones', 'name_cn': '道琼斯', 'category': 'Global Equity'},
    'NASDAQ_100': {'name_en': 'NASDAQ-100', 'name_cn': '纳斯达克100', 'category': 'Global Equity'},
    'NIKKEI_225': {'name_en': 'Nikkei 225', 'name_cn': '日经225', 'category': 'Global Equity'},
    'HSCEI': {'name_en': 'HSCEI', 'name_cn': '恒生国企', 'category': 'Global Equity'},
    'DAX': {'name_en': 'DAX', 'name_cn': '德国DAX', 'category': 'Global Equity'},
    'CSI300': {'name_en': 'CSI 300', 'name_cn': '中证300', 'category': 'Global Equity'},
    'CSI500': {'name_en': 'CSI 500', 'name_cn': '中证500', 'category': 'Global Equity'},
    'HSTECH': {'name_en': 'HSTECH', 'name_cn': '恒生科技', 'category': 'Global Equity'},
    'XLF': {'name_en': 'Financials', 'name_cn': '金融', 'category': 'US Sector'},
    'XLY': {'name_en': 'Consumer Disc.', 'name_cn': '可选消费', 'category': 'US Sector'},
    'XLC': {'name_en': 'Communication', 'name_cn': '通信服务', 'category': 'US Sector'},
    'XLI': {'name_en': 'Industrials', 'name_cn': '工业', 'category': 'US Sector'},
    'XLK': {'name_en': 'Technology', 'name_cn': '科技', 'category': 'US Sector'},
    'XLV': {'name_en': 'Healthcare', 'name_cn': '医疗', 'category': 'US Sector'},
    'XLE': {'name_en': 'Energy', 'name_cn': '能源', 'category': 'US Sector'},
    'IXE': {'name_en': 'Energy Index', 'name_cn': '能源指数', 'category': 'US Sector'},
    'BIOTECH': {'name_en': 'Biotech', 'name_cn': '生物科技', 'category': 'US Sector'},
    'GOLD': {'name_en': 'Gold', 'name_cn': '黄金', 'category': 'Commodity'},
    'CRUDE_OIL': {'name_en': 'Crude Oil', 'name_cn': '原油', 'category': 'Commodity'},
    'COPPER': {'name_en': 'Copper', 'name_cn': '铜', 'category': 'Commodity'},
    'BITCOIN': {'name_en': 'Bitcoin', 'name_cn': '比特币', 'category': 'Crypto'},
    'MAG7': {'name_en': 'Mag 7', 'name_cn': '科技7巨头', 'category': 'Tech Giants'}
}

@st.cache_data(ttl=3600)
def load_bri_data():
    """Load all BRI data from CSV files"""
    data_dir = Path(__file__).parent / 'indicator' / 'bri_results_v2_with_intermediates'
    
    if not data_dir.exists():
        st.error(f"Data directory not found: {data_dir}")
        return {}
    
    all_data = {}
    for csv_file in data_dir.glob('*_BRI_v2_*.csv'):
        # Extract asset name from filename (e.g., DOW_JONES_BRI_v2_20251211_195655.csv -> DOW_JONES)
        # Split by '_BRI_v2_' and take the first part
        asset_name = csv_file.stem.split('_BRI_v2_')[0]
        
        try:
            df = pd.read_csv(csv_file)
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            df = df.sort_values('Date')
            all_data[asset_name] = df
        except Exception as e:
            st.warning(f"Error loading {csv_file.name}: {e}")
            continue
    
    return all_data

@st.cache_data
def get_latest_metrics(all_data):
    """Calculate latest metrics for all assets"""
    metrics = []
    
    for asset_name, df in all_data.items():
        if asset_name not in ASSET_INFO:
            continue
            
        if len(df) < 2:
            continue
        
        # Find the latest row with valid BRI data
        valid_data = df[df['composite_bri'].notna()].copy()
        if len(valid_data) == 0:
            continue
        
        # Get latest valid data
        latest = valid_data.iloc[-1]
        
        # Calculate metrics
        bri = float(latest['composite_bri']) if pd.notna(latest['composite_bri']) else 0.0
        price = float(latest['price']) if pd.notna(latest['price']) else 0.0
        daily_return = float(latest['returns']) if pd.notna(latest['returns']) else 0.0
        
        # Get sub-indicators
        short_bri = float(latest['short_indicator']) if pd.notna(latest['short_indicator']) else 0.0
        mid_bri = float(latest['mid_indicator']) if pd.notna(latest['mid_indicator']) else 0.0
        long_bri = float(latest['long_indicator']) if pd.notna(latest['long_indicator']) else 0.0
        
        # Skip if all indicators are 0 or 0.5 (default/invalid values)
        if bri in [0.0, 0.5] and short_bri in [0.0, 0.5] and mid_bri in [0.0, 0.5]:
            continue
        
        metrics.append({
            'asset': asset_name,
            'name_en': ASSET_INFO[asset_name]['name_en'],
            'name_cn': ASSET_INFO[asset_name]['name_cn'],
            'category': ASSET_INFO[asset_name]['category'],
            'bri': bri,
            'short_bri': short_bri,
            'mid_bri': mid_bri,
            'long_bri': long_bri,
            'price': price,
            'daily_return': daily_return,
            'date': latest['Date']
        })
    
    return pd.DataFrame(metrics)

def create_bubble_chart(metrics_df):
    """Create interactive bubble chart"""
    if metrics_df.empty:
        st.warning("No data available for bubble chart")
        return None
    
    # Prepare data for bubble chart
    metrics_df['size'] = metrics_df['bri'] * 100 + 10  # Scale size
    metrics_df['color'] = metrics_df['daily_return'].apply(
        lambda x: 'green' if x < 0 else 'red'
    )
    metrics_df['label'] = metrics_df.apply(
        lambda row: f"{row['name_en']}<br>{row['name_cn']}", axis=1
    )
    metrics_df['hover_text'] = metrics_df.apply(
        lambda row: f"<b>{row['name_en']} / {row['name_cn']}</b><br>" +
                   f"BRI: {row['bri']:.2%}<br>" +
                   f"Daily Return: {row['daily_return']:.2%}<br>" +
                   f"Price: {row['price']:.2f}",
        axis=1
    )
    
    # Create scatter plot with custom colors
    fig = go.Figure()
    
    # Add bubbles for each category
    categories = metrics_df['category'].unique()
    colors = px.colors.qualitative.Set3
    
    for i, category in enumerate(categories):
        cat_data = metrics_df[metrics_df['category'] == category]
        
        for _, row in cat_data.iterrows():
            # Determine color based on return
            bubble_color = 'rgb(46, 204, 113)' if row['daily_return'] < 0 else 'rgb(231, 76, 60)'
            
            fig.add_trace(go.Scatter(
                x=[row['bri']],
                y=[row['daily_return']],
                mode='markers+text',
                marker=dict(
                    size=row['size'],
                    color=bubble_color,
                    opacity=0.6,
                    line=dict(width=2, color='white')
                ),
                text=row['label'],
                textposition='middle center',
                textfont=dict(size=10, color='white', family='Arial Black'),
                hovertext=row['hover_text'],
                hoverinfo='text',
                name=row['asset'],
                customdata=[row['asset']],
                showlegend=False
            ))
    
    fig.update_layout(
        title={
            'text': 'BRI Bubble Risk Monitor - Heat Map',
            'font': {'size': 24, 'color': '#2c3e50', 'family': 'Arial Black'},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Composite BRI',
        yaxis_title='Daily Return',
        xaxis=dict(
            tickformat='.0%',
            gridcolor='lightgray',
            showgrid=True,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray'
        ),
        yaxis=dict(
            tickformat='.2%',
            gridcolor='lightgray',
            showgrid=True,
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray'
        ),
        plot_bgcolor='white',
        hovermode='closest',
        height=700,
        font=dict(size=12)
    )
    
    return fig

def create_indicator_plots(asset_data, asset_name):
    """Create 4 BRI indicator plots and price plot"""
    if asset_data.empty:
        st.warning(f"No data available for {asset_name}")
        return
    
    # Filter out rows with missing data
    plot_data = asset_data.dropna(subset=['composite_bri', 'short_indicator', 'mid_indicator', 'long_indicator'])
    
    if plot_data.empty:
        st.warning(f"No valid BRI data for {asset_name}")
        return
    
    # Create subplots
    col1, col2 = st.columns(2)
    
    with col1:
        # Short-term BRI
        fig_short = go.Figure()
        fig_short.add_trace(go.Scatter(
            x=plot_data['Date'],
            y=plot_data['short_indicator'],
            mode='lines',
            name='Short-term BRI',
            line=dict(color='#3498db', width=2),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.2)'
        ))
        fig_short.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="High Risk (70%)")
        fig_short.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Elevated (50%)")
        fig_short.add_hline(y=0.3, line_dash="dash", line_color="green", annotation_text="Moderate (30%)")
        fig_short.update_layout(
            title='Short-term BRI (3-month)',
            yaxis_title='BRI Value',
            yaxis=dict(tickformat='.0%'),
            height=300,
            hovermode='x unified'
        )
        st.plotly_chart(fig_short, width="stretch")
        
        # Mid-term BRI
        fig_mid = go.Figure()
        fig_mid.add_trace(go.Scatter(
            x=plot_data['Date'],
            y=plot_data['mid_indicator'],
            mode='lines',
            name='Mid-term BRI',
            line=dict(color='#9b59b6', width=2),
            fill='tozeroy',
            fillcolor='rgba(155, 89, 182, 0.2)'
        ))
        fig_mid.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="High Risk (70%)")
        fig_mid.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Elevated (50%)")
        fig_mid.add_hline(y=0.3, line_dash="dash", line_color="green", annotation_text="Moderate (30%)")
        fig_mid.update_layout(
            title='Mid-term BRI (6-month)',
            yaxis_title='BRI Value',
            yaxis=dict(tickformat='.0%'),
            height=300,
            hovermode='x unified'
        )
        st.plotly_chart(fig_mid, width="stretch")
    
    with col2:
        # Long-term BRI
        fig_long = go.Figure()
        fig_long.add_trace(go.Scatter(
            x=plot_data['Date'],
            y=plot_data['long_indicator'],
            mode='lines',
            name='Long-term BRI',
            line=dict(color='#e74c3c', width=2),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.2)'
        ))
        fig_long.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="High Risk (70%)")
        fig_long.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Elevated (50%)")
        fig_long.add_hline(y=0.3, line_dash="dash", line_color="green", annotation_text="Moderate (30%)")
        fig_long.update_layout(
            title='Long-term BRI (1-year)',
            yaxis_title='BRI Value',
            yaxis=dict(tickformat='.0%'),
            height=300,
            hovermode='x unified'
        )
        st.plotly_chart(fig_long, width="stretch")
        
        # Composite BRI
        fig_composite = go.Figure()
        fig_composite.add_trace(go.Scatter(
            x=plot_data['Date'],
            y=plot_data['composite_bri'],
            mode='lines',
            name='Composite BRI',
            line=dict(color='#16a085', width=3),
            fill='tozeroy',
            fillcolor='rgba(22, 160, 133, 0.2)'
        ))
        fig_composite.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="High Risk (70%)")
        fig_composite.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Elevated (50%)")
        fig_composite.add_hline(y=0.3, line_dash="dash", line_color="green", annotation_text="Moderate (30%)")
        fig_composite.update_layout(
            title='Composite BRI (Average)',
            yaxis_title='BRI Value',
            yaxis=dict(tickformat='.0%'),
            height=300,
            hovermode='x unified'
        )
        st.plotly_chart(fig_composite, width="stretch")
    
    # Price plot (full width)
    st.markdown("---")
    fig_price = go.Figure()
    
    # Add price line
    fig_price.add_trace(go.Scatter(
        x=asset_data['Date'],
        y=asset_data['price'],
        mode='lines',
        name='Price',
        line=dict(color='#2c3e50', width=2),
        fill='tozeroy',
        fillcolor='rgba(44, 62, 80, 0.1)'
    ))
    
    # Add secondary y-axis for BRI
    fig_price.add_trace(go.Scatter(
        x=plot_data['Date'],
        y=plot_data['composite_bri'],
        mode='lines',
        name='Composite BRI',
        line=dict(color='#e74c3c', width=2, dash='dot'),
        yaxis='y2'
    ))
    
    fig_price.update_layout(
        title=f'Price & BRI History - {ASSET_INFO.get(asset_name, {}).get("name_en", asset_name)} / {ASSET_INFO.get(asset_name, {}).get("name_cn", asset_name)}',
        xaxis_title='Date',
        yaxis=dict(title='Price', side='left'),
        yaxis2=dict(
            title='Composite BRI',
            overlaying='y',
            side='right',
            tickformat='.0%'
        ),
        height=400,
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.8)')
    )
    
    st.plotly_chart(fig_price, width="stretch")

def main():
    """Main application"""
    
    # Title
    st.title("📊 BRI Bubble Risk Indicator Monitor")
    st.markdown("**Real-time bubble risk monitoring across global assets**")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading BRI data..."):
        all_data = load_bri_data()
    
    if not all_data:
        st.error("No BRI data found. Please run BRI calculations first.")
        st.info("Run: `python indicator/example_calculate_bri_v2.py`")
        return
    
    # Get latest metrics
    metrics_df = get_latest_metrics(all_data)
    
    if metrics_df.empty:
        st.error("No valid metrics data available")
        return
    
    # Sidebar - filters and controls
    st.sidebar.header("🎛️ Controls")
    
    # Category filter
    categories = ['All'] + sorted(metrics_df['category'].unique().tolist())
    selected_category = st.sidebar.selectbox("Filter by Category", categories)
    
    if selected_category != 'All':
        metrics_df = metrics_df[metrics_df['category'] == selected_category]
    
    # Date range for detail view
    st.sidebar.markdown("---")
    st.sidebar.header("📅 Detail View Settings")
    lookback_days = st.sidebar.slider("Lookback Period (days)", 30, 365*5, 365, 30)
    
    # Display summary statistics
    st.sidebar.markdown("---")
    st.sidebar.header("📈 Summary Statistics")
    st.sidebar.metric("Total Assets", len(metrics_df))
    st.sidebar.metric("Avg BRI", f"{metrics_df['bri'].mean():.2%}")
    st.sidebar.metric("High Risk (>70%)", len(metrics_df[metrics_df['bri'] > 0.7]))
    st.sidebar.metric("Elevated Risk (50-70%)", len(metrics_df[(metrics_df['bri'] >= 0.5) & (metrics_df['bri'] <= 0.7)]))
    
    # Main content - Bubble chart
    st.header("🔮 BRI Heat Map - All Assets")
    st.markdown("""
    **Instructions:**
    - **Bubble Size** = BRI magnitude (larger = higher bubble risk)
    - **Color**: 🟢 Green = Negative return today | 🔴 Red = Positive return today
    - **Click asset name** below to view detailed indicators
    """)
    
    bubble_fig = create_bubble_chart(metrics_df)
    if bubble_fig:
        st.plotly_chart(bubble_fig, width="stretch")
    
    # Asset selection for detail view
    st.markdown("---")
    st.header("📊 Detailed Asset Analysis")
    
    # Sort assets by BRI for display
    metrics_df_sorted = metrics_df.sort_values('bri', ascending=False)
    
    # Create selection buttons
    cols = st.columns(5)
    for idx, (_, row) in enumerate(metrics_df_sorted.iterrows()):
        col = cols[idx % 5]
        with col:
            risk_emoji = "🔴" if row['bri'] > 0.7 else "🟡" if row['bri'] > 0.5 else "🟢"
            if st.button(
                f"{risk_emoji} {row['name_en']}\n{row['name_cn']}\n{row['bri']:.1%}",
                key=row['asset'],
                width="stretch"
            ):
                st.session_state['selected_asset'] = row['asset']
    
    # Display detailed analysis for selected asset
    if 'selected_asset' in st.session_state and st.session_state['selected_asset']:
        selected_asset = st.session_state['selected_asset']
        
        if selected_asset in all_data:
            st.markdown("---")
            asset_info = ASSET_INFO.get(selected_asset, {})
            st.header(f"📈 {asset_info.get('name_en', selected_asset)} / {asset_info.get('name_cn', selected_asset)}")
            
            # Get data with lookback
            asset_data = all_data[selected_asset].copy()
            cutoff_date = asset_data['Date'].max() - timedelta(days=lookback_days)
            asset_data_filtered = asset_data[asset_data['Date'] >= cutoff_date]
            
            # Display current metrics
            current = metrics_df[metrics_df['asset'] == selected_asset].iloc[0]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Composite BRI", f"{current['bri']:.2%}")
            col2.metric("Short-term", f"{current['short_bri']:.2%}")
            col3.metric("Mid-term", f"{current['mid_bri']:.2%}")
            col4.metric("Long-term", f"{current['long_bri']:.2%}")
            col5.metric("Daily Return", f"{current['daily_return']:.2%}",
                       delta=f"{current['daily_return']:.2%}")
            
            # Create plots
            create_indicator_plots(asset_data_filtered, selected_asset)
            
            # Download data option
            st.markdown("---")
            csv = asset_data_filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download Data (CSV)",
                data=csv,
                file_name=f"{selected_asset}_BRI_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()

