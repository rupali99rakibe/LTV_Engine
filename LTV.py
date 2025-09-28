import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page Config & Title
# -----------------------------
st.set_page_config(
    page_title="Fashion LTV Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)

# Top-aligned Dashboard Title
st.markdown('<h1 style="text-align: center; color: #2C3E50;">Fashion LTV Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv('LTV_Data.csv')

# -----------------------------
# Helper Functions
# -----------------------------
def simple_rfm(df):
    r = df.get('Recency', pd.Series(np.nan, index=df.index)).fillna(9999)
    f = df.get('Orders', pd.Series(0, index=df.index))
    m = df.get('Monetary Value', df['Last Order Amount']).fillna(0)
    r_score = 1 - (r / (r.max() + 1))
    f_score = f / (f.max() + 1)
    m_score = m / (m.max() + 1)
    rfm_score = 0.4*r_score + 0.3*f_score + 0.3*m_score
    return pd.DataFrame({'recency': r, 'frequency': f, 'monetary': m, 'rfm_score': rfm_score})

def churn_risk(score):
    return np.select([score>=0.6, score>=0.35], ['Low','Medium'], default='High')

def heuristic_ltv(row, months=6):
    freq = row['frequency']
    aov = row['monetary']/freq if freq>0 else row['monetary']
    return freq/12 * aov * months

def format_currency(x):
    if x >= 1e7: return f"₹{x/1e7:.2f} Cr"
    elif x >= 1e5: return f"₹{x/1e5:.2f} L"
    elif x >= 1e3: return f"₹{x/1e3:.2f} K"
    else: return f"₹{x:.0f}"

def plot_pie(data, title, colors):
    fig = go.Figure(go.Pie(labels=data.index, values=data.values, hole=0.4, marker_colors=colors))
    fig.update_layout(title_text=title, title_x=0.2, legend=dict(orientation="h", y=-0.1))
    return fig

def plot_bar(data, title, color_map=None):
    fig = px.bar(
        x=data.index, y=data.values, text=data.values, color=data.index,
        color_discrete_map=color_map, height=400
    )
    fig.update_layout(title_text=title, title_x=0.2, xaxis_title='', yaxis_title='Count', bargap=0.3)
    return fig

# -----------------------------
# Sidebar Filters
# -----------------------------
for f in ['State', 'District', 'Fashion Segment', 'Customer Name', 'Customer ID']:
    if f in df.columns:
        options = ['All'] + sorted(df[f].dropna().unique().tolist())
        sel = st.sidebar.selectbox(f, options)
        if sel != 'All':
            df = df[df[f]==sel]

# -----------------------------
# Compute Metrics
# -----------------------------
rfm = simple_rfm(df)
stats = pd.concat([df.reset_index(drop=True), rfm.reset_index(drop=True)], axis=1)
stats['churn_risk'] = churn_risk(stats['rfm_score'])
stats['LTV_6m'] = stats.apply(lambda r: heuristic_ltv(r,6), axis=1)
stats['LTV_12m'] = stats.apply(lambda r: heuristic_ltv(r,12), axis=1)

# Customer segmentation
stats['segment'] = np.select(
    [ (stats['Orders']>=4) & (stats['rfm_score']>=0.6),
      stats['Orders']<=1,
      stats['rfm_score']<0.35 ],
    ['Loyal','One-Timer','At-Risk'],
    default='Other'
)

# Segment subsets
segments_dict = {
    "Loyal": stats[stats['segment']=='Loyal'][['Customer ID','Customer Name']],
    "At-Risk": stats[stats['segment']=='At-Risk'][['Customer ID','Customer Name']],
    "One-Timer": stats[stats['segment']=='One-Timer'][['Customer ID','Customer Name']]
}

# -----------------------------
# KPI Cards
# -----------------------------
total_customers = stats.shape[0]
avg_orders = stats['Orders'].mean()
avg_aov = stats['Last Order Amount'].mean()
total_ltv_6m = stats['LTV_6m'].sum()

c1,c2,c3,c4 = st.columns([1.5,1.5,1.5,2])
c1.metric("Total Customers", f"{total_customers:,}")
c2.metric("Avg Orders", f"{avg_orders:.2f}")
c3.metric("Avg Spend", format_currency(avg_aov))
c4.metric("Estimated Revenue", format_currency(total_ltv_6m))

# -----------------------------
# Churn Pie & Customer Segment Bar
# -----------------------------
col1,col2 = st.columns(2)
with col1:
    churn_counts = stats['churn_risk'].value_counts().reindex(['Low','Medium','High']).fillna(0)
    colors = ['#2ECC71','#F1C40F','#E74C3C']
    st.plotly_chart(plot_pie(churn_counts,'Risk Segments Overview',colors), use_container_width=True)

with col2:
    seg_counts = stats['segment'].value_counts().reindex(['Loyal','At-Risk','One-Timer','Other']).fillna(0)
    color_map = {'Loyal':'#2ECC71','At-Risk':'#E74C3C','One-Timer':'#3498DB','Other':'#95A5A6'}
    st.plotly_chart(plot_bar(seg_counts,'Customer Segment Counts',color_map), use_container_width=True)

# -----------------------------
# LTV Histogram
# -----------------------------
stats['LTV_bin'] = pd.cut(stats['LTV_6m'], bins=10)
fig2 = px.histogram(stats, x='LTV_6m', nbins=40, color='LTV_bin', color_discrete_sequence=px.colors.sequential.Viridis)
fig2.update_layout(title_text='LTV Distribution', title_x=0.2, xaxis_title='LTV (₹)', yaxis_title='Number of Customers')
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Geography Bar
# -----------------------------
if 'State' in stats.columns:
    geo = stats.groupby('State').agg({'Customer ID':'count','LTV_6m':'sum'}).rename(columns={'Customer ID':'Customers'}).reset_index()
    fig4 = px.bar(geo, x='State', y='Customers', color='LTV_6m', text='Customers', color_continuous_scale='Viridis', height=400)
    fig4.update_layout(title_text='State-wise Customers', title_x=0.2, xaxis_title='State', yaxis_title='Number of Customers')
    st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# Customer Behavior Time Series
# -----------------------------
stats['Last Purchase Date'] = pd.to_datetime(stats['Last Purchase Date'], errors='coerce')
time_series = stats.groupby(pd.Grouper(key='Last Purchase Date', freq='M')).agg({'Orders':'sum','LTV_6m':'sum'}).reset_index()

col1,col2 = st.columns(2)
with col1:
    fig_ltv = px.line(time_series, x='Last Purchase Date', y='LTV_6m', title='Total LTV Trend Over Time', markers=True)
    fig_ltv.update_layout(xaxis_title='Month', yaxis_title='Total LTV (₹)')
    st.plotly_chart(fig_ltv, use_container_width=True)

with col2:
    fig_orders = px.line(time_series, x='Last Purchase Date', y='Orders', title='Total Orders Trend Over Time', markers=True)
    fig_orders.update_layout(xaxis_title='Month', yaxis_title='Total Orders')
    st.plotly_chart(fig_orders, use_container_width=True)

# -----------------------------
# Product/Fashion Segment Trend
# -----------------------------
product_col = 'Product' if 'Product' in stats.columns else 'Fashion Segment'
product_trend = stats.groupby([pd.Grouper(key='Last Purchase Date', freq='M'), product_col]).agg({'Orders':'sum'}).reset_index()
fig_prod = px.line(product_trend, x='Last Purchase Date', y='Orders', color=product_col, title=f'{product_col} Orders Trend Over Time', markers=True)
fig_prod.update_layout(xaxis_title='Month', yaxis_title='Total Orders')
st.plotly_chart(fig_prod, use_container_width=True)

# -----------------------------
# Create segment subsets
# -----------------------------
loyal_customers = stats[stats['segment'] == 'Loyal'][['Customer ID', 'Customer Name']]
at_risk_customers = stats[stats['segment'] == 'At-Risk'][['Customer ID', 'Customer Name']]
one_timer_customers = stats[stats['segment'] == 'One-Timer'][['Customer ID', 'Customer Name']]

# -----------------------------
# Download Buttons
# -----------------------------
st.markdown("## Download Customer Segments")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.download_button(
        'Download Full Data',
        data=stats.to_csv(index=False).encode('utf-8'),
        file_name='ltv_segmented.csv',
        key='full'
    )

with col2:
    # Example: pick one segment from your dict, e.g., 'Loyal'
    st.download_button(
        'Download Loyal Customers',
        data=segments_dict['Loyal'].to_csv(index=False).encode('utf-8'),
        file_name='loyal_customers.csv',
        key='loyal'
    )

# Second row of download buttons
col3, col4 = st.columns(2)
with col3:
    st.download_button(
        'Download At-Risk Customers', 
        data=at_risk_customers.to_csv(index=False).encode('utf-8'), 
        file_name='at_risk_customers.csv'
    )
with col4:
    st.download_button(
        'Download One-Timer Customers', 
        data=one_timer_customers.to_csv(index=False).encode('utf-8'), 
        file_name='one_timer_customers.csv'
    )
