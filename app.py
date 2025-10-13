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
    page_icon="./fashion_icon.png",  
    layout="wide"
)
st.markdown('<h1 style="text-align: center; color: #2C3E50;">Fashion LTV Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv('LTV_Data.csv')
df['Last Purchase Date'] = pd.to_datetime(df['Last Purchase Date'], errors='coerce')

# -----------------------------
# Sidebar Filters
# -----------------------------
filtered_df = df.copy()
filtered_df['Purchase_Year'] = filtered_df['Last Purchase Date'].dt.year
filtered_df['Purchase_Month'] = filtered_df['Last Purchase Date'].dt.month
months_dict = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

# Add sidebar filters
for f in ['State', 'District', 'Fashion Segment', 'Customer Name', 'Customer ID', 'Gender']:
    if f in filtered_df.columns:
        options = ['All'] + sorted(filtered_df[f].dropna().unique().tolist())
        sel = st.sidebar.selectbox(f, options)
        if sel != 'All':
            filtered_df = filtered_df[filtered_df[f] == sel]

# Year filter
years = ['All'] + sorted(filtered_df['Purchase_Year'].dropna().unique().tolist())
sel_year = st.sidebar.selectbox('Year', years)
if sel_year != 'All':
    filtered_df = filtered_df[filtered_df['Purchase_Year'] == sel_year]

# Month filter
month_options = ['All'] + [months_dict[m] for m in sorted(filtered_df['Purchase_Month'].dropna().unique())]
sel_month = st.sidebar.selectbox('Month', month_options)
if sel_month != 'All':
    sel_month_num = [k for k,v in months_dict.items() if v == sel_month][0]
    filtered_df = filtered_df[filtered_df['Purchase_Month'] == sel_month_num]

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

axis_style = dict(
    title=dict(font=dict(color='black')),
    tickfont=dict(color='black')
)

# -----------------------------
# Plot Functions
# -----------------------------
def plot_pie(data, title, colors):
    fig = go.Figure(go.Pie(labels=data.index, values=data.values, hole=0.4, marker_colors=colors))
    fig.update_layout(
        title=dict(text=title, font=dict(color='black')),
        title_x=0.2,
        legend=dict(orientation="h", y=-0.1, font=dict(color='black')),
        legend_title=dict(text="Segments", font=dict(color='black'))
    )
    return fig

def plot_bar(data, title, color_map=None):
    fig = px.bar(
        x=data.index,
        y=data.values,
        text=data.values,
        color=data.index,
        color_discrete_map=color_map,
        height=400
    )
    fig.update_layout(
        title=dict(text=title, font=dict(color='black')),
        title_x=0.2,
        xaxis_title='',
        yaxis_title='Count',
        bargap=0.3,
        xaxis=axis_style,
        yaxis=axis_style,
        legend_title=dict(text="Category", font=dict(color='black')),
        legend=dict(font=dict(color='black'))
    )
    return fig

def sparkline(data, color):
    fig = go.Figure(go.Scatter(y=data, mode='lines+markers', line=dict(color=color, width=2), marker=dict(size=4)))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0,r=0,t=0,b=0), height=50)
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def growth_indicator(value):
    if value > 0: return f"<span style='color:#2ECC71;'>🔺 {value*100:.1f}%</span>"
    elif value < 0: return f"<span style='color:#E74C3C;'>🔻 {abs(value*100):.1f}%</span>"
    else: return "<span style='color:gray;'>—</span>"

# -----------------------------
# Compute Metrics
# -----------------------------
rfm = simple_rfm(filtered_df)
stats = pd.concat([filtered_df.reset_index(drop=True), rfm.reset_index(drop=True)], axis=1)
stats['churn_risk'] = churn_risk(stats['rfm_score'])
stats['Total_Spend'] = stats.apply(lambda r: heuristic_ltv(r,6), axis=1) # Updated here
stats['LTV_12mos'] = stats.apply(lambda r: heuristic_ltv(r,12), axis=1)

# Customer segmentation
stats['segment'] = np.select(
    [ (stats['Orders']>=4) & (stats['rfm_score']>=0.6),
      stats['Orders']<=1,
      stats['rfm_score']<0.35 ],
    ['Loyal','One-Timer','At-Risk'],
    default='Other'
)

segments_dict = {
    "Loyal": stats[stats['segment']=='Loyal'][['Customer ID','Customer Name']],
    "At-Risk": stats[stats['segment']=='At-Risk'][['Customer ID','Customer Name']],
    "One-Timer": stats[stats['segment']=='One-Timer'][['Customer ID','Customer Name']]
}

# -----------------------------
# Colors & Palette
# -----------------------------
palette = {'background':'#ffffff','text':'#2C3E50','highlight':'#34495E'}

# -----------------------------
# KPI Cards
# -----------------------------
st.markdown("### Key Business Highlights")
st.markdown("""
These cards show important business performance numbers:
- **Total Customers** → How many customers are active in our data.  
- **Average Orders** → The average number of orders per customer.  
- **Average Spend** → How much an average customer spends.  
- **Total Spend** → Total customer spending (based on heuristic LTV).  
The small trend line shows whether these values are going up or down.
""") # Updated description

kpi_style = f"""
<style>
.kpi-card {{ background-color: {palette['background']}; color: {palette['text']}; padding:20px; border-radius:12px; text-align:center; box-shadow:0 4px 8px rgba(0,0,0,0.1); }}
.kpi-icon {{ font-size:28px; margin-bottom:6px; }}
.kpi-title {{ font-size:16px; font-weight:600; color:{palette['highlight']}; margin-bottom:6px; }}
.kpi-value {{ font-size:22px; font-weight:bold; }}
.kpi-growth {{ font-size:14px; margin-top:4px; }}
.kpi-sparkline {{ margin-top:6px; }}
.icon-customers {{ color: #3498DB; }}
.icon-orders {{ color: #9B59B6; }}
.icon-spend {{ color: #F39C12; }}
.icon-revenue {{ color: #2ECC71; }}
</style>
"""
st.markdown(kpi_style, unsafe_allow_html=True)

# -----------------------------
# Monthly Trends & KPIs
# -----------------------------
monthly = stats.groupby(pd.Grouper(key='Last Purchase Date', freq='M')).agg({
    'Customer ID': 'nunique',
    'Orders': 'mean',
    'Last Order Amount': 'mean',
    'Total_Spend': 'sum' # Updated here
}).reset_index()

total_customers = stats.shape[0]
avg_orders = stats['Orders'].mean()
avg_aov = stats['Last Order Amount'].mean()
total_spend_val = stats['Total_Spend'].sum() # Updated here

if len(monthly) >= 2:
    latest, prev = monthly.iloc[-1], monthly.iloc[-2]
    growth_customers = (latest['Customer ID'] - prev['Customer ID']) / prev['Customer ID'] if prev['Customer ID'] else 0
    growth_orders = (latest['Orders'] - prev['Orders']) / prev['Orders'] if prev['Orders'] else 0
    growth_aov = (latest['Last Order Amount'] - prev['Last Order Amount']) / prev['Last Order Amount'] if prev['Last Order Amount'] else 0
    growth_spend = (latest['Total_Spend'] - prev['Total_Spend']) / prev['Total_Spend'] if prev['Total_Spend'] else 0 # Updated here
else:
    growth_customers = growth_orders = growth_aov = growth_spend = 0

monthly_customers = monthly['Customer ID'].tolist() if len(monthly) >= 2 else [total_customers]
monthly_orders = monthly['Orders'].tolist() if len(monthly) >= 2 else [avg_orders]
monthly_aov = monthly['Last Order Amount'].tolist() if len(monthly) >= 2 else [avg_aov]
monthly_spend = monthly['Total_Spend'].tolist() if len(monthly) >= 2 else [total_spend_val] # Updated here

c1,c2,c3,c4 = st.columns([1.5,1.5,1.5,2])
with c1:
    spark_html = sparkline(monthly_customers,"#3498DB")
    st.markdown(f"<div class='kpi-card'><div class='kpi-icon icon-customers'>👥</div><div class='kpi-title'>Total Customers</div><div class='kpi-value'>{total_customers:,}</div><div class='kpi-growth'>{growth_indicator(growth_customers)}</div><div class='kpi-sparkline'>{spark_html}</div></div>", unsafe_allow_html=True)
with c2:
    spark_html = sparkline(monthly_orders,"#9B59B6")
    st.markdown(f"<div class='kpi-card'><div class='kpi-icon icon-orders'>📦</div><div class='kpi-title'>Avg Orders</div><div class='kpi-value'>{avg_orders:.2f}</div><div class='kpi-growth'>{growth_indicator(growth_orders)}</div><div class='kpi-sparkline'>{spark_html}</div></div>", unsafe_allow_html=True)
with c3:
    spark_html = sparkline(monthly_aov,"#F39C12")
    st.markdown(f"<div class='kpi-card'><div class='kpi-icon icon-spend'>💰</div><div class='kpi-title'>Avg Spend</div><div class='kpi-value'>{format_currency(avg_aov)}</div><div class='kpi-growth'>{growth_indicator(growth_aov)}</div><div class='kpi-sparkline'>{spark_html}</div></div>", unsafe_allow_html=True)
with c4:
    spark_html = sparkline(monthly_spend,"#2ECC71")
    st.markdown(f"<div class='kpi-card'><div class='kpi-icon icon-revenue'>📈</div><div class='kpi-title'>Total Spend </div><div class='kpi-value'>{format_currency(total_spend_val)}</div><div class='kpi-growth'>{growth_indicator(growth_spend)}</div><div class='kpi-sparkline'>{spark_html}</div></div>", unsafe_allow_html=True) # Updated display title

# -----------------------------
# Churn Pie & Segment Bar
# -----------------------------
st.markdown("### Customer Risk and Loyalty Overview")
st.markdown("""
- **Left Chart (Pie):** Shows which customers are safe (Low risk), need attention (Medium), or likely to stop buying (High).  
- **Right Chart (Bar):** Groups customers by loyalty: Loyal (repeat buyers), At-Risk (might churn), One-Timer (bought once).  
These visuals help you see which customer group needs action.
""")

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
# Spend Histogram
# -----------------------------
st.markdown("### Spending Patterns")
st.markdown("""
This chart shows how much customers spend in total.  
It helps identify if most people are low, medium, or high spenders.
""") # Updated description

stats['Spend_range'] = pd.cut(
    stats['Total_Spend'], # Updated here
    bins=np.linspace(stats['Total_Spend'].min(), stats['Total_Spend'].max(), 11) # Updated here
)
bin_labels = [str(b) for b in stats['Spend_range'].cat.categories]
fig2 = px.histogram(
    stats,
    x='Total_Spend', # Updated here
    nbins=40,
    color='Spend_range',
    category_orders={"Spend_range": bin_labels},
    color_discrete_sequence=px.colors.sequential.Viridis
)
fig2.update_layout(
    title=dict(text='Customer Spending Patterns', font=dict(color='black')),
    title_x=0.2,
    xaxis_title='Total Spend (₹)', # Updated label
    yaxis_title='Customers',
    xaxis=axis_style,
    yaxis=axis_style,
    legend=dict(font=dict(color='black')),
    legend_title=dict(text="Spend Range", font=dict(color='black'))
)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# State-wise Bar
# -----------------------------
st.markdown("### State-wise Distribution")
st.markdown("""
This bar chart shows the number of customers in each state and how much they spent.  
It helps you see which states have more active or high-value buyers.
""")

if 'State' in stats.columns: 
    geo = stats.groupby('State').agg({'Customer ID': 'count', 'Total_Spend': 'sum'}) \
        .rename(columns={'Customer ID': 'Customers'}).reset_index() # Updated here
    
    fig4 = px.bar(
        geo,
        x='State',
        y='Customers',
        color='Total_Spend', # Updated here
        text='Customers',
        color_continuous_scale='Viridis',
        height=400
    )
    
    fig4.update_layout(
        title=dict(text='State-wise Customers', font=dict(color='black', size=16)),
        title_x=0.2,
        xaxis_title='State',
        yaxis_title='Number of Customers',
        xaxis=axis_style,
        yaxis=axis_style,
        legend=dict(font=dict(color='black')),
        legend_title=dict(text="Total Spend", font=dict(color='black')) # Updated label
    )

    fig4.update_coloraxes(
        colorbar=dict(
            title=dict(font=dict(color='black')),
            tickfont=dict(color='black')
        )
    )

    st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# Orders & Spend Trends
# -----------------------------
st.markdown("### Purchase and Spend Trends Over Time")
st.markdown("""
These line charts track how orders and spending change month by month.  
They show whether our business is growing or slowing down.
""")

time_series = stats.groupby(pd.Grouper(key='Last Purchase Date', freq='M')).agg({'Orders':'sum','Total_Spend':'sum'}).reset_index() # Updated here
col1,col2 = st.columns(2)
with col1:
    fig_spend = px.line(time_series, x='Last Purchase Date', y='Total_Spend', title='Total Spend Trend Over Time', markers=True) # Updated here
    fig_spend.update_layout(
        title=dict(text='Total Spend Trend Over Time', font=dict(color='black')), # Updated title
        xaxis_title='Year', yaxis_title='Total Spend (₹)',
        xaxis=axis_style, yaxis=axis_style,
        legend=dict(font=dict(color='black')),
        legend_title=dict(text="Legend", font=dict(color='black'))
    )
    st.plotly_chart(fig_spend, use_container_width=True)

with col2:
    fig_orders = px.line(time_series, x='Last Purchase Date', y='Orders', title='Total Orders Trend Over Time', markers=True)
    fig_orders.update_layout(
        title=dict(text='Total Orders Trend Over Time', font=dict(color='black')),
        xaxis_title='Year', yaxis_title='Total Orders',
        xaxis=axis_style, yaxis=axis_style,
        legend=dict(font=dict(color='black')),
        legend_title=dict(text="Legend", font=dict(color='black'))
    )
    st.plotly_chart(fig_orders, use_container_width=True)

# -----------------------------
# Product/Fashion Segment Trend
# -----------------------------
st.markdown("### Fashion Segment Performance")
st.markdown("""
This chart compares how different fashion segments or product categories perform over time.  
You can see which product types are trending up or down.
""")

product_col = 'Product' if 'Product' in stats.columns else 'Fashion Segment'
product_trend = stats.groupby([pd.Grouper(key='Last Purchase Date', freq='M'), product_col]).agg({'Orders':'sum'}).reset_index()
fig_prod = px.line(product_trend, x='Last Purchase Date', y='Orders', color=product_col, title=f'{product_col} Orders Trend Over Time', markers=True)
fig_prod.update_layout(
    title=dict(text=f'{product_col} Orders Trend Over Time', font=dict(color='black')),
    xaxis_title='Month', yaxis_title='Total Orders',
    xaxis=axis_style, yaxis=axis_style,
    legend=dict(font=dict(color='black')),
    legend_title=dict(text=product_col, font=dict(color='black'))
)
st.plotly_chart(fig_prod, use_container_width=True)

# -----------------------------
# Download Buttons
# -----------------------------
st.markdown("### Download Reports")
st.markdown("""
You can download complete customer data or just specific groups like Loyal or At-Risk customers for further action or marketing campaigns.
""")

col1, col2 = st.columns(2)
with col1:
    st.download_button('Download Full Data', data=stats.to_csv(index=False).encode('utf-8'), file_name='ltv_segmented.csv', key='full')
with col2:
    st.download_button('Download Loyal Customers', data=segments_dict['Loyal'].to_csv(index=False).encode('utf-8'), file_name='loyal_customers.csv', key='loyal')

col3, col4 = st.columns(2)
with col3:
    st.download_button('Download At-Risk Customers', data=segments_dict['At-Risk'].to_csv(index=False).encode('utf-8'), file_name='at_risk_customers.csv')
with col4:
    st.download_button('Download One-Timer Customers', data=segments_dict['One-Timer'].to_csv(index=False).encode('utf-8'), file_name='one_timer_customers.csv')