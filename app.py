import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai
import json
import re
from datetime import datetime, timedelta
from forex_python.converter import CurrencyRates
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import webbrowser
from collections import defaultdict

# --- Configuration ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "AIzaSyDB1x5oKdlhXMDqXwulORcO9G78qjOy_b8")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Enhanced country data
COUNTRIES = {
    'USA': {
        'currency': 'USD',
        'stores': ['Amazon', 'Walmart'],
        'local_stores': ['Best Buy', 'Target', 'Walmart'],
        'icon': '🇺🇸',
        'timezone': 'America/New_York'
    },
    'India': {
        'currency': 'INR', 
        'stores': ['Amazon India', 'Flipkart'],
        'local_stores': ['Reliance Digital', 'Croma'],
        'icon': '🇮🇳',
        'timezone': 'Asia/Kolkata'
    },
    'UK': {
        'currency': 'GBP',
        'stores': ['Amazon UK', 'Argos'],
        'local_stores': ['Currys', 'John Lewis'],
        'icon': '🇬🇧',
        'timezone': 'Europe/London'
    }
}

# Initialize session state
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = defaultdict(dict)
if 'social_trends' not in st.session_state:
    st.session_state.social_trends = {}
if 'local_prices' not in st.session_state:
    st.session_state.local_prices = {}

# --- Core Functions ---
@st.cache_data(ttl=3600)
def get_exchange_rates():
    """Get exchange rates with fallback"""
    try:
        c = CurrencyRates()
        return {
            'USD': 1,
            'INR': c.get_rate('INR', 'USD'),
            'GBP': c.get_rate('GBP', 'USD')
        }
    except:
        return {'USD':1, 'INR':0.012, 'GBP':1.25}

def safe_convert_to_float(value, default=0.0):
    """Safely convert any value to float"""
    if value is None:
        return default
    try:
        if isinstance(value, str):
            value = re.sub(r'[^\d.]', '', value)
        return float(value)
    except (ValueError, TypeError):
        return default

def parse_response(response_text):
    """Robust JSON parsing with multiple fallbacks"""
    if not response_text:
        return None
    try:
        # Try direct JSON parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown
            match = re.search(r'```(?:json)?\n(.*?)\n```', response_text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            # Fallback to first JSON object
            json_str = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_str:
                return json.loads(json_str.group(0))
        return None
    except Exception:
        return None

def get_country_prices(product, country):
    """Get prices for a product in a specific country"""
    prompt = f"""Provide price for {product} in {country} from {COUNTRIES[country]['stores'][0]} in this format:
    {{
        "store": "StoreName",
        "price": 999.99,
        "shipping": 0.00,
        "url": "https://store.com/product"
    }}
    Price in {COUNTRIES[country]['currency']}, numbers only, no symbols"""
    
    try:
        response = model.generate_content(prompt)
        data = parse_response(response.text)
        
        # Ensure we're working with a dictionary
        if isinstance(data, list):
            if len(data) > 0:
                return data[0]  # Take first item if it's a list
            return None
        return data
    except Exception as e:
        st.error(f"Error getting prices for {country}: {str(e)}")
        return None

def fetch_all_prices(product):
    """Fetch prices from all countries in parallel"""
    start_time = time.time()
    exchange_rates = get_exchange_rates()
    results = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(get_country_prices, product, c): c for c in COUNTRIES}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, future in enumerate(as_completed(futures)):
            country = futures[future]
            data = future.result()
            
            if data and isinstance(data, dict):  # Ensure data is a dictionary
                price = safe_convert_to_float(data.get('price'))
                shipping = safe_convert_to_float(data.get('shipping'))
                total = price + shipping
                
                results.append({
                    'store': data.get('store', 'Unknown'),
                    'country': country,
                    'price': price,
                    'shipping': shipping,
                    'total': total,
                    'usd_price': total * exchange_rates.get(COUNTRIES[country]['currency'], 1),
                    'icon': COUNTRIES[country]['icon'],
                    'url': data.get('url', '#')
                })
            
            progress = (i + 1) / len(COUNTRIES)
            progress_bar.progress(progress)
            status_text.text(f"Scanned {i+1}/{len(COUNTRIES)} countries")
    
    st.success(f"Completed in {time.time() - start_time:.2f}s")
    return sorted(results, key=lambda x: x['usd_price']) if results else None
# --- New Features ---
def check_local_inventory(product, country, zip_code=None):
    """Mock local store inventory checker"""
    if not zip_code:
        return None
        
    # Mock data - replace with actual API calls
    mock_stores = [
        {
            'name': random.choice(COUNTRIES[country]['local_stores']),
            'price': round(random.uniform(0.8, 1.2) * 1000, 2),
            'distance': round(random.uniform(1, 10), 1),
            'stock': random.choice(['In Stock', 'Low Stock', 'Out of Stock']),
            'address': f"{random.randint(1, 100)} Main St, {zip_code}"
        } for _ in range(random.randint(2, 5))
    ]
    return sorted(mock_stores, key=lambda x: x['distance'])

def get_social_trends(product):
    """Mock social media trends - replace with Twitter/Instagram API"""
    return {
        'popularity': random.randint(30, 95),
        'sentiment': random.choice(['positive', 'neutral', 'negative']),
        'trending_hashtags': [
            f"#{product.replace(' ', '')}",
            f"#Buy{product.replace(' ', '')}",
            f"#Deal{product.split()[0]}"
        ],
        'mentions': random.randint(100, 5000)
    }

# --- Streamlit UI ---
st.set_page_config(page_title="Global Price Intelligence Pro", layout="wide")
st.title("🌐 Ultimate Shopping Assistant Pro")

# Tab layout
tab1, tab2, tab3 = st.tabs(["🛒 Price Comparison", "📈 Market Trends", "🔔 My Alerts"])

# --- Tab 1: Price Comparison ---
with tab1:
    # Country selection with dynamic theming
    country = st.selectbox("Select Country:", list(COUNTRIES.keys()), key="country_select")
    
    # Display country info
    st.header(f"{COUNTRIES[country]['icon']} {country} Price Comparison")
    st.caption(f"Currency: {COUNTRIES[country]['currency']} | Stores: {', '.join(COUNTRIES[country]['stores'])}")
    
    product = st.text_input("Enter product:", placeholder="iPhone 16 Pro 256GB", key=f"product_{country}")
    
    if product:
        # Online price comparison
        with st.spinner(f"Searching {', '.join(COUNTRIES[country]['stores'])}..."):
            prices = fetch_all_prices(product)
            
            if prices:
                # Filter prices for selected country
                country_prices = [p for p in prices if p['country'] == country]
                
                if country_prices:
                    df = pd.DataFrame(country_prices)
                    cheapest = df.iloc[0]
                    
                    # Display results
                    st.subheader("💻 Online Prices")
                    cols = st.columns(3)
                    cols[0].metric("Cheapest Option", 
                                 f"{COUNTRIES[country]['currency']} {cheapest['total']:.2f}",
                                 f"{cheapest['store']}")
                    cols[1].button("View Deal", on_click=lambda: webbrowser.open(cheapest['url']))
                    
                    # Price chart
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.barh(df['store'], df['total'], color='#4B8BBE')
                    ax.set_xlabel(f"Price ({COUNTRIES[country]['currency']})")
                    st.pyplot(fig)
                
                # Local store inventory
                st.subheader("📍 Local Store Availability")
                zip_code = st.text_input("Enter your zip/postal code:", key=f"zip_{country}")
                
                if zip_code:
                    local_stores = check_local_inventory(product, country, zip_code)
                    if local_stores:
                        for store in local_stores[:3]:  # Show top 3 closest
                            with st.expander(f"🏬 {store['name']} ({store['distance']} km)"):
                                cols = st.columns(3)
                                cols[0].metric("Price", f"{COUNTRIES[country]['currency']} {store['price']:.2f}")
                                cols[1].metric("Stock", store['stock'])
                                cols[2].write(f"Address: {store['address']}")
                    else:
                        st.warning("No local stores found with this product")
                
                # Social media trends
                st.subheader("📱 Social Media Buzz")
                trends = get_social_trends(product)
                cols = st.columns(3)
                cols[0].metric("Popularity", f"{trends['popularity']}/100")
                cols[1].metric("Sentiment", trends['sentiment'].title())
                cols[2].metric("Mentions", trends['mentions'])
                
                st.write("**Trending Hashtags:**")
                for tag in trends['trending_hashtags']:
                    st.write(f"- {tag}")
                
                # Price alert
                with st.expander("🔔 Set Price Alert"):
                    alert_price = st.number_input(
                        "Notify me when price drops below:", 
                        value=float(cheapest['total']) * 0.9 if country_prices else 0,
                        key=f"alert_{country}"
                    )
                    if st.button("Create Alert", key=f"alert_btn_{country}"):
                        st.session_state.alerts.append({
                            'product': product,
                            'target': alert_price,
                            'current': cheapest['total'] if country_prices else 0,
                            'country': country,
                            'date': datetime.now().strftime('%Y-%m-%d')
                        })
                        st.success("Price alert created!")
            else:
                st.error("No prices found for this product")

# [Rest of your tabs (tab2 and tab3) remain the same]
# --- Tab 2: Market Trends ---
with tab2:
    st.header("E-Commerce Trend Analyzer")
    st.markdown("Discover emerging product trends across major retailers")
    
    # Trend analysis parameters
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Category", 
                              ["Electronics", "Fashion", "Home & Kitchen", "Beauty"])
    with col2:
        timeframe = st.selectbox("Timeframe", 
                               ["Last 7 days", "Last 30 days", "Last 90 days"])
    
    if st.button("Analyze Trends"):
        with st.spinner("🧠 Analyzing trends with AI..."):
            try:
                # Simulated trend data (replace with actual analysis)
                trends = [
                    {"product": "Wireless Earbuds", "growth": 45, "avg_price": 89.99},
                    {"product": "Yoga Mats", "growth": 32, "avg_price": 29.99},
                    {"product": "Air Fryers", "growth": 28, "avg_price": 119.99},
                ]
                
                # Display trends
                st.subheader(f"Emerging Trends in {category}")
                
                # Trend visualization
                fig, ax = plt.subplots(figsize=(8, 4))
                trend_df = pd.DataFrame(trends)
                ax.barh(trend_df['product'], trend_df['growth'], color='#4B8BBE')
                ax.set_xlabel('Growth Percentage')
                ax.set_title('Product Popularity Growth')
                st.pyplot(fig)
                
                # AI trend analysis
                if genai:
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    response = model.generate_content(
                        f"Analyze these e-commerce trends: {trends} in {category} "
                        f"over {timeframe}. Provide 2-3 key insights."
                    )
                    
                    st.success("🔮 AI Trend Insights:")
                    st.write(response.text)
                else:
                    st.warning("Gemini API not available - using OpenAI instead")
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "Provide 2-3 concise insights about e-commerce trends."},
                            {"role": "user", "content": f"Analyze these trends {trends} in {category} over {timeframe}"}
                        ]
                    )
                    st.success("🔮 AI Trend Insights:")
                    st.write(response.choices[0].message.content)
                
                # Price distribution
                st.subheader("Price Distribution Analysis")
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                ax2.scatter(trend_df['product'], trend_df['avg_price'], color='#FF6E54', s=100)
                ax2.set_ylabel('Average Price ($)')
                ax2.set_title('Price vs. Popularity')
                st.pyplot(fig2)
            
            except Exception as e:
                st.error(f"Error analyzing trends: {str(e)}")

# --- Tab 3: Price Alerts ---
with tab3:
    st.header("Your Price Alerts")
    
    if not st.session_state.alerts:
        st.info("No active alerts")
    else:
        for i, alert in enumerate(st.session_state.alerts):
            with st.container(border=True):
                cols = st.columns([3, 1, 1, 1])
                with cols[0]:
                    st.write(f"**{alert['product']}**")
                    st.caption(f"{alert['country']} | Alert below ${alert['target']:.2f}")
                
                with cols[1]:
                    diff = alert['current'] - alert['target']
                    st.metric("Current", f"${alert['current']:.2f}", f"{diff:.2f}")
                
                with cols[2]:
                    st.write("Last checked")
                    st.caption(alert['date'])
                
                with cols[3]:
                    if st.button("Delete", key=f"del_{i}"):
                        st.session_state.alerts.pop(i)
                        st.rerun()

# --- Sidebar ---
with st.sidebar:
    st.write("### 🌍 Supported Countries")
    for country in COUNTRIES:
        st.write(f"{COUNTRIES[country]['icon']} {country}")
    
    st.write("### ✨ Features")
    st.write("- Real-time price tracking")
    st.write("- Historical price charts")
    st.write("- Local store inventory")
    st.write("- Social media analysis")
    st.write("- Market trend intelligence")

# --- Footer ---
st.caption(f"Data updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | v2.1.0")