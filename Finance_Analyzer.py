import io
import re
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Private Finance Statement Analyzer', page_icon='💳', layout='wide')

CATEGORY_RULES = {
    'Income': [r'payroll', r'salary', r'direct dep', r'direct deposit', r'ach credit', r'interest payment', r'dividend', r'refund'],
    'Housing': [r'rent', r'mortgage', r'property tax', r'hoa'],
    'Utilities': [r'eversource', r'utility', r'water', r'electric', r'gas bill', r'internet', r'comcast', r'verizon', r'at&t', r'tmobile'],
    'Groceries': [r'walmart', r'costco', r'trader joe', r'whole foods', r'stop & shop', r'kroger', r'aldi', r'instacart', r'grocery'],
    'Dining': [r'restaurant', r'uber eats', r'doordash', r'grubhub', r'dunkin', r'starbucks', r'mcdonald', r'chipotle', r'panera', r'cafe'],
    'Transport': [r'uber', r'lyft', r'shell', r'exxon', r'gas station', r'parking', r'mta', r'amtrak', r'toll'],
    'Shopping': [r'amazon', r'target', r'ebay', r'etsy', r'best buy', r'macy'],
    'Health': [r'cvs', r'walgreens', r'pharmacy', r'dental', r'medical', r'hospital', r'vision'],
    'Insurance': [r'insurance', r'geico', r'allstate', r'state farm', r'progressive'],
    'Subscriptions': [r'netflix', r'spotify', r'apple.com/bill', r'youtube', r'hulu', r'prime video', r'adobe', r'chatgpt', r'openai'],
    'Travel': [r'hotel', r'airbnb', r'booking.com', r'delta', r'united', r'american airlines', r'southwest'],
    'Transfers': [r'zelle', r'venmo', r'cash app', r'transfer', r'payment to', r'payment from', r'internal transfer'],
    'Fees': [r'fee', r'finance charge', r'late charge', r'interest charge', r'overdraft'],
    'Cash': [r'atm', r'cash withdrawal'],
}

ESSENTIAL_CATEGORIES = {'Housing', 'Utilities', 'Groceries', 'Health', 'Insurance', 'Transport'}
NON_ESSENTIAL_CATEGORIES = {'Dining', 'Shopping', 'Subscriptions', 'Travel', 'Cash'}


def normalize_columns(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map = {}
    for c in df.columns:
        if c in ['date', 'posted date', 'transaction date', 'posting date']:
            rename_map[c] = 'date'
        elif c in ['description', 'details', 'merchant', 'transaction description', 'name', 'memo']:
            rename_map[c] = 'description'
        elif c in ['amount', 'transaction amount']:
            rename_map[c] = 'amount'
        elif c in ['debit', 'withdrawal']:
            rename_map[c] = 'debit'
        elif c in ['credit', 'deposit']:
            rename_map[c] = 'credit'
        elif c in ['type', 'transaction type']:
            rename_map[c] = 'type'
    df = df.rename(columns=rename_map)
    return df


def parse_amount(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.number)):
        return float(val)
    s = str(val).strip().replace('$', '').replace(',', '')
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]
    try:
        return float(s)
    except:
        return np.nan


def standardize_transactions(df, source_name='uploaded'):
    df = normalize_columns(df.copy())
    if 'date' not in df.columns or 'description' not in df.columns:
        raise ValueError('Required columns not found. Please include Date and Description, plus Amount or Debit/Credit columns.')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].notna()].copy()
    if 'amount' in df.columns:
        df['net_amount'] = df['amount'].apply(parse_amount)
    else:
        debit = df['debit'].apply(parse_amount) if 'debit' in df.columns else 0
        credit = df['credit'].apply(parse_amount) if 'credit' in df.columns else 0
        df['net_amount'] = credit.fillna(0) - debit.fillna(0)
    df = df[df['net_amount'].notna()].copy()
    df['description'] = df['description'].astype(str).str.strip()
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['source_file'] = source_name
    return df[['date', 'month', 'description', 'net_amount', 'source_file']]


def categorize(desc, amt):
    d = str(desc).lower()
    for cat, patterns in CATEGORY_RULES.items():
        for p in patterns:
            if re.search(p, d):
                return cat
    if amt > 0:
        return 'Income'
    return 'Other'


def analyze(df):
    df = df.copy()
    df['category'] = [categorize(d, a) for d, a in zip(df['description'], df['net_amount'])]
    df['income'] = np.where(df['net_amount'] > 0, df['net_amount'], 0.0)
    df['expense'] = np.where(df['net_amount'] < 0, -df['net_amount'], 0.0)
    monthly = df.groupby('month', as_index=False).agg(
        income=('income', 'sum'),
        expenses=('expense', 'sum')
    )
    monthly['net_savings'] = monthly['income'] - monthly['expenses']
    monthly['savings_rate_pct'] = np.where(monthly['income'] > 0, (monthly['net_savings'] / monthly['income']) * 100, 0)
    cat_month = df[df['expense'] > 0].groupby(['month', 'category'], as_index=False)['expense'].sum()
    cat_total = df[df['expense'] > 0].groupby('category', as_index=False)['expense'].sum().sort_values('expense', ascending=False)
    return df, monthly, cat_month, cat_total


def savings_suggestions(monthly, cat_total):
    tips = []
    total_exp = float(cat_total['expense'].sum()) if not cat_total.empty else 0.0
    avg_income = float(monthly['income'].mean()) if not monthly.empty else 0.0
    avg_exp = float(monthly['expenses'].mean()) if not monthly.empty else 0.0
    avg_save_rate = float(monthly['savings_rate_pct'].mean()) if not monthly.empty else 0.0

    if avg_income > 0 and avg_save_rate < 20:
        needed = max(0, 0.2 * avg_income - (avg_income - avg_exp))
        tips.append(f'Your average savings rate is {avg_save_rate:.1f}%. To reach a 20% savings rate, target about ${needed:,.0f} in extra monthly savings.')

    for _, row in cat_total.head(5).iterrows():
        cat = row['category']
        amt = float(row['expense'])
        pct = (amt / total_exp * 100) if total_exp else 0
        if cat in NON_ESSENTIAL_CATEGORIES and pct >= 8:
            cut_10 = amt * 0.10
            cut_20 = amt * 0.20
            tips.append(f'{cat} is a high discretionary category at ${amt:,.0f} total ({pct:.1f}% of expenses). Cutting 10% could save about ${cut_10:,.0f}; cutting 20% could save about ${cut_20:,.0f}.')
        elif cat in ESSENTIAL_CATEGORIES and pct >= 20:
            tips.append(f'{cat} is one of your biggest essential categories at ${amt:,.0f} total ({pct:.1f}% of expenses). Review for optimization opportunities such as plan changes, rate shopping, or bill negotiation.')

    subs = cat_total[cat_total['category'] == 'Subscriptions']
    if not subs.empty and float(subs['expense'].iloc[0]) > 50:
        tips.append('Subscriptions appear meaningful. Review duplicate streaming, software, and membership charges for cancellations or downgrades.')

    dining = cat_total[cat_total['category'] == 'Dining']
    groceries = cat_total[cat_total['category'] == 'Groceries']
    if not dining.empty and not groceries.empty:
        if float(dining['expense'].iloc[0]) > 0.6 * float(groceries['expense'].iloc[0]):
            tips.append('Dining spend is high relative to groceries. Shifting a few meals per week from takeout/restaurants to groceries may materially improve monthly savings.')

    if not tips:
        tips.append('Your spending appears fairly balanced. Focus on consistent saving automation and reviewing the top 2 expense categories for incremental improvements.')
    return tips


def export_summary(monthly, cat_total):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        monthly.to_excel(writer, index=False, sheet_name='Monthly Summary')
        cat_total.to_excel(writer, index=False, sheet_name='Category Totals')
    return buf.getvalue()


st.title('💳 Private Finance Statement Analyzer')
st.caption('Upload bank, savings, and credit card statements to analyze income, spending, categories, and savings opportunities — processed in memory only.')

with st.sidebar:
    st.header('Privacy first')
    st.success('Files are processed only during your session and are not intentionally saved by the app.')
    st.write('Recommended file types: CSV or Excel exports from your bank or card provider.')
    st.write('Expected columns: Date, Description, and either Amount or Debit/Credit.')

files = st.file_uploader('Upload one or more statements', type=['csv', 'xlsx', 'xls'], accept_multiple_files=True)

if not files:
    st.info('Upload statements to begin. Example sources: checking, savings, and credit card exports.')
    st.markdown('''
### What this app does
- Combines multiple statement files
- Calculates monthly income, expenses, and net savings
- Categorizes spending automatically
- Highlights top spending areas
- Suggests where you may be able to cut back
- Avoids persistent file storage in app logic
''')
    st.stop()

all_frames = []
errors = []
for f in files:
    try:
        if f.name.lower().endswith('.csv'):
            raw = pd.read_csv(f)
        else:
            raw = pd.read_excel(f)
        all_frames.append(standardize_transactions(raw, f.name))
    except Exception as e:
        errors.append(f'{f.name}: {e}')

if errors:
    for e in errors:
        st.error(e)

if not all_frames:
    st.stop()

transactions = pd.concat(all_frames, ignore_index=True).drop_duplicates()
transactions, monthly, cat_month, cat_total = analyze(transactions)

suggestions = savings_suggestions(monthly, cat_total)

col1, col2, col3, col4 = st.columns(4)
col1.metric('Months analyzed', int(monthly.shape[0]))
col2.metric('Total income', f"${monthly['income'].sum():,.0f}")
col3.metric('Total expenses', f"${monthly['expenses'].sum():,.0f}")
col4.metric('Net savings', f"${monthly['net_savings'].sum():,.0f}")

st.subheader('Monthly trend')
fig = go.Figure()
fig.add_trace(go.Bar(x=monthly['month'], y=monthly['income'], name='Income'))
fig.add_trace(go.Bar(x=monthly['month'], y=monthly['expenses'], name='Expenses'))
fig.add_trace(go.Scatter(x=monthly['month'], y=monthly['net_savings'], name='Net savings', mode='lines+markers'))
fig.update_layout(barmode='group', height=420, xaxis_title='', yaxis_title='Amount ($)')
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader('Expenses by category')
    pie = px.pie(cat_total.head(10), names='category', values='expense', hole=0.45)
    pie.update_layout(height=420)
    st.plotly_chart(pie, use_container_width=True)
with c2:
    st.subheader('Top categories')
    bar = px.bar(cat_total.head(10).sort_values('expense'), x='expense', y='category', orientation='h')
    bar.update_layout(height=420, xaxis_title='Amount ($)', yaxis_title='')
    st.plotly_chart(bar, use_container_width=True)

st.subheader('Savings suggestions')
for s in suggestions:
    st.write(f'- {s}')

st.subheader('Monthly summary')
st.dataframe(monthly, use_container_width=True)

st.subheader('Category totals')
st.dataframe(cat_total, use_container_width=True)

with st.expander('See categorized transactions'):
    st.dataframe(transactions.sort_values('date', ascending=False), use_container_width=True, height=350)

excel_bytes = export_summary(monthly, cat_total)
st.download_button('Download summary workbook', data=excel_bytes, file_name='finance_summary.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

st.markdown('---')
st.markdown('''
### Hosting notes
- Safe for GitHub and Streamlit deployment.
- The app code does not write uploaded files to disk.
- Uploaded files are processed in memory for analysis during the active session.
- Do not add custom logging that captures uploaded statement contents.
''')
