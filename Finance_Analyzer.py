import io
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Finance Statement Analyzer', page_icon='💼', layout='wide')

CATEGORY_RULES = {
    'Income': [r'payroll', r'salary', r'direct deposit', r'direct dep', r'ach credit', r'interest', r'dividend', r'refund', r'cashback reward', r'payment thank you'],
    'Housing': [r'rent', r'mortgage', r'property tax', r'hoa'],
    'Utilities': [r'utility', r'electric', r'gas', r'water', r'internet', r'comcast', r'xfinity', r'optimum', r'phone', r'verizon', r't-mobile', r'at&t'],
    'Groceries': [r'walmart', r'costco', r'trader joe', r'whole foods', r'aldi', r'stop & shop', r'kroger', r'grocery', r'instacart'],
    'Dining': [r'restaurant', r'grubhub', r'ubereats', r'uber eats', r'doordash', r'dunkin', r'starbucks', r'chipotle', r'mcdonald', r'panera', r'pizza'],
    'Transport': [r'uber', r'lyft', r'shell', r'exxon', r'chevron', r'fuel', r'gas station', r'parking', r'toll', r'mta', r'nj transit'],
    'Shopping': [r'amazon', r'target', r'best buy', r'macy', r'etsy', r'ebay', r'old navy', r'nike', r'kohls'],
    'Health': [r'cvs', r'walgreens', r'pharmacy', r'medical', r'dental', r'hospital', r'urgent care', r'vision'],
    'Insurance': [r'insurance', r'geico', r'progressive', r'allstate', r'state farm'],
    'Subscriptions': [r'netflix', r'spotify', r'hulu', r'apple.com/bill', r'youtube', r'prime video', r'adobe', r'openai', r'chatgpt', r'icloud'],
    'Travel': [r'airbnb', r'hotel', r'booking.com', r'delta', r'united', r'southwest', r'american airlines'],
    'Transfers': [r'zelle', r'venmo', r'cash app', r'transfer', r'payment from', r'payment to', r'ach debit', r'ach withdrawal'],
    'Fees': [r'fee', r'finance charge', r'late fee', r'interest charge', r'overdraft'],
    'Cash': [r'atm', r'cash withdrawal'],
}

NON_ESSENTIAL = {'Dining', 'Shopping', 'Subscriptions', 'Travel', 'Cash'}
ESSENTIAL = {'Housing', 'Utilities', 'Groceries', 'Health', 'Insurance', 'Transport'}
DATE_PATTERN = re.compile(r'(\d{1,2}/\d{1,2}(?:/\d{2,4})?)')
AMOUNT_PATTERN = re.compile(r'\(?\$?[-]?[\d,]+\.\d{2}\)?')


def parse_amount(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip().replace('$', '').replace(',', '')
    if not s:
        return np.nan
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]
    try:
        return float(s)
    except Exception:
        return np.nan


def classify_category(desc, amount):
    text = str(desc).lower()
    for cat, pats in CATEGORY_RULES.items():
        if any(re.search(p, text) for p in pats):
            return cat
    return 'Income' if amount > 0 else 'Other'


def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    mapping = {
        'posted date': 'date', 'transaction date': 'date', 'posting date': 'date',
        'details': 'description', 'merchant': 'description', 'transaction description': 'description', 'memo': 'description', 'name': 'description',
        'transaction amount': 'amount', 'debit amount': 'debit', 'credit amount': 'credit', 'withdrawal': 'debit', 'deposit': 'credit'
    }
    df = df.rename(columns={c: mapping.get(c, c) for c in df.columns})
    return df


def standardize_tabular(df, source_name):
    df = clean_columns(df)
    if 'date' not in df.columns or 'description' not in df.columns:
        raise ValueError('Missing date/description columns.')
    if 'amount' in df.columns:
        df['net_amount'] = df['amount'].apply(parse_amount)
    else:
        debit = df['debit'].apply(parse_amount) if 'debit' in df.columns else 0
        credit = df['credit'].apply(parse_amount) if 'credit' in df.columns else 0
        df['net_amount'] = pd.Series(credit).fillna(0) - pd.Series(debit).fillna(0)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].notna() & df['net_amount'].notna()].copy()
    df['description'] = df['description'].astype(str).str.strip()
    df['source_file'] = source_name
    return df[['date', 'description', 'net_amount', 'source_file']]


def detect_pdf_type(text, filename):
    lower = (text[:8000] + ' ' + filename).lower()
    if 'chase' in lower:
        return 'chase'
    if 'citi' in lower or 'costco anywhere visa' in lower:
        return 'citi'
    return 'generic'


def extract_pdf_lines(file_bytes):
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text(x_tolerance=2, y_tolerance=2) or ''
            pages.append(txt)
    return pages


def parse_generic_pdf(file_bytes, source_name):
    rows = []
    for text in extract_pdf_lines(file_bytes):
        for line in text.split('\n'):
            s = ' '.join(line.split())
            if len(s) < 8:
                continue
            dm = DATE_PATTERN.match(s)
            amts = AMOUNT_PATTERN.findall(s)
            if not dm or not amts:
                continue
            date_txt = dm.group(1)
            amount_txt = amts[-1]
            amount = parse_amount(amount_txt)
            body = s.replace(date_txt, '', 1)
            idx = body.rfind(amount_txt)
            if idx != -1:
                body = body[:idx]
            desc = body.strip(' -')
            if desc and pd.notna(amount):
                rows.append({'date': date_txt, 'description': desc, 'net_amount': amount, 'source_file': source_name})
    return pd.DataFrame(rows)


def parse_chase_pdf(file_bytes, source_name):
    rows = []
    for text in extract_pdf_lines(file_bytes):
        for line in text.split('\n'):
            s = ' '.join(line.split())
            if len(s) < 8:
                continue
            if any(skip in s.lower() for skip in ['payment information', 'credit limit', 'account summary', 'interest charged', 'fees charged']):
                continue
            dates = DATE_PATTERN.findall(s)
            amts = AMOUNT_PATTERN.findall(s)
            if not dates or not amts:
                continue
            date_txt = dates[0]
            amount_txt = amts[-1]
            amount = -abs(parse_amount(amount_txt))
            desc = s
            desc = desc.replace(date_txt, '', 1)
            idx = desc.rfind(amount_txt)
            if idx != -1:
                desc = desc[:idx]
            desc = re.sub(r'\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b', '', desc).strip(' -')
            if len(desc) >= 3 and pd.notna(amount):
                if 'payment thank you' in desc.lower() or 'automatic payment' in desc.lower():
                    amount = abs(amount)
                rows.append({'date': date_txt, 'description': desc, 'net_amount': amount, 'source_file': source_name})
    return pd.DataFrame(rows)


def parse_citi_pdf(file_bytes, source_name):
    rows = []
    for text in extract_pdf_lines(file_bytes):
        for line in text.split('\n'):
            s = ' '.join(line.split())
            if len(s) < 8:
                continue
            if any(skip in s.lower() for skip in ['account summary', 'credit line', 'minimum payment', 'late payment warning', 'fees charged']):
                continue
            dates = DATE_PATTERN.findall(s)
            amts = AMOUNT_PATTERN.findall(s)
            if not dates or not amts:
                continue
            date_txt = dates[0]
            amount_txt = amts[-1]
            raw_amount = parse_amount(amount_txt)
            desc = s.replace(date_txt, '', 1)
            idx = desc.rfind(amount_txt)
            if idx != -1:
                desc = desc[:idx]
            desc = re.sub(r'\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b', '', desc).strip(' -')
            if len(desc) < 3 or pd.isna(raw_amount):
                continue
            amount = -abs(raw_amount)
            low = desc.lower()
            if any(k in low for k in ['payment', 'cashback', 'refund', 'credit voucher', 'merchant credit']):
                amount = abs(raw_amount)
            rows.append({'date': date_txt, 'description': desc, 'net_amount': amount, 'source_file': source_name})
    return pd.DataFrame(rows)


def parse_pdf_statement(uploaded_file):
    file_bytes = uploaded_file.read()
    pages = extract_pdf_lines(file_bytes)
    combined_text = '\n'.join(pages)
    pdf_type = detect_pdf_type(combined_text, uploaded_file.name)
    if pdf_type == 'chase':
        df = parse_chase_pdf(file_bytes, uploaded_file.name)
    elif pdf_type == 'citi':
        df = parse_citi_pdf(file_bytes, uploaded_file.name)
    else:
        df = parse_generic_pdf(file_bytes, uploaded_file.name)
    if df.empty:
        raise ValueError('No transactions could be extracted from this PDF.')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].notna() & df['net_amount'].notna()].copy()
    return df[['date', 'description', 'net_amount', 'source_file']]


def build_dataset(files):
    frames = []
    errors = []
    for f in files:
        try:
            name = f.name.lower()
            if name.endswith('.csv'):
                frames.append(standardize_tabular(pd.read_csv(f), f.name))
            elif name.endswith('.xlsx') or name.endswith('.xls'):
                frames.append(standardize_tabular(pd.read_excel(f), f.name))
            elif name.endswith('.pdf'):
                frames.append(parse_pdf_statement(f))
            else:
                errors.append(f'{f.name}: unsupported file type')
        except Exception as e:
            errors.append(f'{f.name}: {e}')
    if not frames:
        return pd.DataFrame(), errors
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    return df, errors


def analyze(df):
    df = df.copy()
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['category'] = [classify_category(d, a) for d, a in zip(df['description'], df['net_amount'])]
    df['income'] = np.where(df['net_amount'] > 0, df['net_amount'], 0.0)
    df['expense'] = np.where(df['net_amount'] < 0, -df['net_amount'], 0.0)

    monthly = df.groupby('month', as_index=False).agg(
        income=('income', 'sum'),
        expenses=('expense', 'sum'),
        transactions=('net_amount', 'count')
    )
    monthly['net'] = monthly['income'] - monthly['expenses']
    monthly['savings_rate'] = np.where(monthly['income'] > 0, (monthly['net'] / monthly['income']) * 100, np.nan)
    monthly = monthly.sort_values('month')

    cat = df[df['expense'] > 0].groupby('category', as_index=False)['expense'].sum().sort_values('expense', ascending=False)
    month_cat = df[df['expense'] > 0].groupby(['month', 'category'], as_index=False)['expense'].sum()
    return df, monthly, cat, month_cat


def generate_tips(monthly, cat):
    tips = []
    if monthly.empty:
        return tips
    avg_income = monthly['income'].mean()
    avg_expenses = monthly['expenses'].mean()
    avg_savings_rate = monthly['savings_rate'].dropna().mean() if monthly['savings_rate'].notna().any() else 0
    if avg_income > 0 and avg_savings_rate < 20:
        gap = max(0, 0.2 * avg_income - (avg_income - avg_expenses))
        tips.append(f'Try to free up about ${gap:,.0f} per month to reach a 20% savings rate.')
    total_expense = cat['expense'].sum() if not cat.empty else 0
    for _, row in cat.head(6).iterrows():
        c = row['category']
        amt = row['expense']
        share = (amt / total_expense * 100) if total_expense else 0
        if c in NON_ESSENTIAL and share >= 8:
            tips.append(f'{c} is {share:.1f}% of spending. A 15% reduction would save about ${amt * 0.15:,.0f}.')
        elif c in ESSENTIAL and share >= 20:
            tips.append(f'{c} is a major cost area. Review vendors, plans, or usage for savings opportunities.')
    if not tips:
        tips.append('Spending looks relatively balanced. Focus on automating savings and reviewing your top one or two categories monthly.')
    return tips[:6]


def to_excel(monthly, cat, txns):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        monthly.to_excel(writer, index=False, sheet_name='Monthly Summary')
        cat.to_excel(writer, index=False, sheet_name='Categories')
        txns.to_excel(writer, index=False, sheet_name='Transactions')
    return buf.getvalue()


st.markdown('''
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.metric-card {background: #f8fafc; padding: 1rem; border-radius: 14px; border: 1px solid #e5e7eb;}
.tip-card {background: #f8fafc; border-left: 4px solid #0f766e; padding: 0.9rem 1rem; border-radius: 10px; margin-bottom: 0.75rem;}
.small-note {color: #475569; font-size: 0.93rem;}
</style>
''', unsafe_allow_html=True)

st.title('💼 Finance Statement Analyzer')
st.write('Upload checking, savings, credit card, CSV, Excel, or PDF statements. The app combines them into a cleaner monthly income and expense view with category insights and savings suggestions.')

with st.sidebar:
    st.header('Upload files')
    uploaded = st.file_uploader('Statements', type=['csv', 'xlsx', 'xls', 'pdf'], accept_multiple_files=True)
    st.markdown('<div class="small-note">Privacy note: this app processes uploaded files during the session and does not intentionally save the uploaded source files.</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">PDF support is best for text-based statements. Chase and Citi layouts now have custom parsing logic.</div>', unsafe_allow_html=True)

if not uploaded:
    st.info('Upload your statements to begin.')
    st.stop()

df, errors = build_dataset(uploaded)

if errors:
    for err in errors:
        st.error(err)

if df.empty:
    st.stop()

df, monthly, cat, month_cat = analyze(df)
tips = generate_tips(monthly, cat)

c1, c2, c3, c4 = st.columns(4)
c1.metric('Months', len(monthly))
c2.metric('Income', f"${monthly['income'].sum():,.0f}")
c3.metric('Expenses', f"${monthly['expenses'].sum():,.0f}")
c4.metric('Net savings', f"${monthly['net'].sum():,.0f}")

st.subheader('Monthly cash flow')
fig = go.Figure()
fig.add_bar(x=monthly['month'], y=monthly['income'], name='Income', marker_color='#16a34a')
fig.add_bar(x=monthly['month'], y=monthly['expenses'], name='Expenses', marker_color='#dc2626')
fig.add_scatter(x=monthly['month'], y=monthly['net'], mode='lines+markers', name='Net', line=dict(color='#0f172a', width=3))
fig.update_layout(height=420, barmode='group', margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation='h'))
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([1.1, 0.9])
with left:
    st.subheader('Monthly summary')
    display_monthly = monthly.copy()
    for col in ['income', 'expenses', 'net', 'savings_rate']:
        if col in display_monthly.columns:
            display_monthly[col] = display_monthly[col].round(2)
    st.dataframe(display_monthly, use_container_width=True, hide_index=True)
with right:
    st.subheader('Savings suggestions')
    for tip in tips:
        st.markdown(f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader('Top expense categories')
    if not cat.empty:
        bar = px.bar(cat.head(10).sort_values('expense'), x='expense', y='category', orientation='h', text='expense')
        bar.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        bar.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), xaxis_title='Amount', yaxis_title='')
        st.plotly_chart(bar, use_container_width=True)
with col_b:
    st.subheader('Expense mix')
    if not cat.empty:
        pie = px.pie(cat.head(8), names='category', values='expense', hole=0.5)
        pie.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(pie, use_container_width=True)

st.subheader('Transactions')
show = df.sort_values('date', ascending=False).copy()
show['date'] = show['date'].dt.strftime('%Y-%m-%d')
show['net_amount'] = show['net_amount'].round(2)
st.dataframe(show[['date', 'description', 'category', 'net_amount', 'source_file']], use_container_width=True, hide_index=True, height=360)

st.download_button(
    'Download analysis workbook',
    data=to_excel(monthly, cat, show),
    file_name='finance_analysis.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
