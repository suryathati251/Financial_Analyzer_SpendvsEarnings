import io
import re
import numpy as np
import pandas as pd
import pdfplumber
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Finance Statement Analyzer', page_icon='💼', layout='wide')

CATEGORY_RULES = {
    'Income': [r'payroll', r'salary', r'direct deposit', r'direct dep', r'ach credit', r'interest', r'dividend', r'refund', r'cashback', r'payment thank you', r'merchant credit'],
    'Housing': [r'rent', r'mortgage', r'property tax', r'hoa'],
    'Utilities': [r'utility', r'electric', r'firstenergy', r'water', r'internet', r'comcast', r'xfinity', r'phone', r'verizon', r't-mobile', r'at&t', r'att\* bill payment'],
    'Groceries': [r'walmart', r'costco', r'trader joe', r'whole foods', r'aldi', r'stop & shop', r'kroger', r'wegmans', r'acme', r'patel brothers', r'grocer', r'shoprite', r'apna bazar'],
    'Dining': [r'restaurant', r'grubhub', r'ubereats', r'uber eats', r'doordash', r'dunkin', r'starbucks', r'chipotle', r'mcdonald', r'panera', r'pizza', r'paris baguette', r'burger king', r'domino', r'cafe', r'taco'],
    'Transport': [r'uber', r'lyft', r'shell', r'exxon', r'chevron', r'fuel', r'gas station', r'parking', r'toll', r'mta', r'nj transit', r'ezpass'],
    'Shopping': [r'amazon', r'target', r'best buy', r'macy', r'etsy', r'ebay', r'old navy', r'nike', r'kohls', r'tjmaxx', r'temu', r'aliexpress', r'home goods', r'primark'],
    'Health': [r'cvs', r'walgreens', r'pharmacy', r'medical', r'dental', r'hospital', r'urgent care', r'vision', r'eyecare'],
    'Insurance': [r'insurance', r'geico', r'progressive', r'allstate', r'state farm', r'condo insurance'],
    'Subscriptions': [r'netflix', r'spotify', r'hulu', r'apple.com/bill', r'youtube', r'prime', r'adobe', r'openai', r'chatgpt', r'icloud', r'patreon', r'linkedin'],
    'Travel': [r'airbnb', r'hotel', r'booking.com', r'delta', r'united', r'southwest', r'american air', r'alaska air', r'chase travel', r'cruise', r'dcl', r'frontier', r'parking auth'],
    'Transfers': [r'zelle', r'venmo', r'cash app', r'transfer', r'payment from', r'payment to', r'ach debit', r'ach withdrawal'],
    'Fees': [r'fee', r'finance charge', r'late fee', r'interest charge', r'overdraft', r'annual membership fee'],
    'Cash': [r'atm', r'cash withdrawal'],
    'Education': [r'outschool', r'board of educa', r'chessbrainz', r'jobtestprep', r'kids united'],
    'Entertainment': [r'amc', r'sixflags', r'urban air', r'boat', r'heritage amusements', r'lava hot springs', r'entertainment'],
}

NON_ESSENTIAL = {'Dining', 'Shopping', 'Subscriptions', 'Travel', 'Cash', 'Entertainment'}
ESSENTIAL = {'Housing', 'Utilities', 'Groceries', 'Health', 'Insurance', 'Transport', 'Education'}

CHASE_LINE = re.compile(r'([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})(.+?)(\$-?[\d,]+\.\d{2})$')
CITI_LINE = re.compile(r'([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})(.+?)(\$-?[\d,]+\.\d{2})$')


def parse_amount(value):
    s = str(value).strip().replace('$', '').replace(',', '')
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]
    try:
        return float(s)
    except Exception:
        return np.nan


def classify_category(desc, amount):
    text = str(desc).lower()
    for cat, patterns in CATEGORY_RULES.items():
        if any(re.search(p, text) for p in patterns):
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
    return df.rename(columns={c: mapping.get(c, c) for c in df.columns})


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


def read_pdf_pages(file_bytes):
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text(x_tolerance=1, y_tolerance=1) or ''
            pages.append(txt)
    return pages


def detect_pdf_layout(text, filename):
    t = (text + ' ' + filename).lower()
    if 'annual account summary' in t and 'citi online' in t:
        return 'citi_annual'
    if 'spending report' in t and 'jpmorgan chase bank' in t:
        return 'chase_annual'
    return 'generic'


def extract_chase_annual(file_bytes, source_name):
    rows = []
    current_category = None
    pages = read_pdf_pages(file_bytes)
    category_headers = {
        'AUTOMOTIVE','BILLS_AND_UTILITIES','FOOD_AND_DRINK','GAS','GROCERIES','HEALTH_AND_WELLNESS','HOME',
        'PERSONAL','SHOPPING','TRAVEL','ENTERTAINMENT','EDUCATION','GIFTS_AND_DONATIONS','PROFESSIONAL_SERVICES','FEES_AND_ADJUSTMENTS'
    }
    for text in pages:
        for raw in text.split('\n'):
            line = ' '.join(raw.split())
            if not line:
                continue
            upper = line.strip().upper()
            if upper in category_headers:
                current_category = upper
                continue
            if any(skip in line for skip in ['Transaction DatePosted DateDescriptionAmount', 'Spending By Category', 'CategoryTotal Amount', 'Total$', 'JPMorgan Chase Bank', 'Equal Opportunity Lender', 'Jan 01, 2025 to Dec 31, 2025 Spending Report']):
                continue
            if line.startswith('Total $') or line.startswith('Total$') or line.endswith('continues...'):
                continue
            m = CHASE_LINE.match(line)
            if not m:
                continue
            trans_date, posted_date, desc, amount_txt = m.groups()
            desc = desc.strip()
            amount = parse_amount(amount_txt)
            if pd.isna(amount):
                continue
            net = -abs(amount)
            if amount < 0:
                net = abs(amount)
            rows.append({
                'date': trans_date,
                'description': desc,
                'net_amount': net,
                'source_file': source_name,
                'statement_category': current_category
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('No Chase annual transactions extracted.')
    return df


def extract_citi_annual(file_bytes, source_name):
    rows = []
    current_category = None
    pages = read_pdf_pages(file_bytes)
    skip_fragments = ['This report was generated by Citi Online', 'Annual Account Summary', 'Totals by Category', 'Total by Month', 'page ', 'If you earn rewards by category', 'Notes', 'Subtotal$', 'Subtotal $', 'Total$', 'Total $']
    known_headers = {'Entertainment','Merchandise','Organizations','Restaurants','Services','Vehicle Services'}
    for text in pages:
        for raw in text.split('\n'):
            line = ' '.join(raw.split())
            if not line:
                continue
            if any(frag in line for frag in skip_fragments):
                continue
            if line in known_headers:
                current_category = line
                continue
            if line in ['DateDescriptionAmount', 'Date Description Amount', 'CategoryAmount', 'MonthAmount']:
                continue
            m = CITI_LINE.match(line)
            if not m:
                continue
            trans_date, desc, amount_txt = m.groups()
            desc = desc.strip()
            amount = parse_amount(amount_txt)
            if pd.isna(amount):
                continue
            net = -abs(amount)
            if amount < 0:
                net = abs(amount)
            rows.append({
                'date': trans_date,
                'description': desc,
                'net_amount': net,
                'source_file': source_name,
                'statement_category': current_category
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('No Citi annual transactions extracted.')
    return df


def extract_generic_pdf(file_bytes, source_name):
    rows = []
    pages = read_pdf_pages(file_bytes)
    generic = re.compile(r'([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4}).+?(\$-?[\d,]+\.\d{2})$')
    for text in pages:
        for raw in text.split('\n'):
            line = ' '.join(raw.split())
            m = generic.match(line)
            if not m:
                continue
            date_txt = m.group(1)
            amount_txt = m.group(2)
            desc = line.replace(date_txt, '', 1)
            idx = desc.rfind(amount_txt)
            if idx != -1:
                desc = desc[:idx]
            amount = parse_amount(amount_txt)
            if pd.isna(amount):
                continue
            net = amount
            rows.append({'date': date_txt, 'description': desc.strip(), 'net_amount': net, 'source_file': source_name, 'statement_category': None})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('No generic PDF transactions extracted.')
    return df


def parse_pdf_statement(uploaded_file):
    file_bytes = uploaded_file.read()
    pages = read_pdf_pages(file_bytes)
    text = '\n'.join(pages)
    layout = detect_pdf_layout(text, uploaded_file.name)
    if layout == 'chase_annual':
        df = extract_chase_annual(file_bytes, uploaded_file.name)
    elif layout == 'citi_annual':
        df = extract_citi_annual(file_bytes, uploaded_file.name)
    else:
        df = extract_generic_pdf(file_bytes, uploaded_file.name)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df[df['date'].notna() & df['net_amount'].notna()].copy()
    return df[['date', 'description', 'net_amount', 'source_file', 'statement_category']]


def build_dataset(files):
    frames = []
    errors = []
    for f in files:
        try:
            name = f.name.lower()
            if name.endswith('.csv'):
                t = standardize_tabular(pd.read_csv(f), f.name)
                t['statement_category'] = None
                frames.append(t)
            elif name.endswith('.xlsx') or name.endswith('.xls'):
                t = standardize_tabular(pd.read_excel(f), f.name)
                t['statement_category'] = None
                frames.append(t)
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


def normalize_final_categories(df):
    df = df.copy()
    if 'statement_category' in df.columns:
        mapping = {
            'BILLS_AND_UTILITIES': 'Utilities', 'FOOD_AND_DRINK': 'Dining', 'GAS': 'Transport', 'GROCERIES': 'Groceries',
            'HEALTH_AND_WELLNESS': 'Health', 'SHOPPING': 'Shopping', 'TRAVEL': 'Travel', 'ENTERTAINMENT': 'Entertainment',
            'EDUCATION': 'Education', 'FEES_AND_ADJUSTMENTS': 'Fees', 'GIFTS_AND_DONATIONS': 'Other', 'AUTOMOTIVE': 'Transport',
            'PROFESSIONAL_SERVICES': 'Other', 'HOME': 'Housing', 'PERSONAL': 'Other',
            'Merchandise': 'Shopping', 'Restaurants': 'Dining', 'Services': 'Other', 'Vehicle Services': 'Transport',
            'Entertainment': 'Entertainment', 'Organizations': 'Education'
        }
        df['statement_category_mapped'] = df['statement_category'].map(mapping)
    else:
        df['statement_category_mapped'] = None
    predicted = [classify_category(d, a) for d, a in zip(df['description'], df['net_amount'])]
    df['category'] = [m if pd.notna(m) and m else p for m, p in zip(df['statement_category_mapped'], predicted)]
    return df


def analyze(df):
    df = normalize_final_categories(df)
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['income'] = np.where(df['net_amount'] > 0, df['net_amount'], 0.0)
    df['expense'] = np.where(df['net_amount'] < 0, -df['net_amount'], 0.0)
    monthly = df.groupby('month', as_index=False).agg(income=('income', 'sum'), expenses=('expense', 'sum'), transactions=('net_amount', 'count'))
    monthly['net'] = monthly['income'] - monthly['expenses']
    monthly['savings_rate'] = np.where(monthly['income'] > 0, (monthly['net'] / monthly['income']) * 100, np.nan)
    monthly = monthly.sort_values('month')
    cat = df[df['expense'] > 0].groupby('category', as_index=False)['expense'].sum().sort_values('expense', ascending=False)
    return df, monthly, cat


def generate_tips(monthly, cat):
    tips = []
    if monthly.empty:
        return ['No monthly summary available yet.']
    avg_income = monthly['income'].mean()
    avg_exp = monthly['expenses'].mean()
    avg_sr = monthly['savings_rate'].dropna().mean() if monthly['savings_rate'].notna().any() else 0
    if avg_income > 0 and avg_sr < 20:
        gap = max(0, 0.2 * avg_income - (avg_income - avg_exp))
        tips.append(f'To reach a 20% savings rate, aim to free up about ${gap:,.0f} per month.')
    total = cat['expense'].sum() if not cat.empty else 0
    for _, row in cat.head(6).iterrows():
        share = (row['expense'] / total * 100) if total else 0
        if row['category'] in NON_ESSENTIAL and share >= 8:
            tips.append(f"{row['category']} is {share:.1f}% of your spending. Cutting 10% could save about ${row['expense'] * 0.10:,.0f}.")
        elif row['category'] in ESSENTIAL and share >= 20:
            tips.append(f"{row['category']} is a large essential category. Review providers, plans, and usage for possible savings.")
    return tips[:6] if tips else ['Spending is fairly balanced. Review your top one or two categories monthly and automate savings if possible.']


def to_excel(monthly, cat, txns):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        monthly.to_excel(writer, index=False, sheet_name='Monthly Summary')
        cat.to_excel(writer, index=False, sheet_name='Categories')
        txns.to_excel(writer, index=False, sheet_name='Transactions')
    return buf.getvalue()


st.markdown('''
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.tip-card {background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.75rem;}
.small-note {color:#475569;font-size:0.92rem;}
</style>
''', unsafe_allow_html=True)

st.title('💼 Finance Statement Analyzer')
st.write('Upload checking, savings, credit card, CSV, Excel, or annual summary PDF statements. The app extracts transactions, builds monthly summaries, categorizes spending, and suggests savings opportunities.')

with st.sidebar:
    st.header('Upload files')
    uploaded = st.file_uploader('Statements', type=['csv', 'xlsx', 'xls', 'pdf'], accept_multiple_files=True)
    st.markdown('<div class="small-note">Updated parser supports Chase annual spending reports and Citi annual account summaries like the samples you shared.</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Uploaded files are processed during the session and are not intentionally saved by the app.</div>', unsafe_allow_html=True)

if not uploaded:
    st.info('Upload one or more statements to begin.')
    st.stop()

df, errors = build_dataset(uploaded)
if errors:
    for e in errors:
        st.error(e)
if df.empty:
    st.stop()

df, monthly, cat = analyze(df)
tips = generate_tips(monthly, cat)

m1, m2, m3, m4 = st.columns(4)
m1.metric('Months', len(monthly))
m2.metric('Income', f"${monthly['income'].sum():,.0f}")
m3.metric('Expenses', f"${monthly['expenses'].sum():,.0f}")
m4.metric('Net', f"${monthly['net'].sum():,.0f}")

st.subheader('Monthly cash flow')
fig = go.Figure()
fig.add_bar(x=monthly['month'], y=monthly['income'], name='Income', marker_color='#16a34a')
fig.add_bar(x=monthly['month'], y=monthly['expenses'], name='Expenses', marker_color='#dc2626')
fig.add_scatter(x=monthly['month'], y=monthly['net'], name='Net', mode='lines+markers', line=dict(color='#0f172a', width=3))
fig.update_layout(height=420, barmode='group', margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([1.15, 0.85])
with left:
    st.subheader('Monthly summary')
    view = monthly.copy()
    for c in ['income', 'expenses', 'net', 'savings_rate']:
        view[c] = view[c].round(2)
    st.dataframe(view, use_container_width=True, hide_index=True)
with right:
    st.subheader('Savings suggestions')
    for tip in tips:
        st.markdown(f'<div class="tip-card">{tip}</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader('Top categories')
    if not cat.empty:
        b = px.bar(cat.head(10).sort_values('expense'), x='expense', y='category', orientation='h', text='expense')
        b.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        b.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), xaxis_title='Expense amount', yaxis_title='')
        st.plotly_chart(b, use_container_width=True)
with c2:
    st.subheader('Expense mix')
    if not cat.empty:
        p = px.pie(cat.head(8), names='category', values='expense', hole=0.5)
        p.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(p, use_container_width=True)

st.subheader('Extracted transactions')
show = df.sort_values('date', ascending=False).copy()
show['date'] = show['date'].dt.strftime('%Y-%m-%d')
show['net_amount'] = show['net_amount'].round(2)
st.dataframe(show[['date', 'description', 'statement_category', 'category', 'net_amount', 'source_file']], use_container_width=True, hide_index=True, height=380)

st.download_button('Download analysis workbook', data=to_excel(monthly, cat, show), file_name='finance_analysis.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
