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
    'Income': [r'payroll', r'direct deposit', r'payment from', r'zelle payment from', r'interest', r'dividend', r'refund', r'credit'],
    'Housing': [r'rent', r'mortgage', r'hoa', r'clickpay', r'firstservice'],
    'Utilities': [r'utility', r'firstenergy', r'public services electric', r'pse&g', r'att bill payment', r'cricket wireless', r'phone', r'internet'],
    'Groceries': [r'costco', r'stop shop', r'shoprite', r'trader joe', r'patel brothers', r'acme', r'whole foods', r'wegmans', r'apna bazar', r'walmart'],
    'Dining': [r'restaurant', r'dhaba', r'panera', r'starbucks', r'bagel', r'wonder', r'burger', r'taco', r'pizza', r'paris baguette', r'sushi'],
    'Transport': [r'ezpass', r'gas', r'quick chek', r'wawa', r'bp', r'exxon', r'uber', r'lyft', r'parking', r'tesla supercharger', r'crown car wash'],
    'Shopping': [r'amazon', r'target', r'macys', r'lacoste', r'costco.com', r'temu', r'underarmour', r'apple\.combill', r'home depot'],
    'Health': [r'walgreens', r'pharmacy', r'orthopaedic', r'peds', r'diagnostics', r'optical', r'eyecare', r'ouraring'],
    'Insurance': [r'geico', r'insurance'],
    'Subscriptions': [r'patreon', r'uber one', r'youtube', r'disney plus', r'hp instant ink', r'linkedin', r'tesla subscription', r'apple\.combill'],
    'Travel': [r'airbnb', r'united', r'american air', r'alaska air', r'chase travel', r'cruise', r'dcl', r'frontier'],
    'Transfers': [r'payment thank you', r'online payment', r'payment to chase card', r'online realtime transfer', r'online transfer to sav', r'zelle payment to', r'robinhood debits', r'transfer to bofa'],
    'Fees': [r'fee', r'interest charge', r'annual membership fee', r'late fee'],
    'Education': [r'outschool', r'quizlet', r'board of educa', r'taekwondo', r'swim school'],
    'Entertainment': [r'sixflags', r'amc', r'urban air'],
}

MONTHS = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
CHASE_ANNUAL_RE = re.compile(rf'({MONTHS}\s+\d{{1,2}},\s+\d{{4}})({MONTHS}\s+\d{{1,2}},\s+\d{{4}})(.+?)(-?[\d,]+\.\d{{2}})$')
CITI_ANNUAL_RE = re.compile(rf'({MONTHS}\s+\d{{1,2}},\s+\d{{4}})(.+?)(-?[\d,]+\.\d{{2}})$')
CHASE_CC_LINE_RE = re.compile(r'^(\d{4})\s+(.+?)\s+(-?[\d,]+\.\d{2})$')
CHASE_CHECKING_RE = re.compile(r'^(\d{4})\s+(.+?)\s+(-?[\d,]+\.\d{2})\s+(-?[\d,]+\.\d{2})$')
CITI_MONTHLY_RE = re.compile(r'^(\d{4})\s+(\d{4})\s+(.+?)\s+(-?[\d,]+\.\d{2})$')
AMEX_LINE_RE = re.compile(r'^(\d{4,8})\s+(.+?)\s+(?:Pay Over Time andor Cash Advance|MERCHANDISE|DISCOUNT STORE|RESTAURANT-BAR & CAFE|RESTAURANT|OTHER|TRANSPORTATION|SERVICES)?\s*(-?[\d,]+\.\d{2})$')


def parse_amount(v):
    s = str(v).strip().replace('$', '').replace(',', '')
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]
    try:
        return float(s)
    except Exception:
        return np.nan


def compact(s):
    return ' '.join(str(s).split())


def classify_category(desc, amount):
    text = str(desc).lower()
    for cat, pats in CATEGORY_RULES.items():
        if any(re.search(p, text) for p in pats):
            return cat
    return 'Income' if amount > 0 else 'Other'


def read_pdf_pages(file_bytes):
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text(x_tolerance=1, y_tolerance=1) or '')
    return pages


def detect_layout(text, filename):
    t = (text + ' ' + filename).lower()
    if 'annual account summary' in t and 'citi' in t:
        return 'citi_annual'
    if 'spending report' in t and 'jpmorgan chase' in t:
        return 'chase_annual'
    if 'platinum card' in t and 'american express' in t:
        return 'amex_monthly'
    if 'costco anywhere visa card by citi' in t and 'cardholder summary' in t:
        return 'citi_monthly'
    if 'checking summary' in t and 'transaction detail' in t and 'jpmorgan chase bank' in t:
        return 'chase_checking'
    if 'account activity' in t and ('openingclosing date' in t or 'opening/closing date' in t):
        return 'chase_credit_monthly'
    if 'bank of america' in t or 'bankofamerica' in t or 'bank of america advantage' in t:
        return 'generic_bank'
    return 'generic_bank'


def finalize_df(rows, source_name):
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('No transactions extracted.')
    df['date'] = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
    df = df[df['date'].notna() & df['net_amount'].notna()].copy()
    df['source_file'] = source_name
    return df[['date', 'description', 'net_amount', 'source_file', 'statement_category']]


def extract_chase_annual(file_bytes, source_name):
    headers = {'AUTOMOTIVE','BILLSANDUTILITIES','FOODANDDRINK','GAS','GROCERIES','HEALTHANDWELLNESS','HOME','PERSONAL','SHOPPING','TRAVEL','ENTERTAINMENT','EDUCATION','GIFTSANDDONATIONS','PROFESSIONALSERVICES','FEESANDADJUSTMENTS'}
    rows, current = [], None
    for text in read_pdf_pages(file_bytes):
        for raw in text.split('\n'):
            line = compact(raw)
            if not line:
                continue
            upper = re.sub(r'[^A-Z]', '', line.upper())
            if upper in headers:
                current = upper
                continue
            if any(x in line for x in ['Transaction DatePosted DateDescriptionAmount', 'Spending By Category', 'CategoryTotal Amount', 'JPMorgan Chase Bank']):
                continue
            if line.startswith('Total ') or 'Spending Report' in line and 'Jan 01' in line:
                continue
            m = CHASE_ANNUAL_RE.search(line)
            if not m:
                continue
            d1, d2, desc, amt = m.groups()
            a = parse_amount(amt)
            net = abs(a) if a < 0 else -abs(a)
            rows.append({'date': d1, 'description': compact(desc), 'net_amount': net, 'statement_category': current})
    if not rows:
        raise ValueError('No Chase annual transactions extracted.')
    return finalize_df(rows, source_name)


def extract_citi_annual(file_bytes, source_name):
    rows, current = [], None
    known = {'Entertainment','Merchandise','Organizations','Restaurants','Services','Vehicle Services'}
    for text in read_pdf_pages(file_bytes):
        for raw in text.split('\n'):
            line = compact(raw)
            if line in known:
                current = line
                continue
            if any(x in line for x in ['Annual Account Summary', 'Totals by Category', 'Total by Month', 'DateDescriptionAmount', 'Subtotal', 'This report was generated by Citi Online']):
                continue
            m = CITI_ANNUAL_RE.search(line)
            if not m:
                continue
            d, desc, amt = m.groups()
            a = parse_amount(amt)
            net = abs(a) if a < 0 else -abs(a)
            rows.append({'date': d, 'description': compact(desc), 'net_amount': net, 'statement_category': current})
    if not rows:
        raise ValueError('No Citi annual transactions extracted.')
    return finalize_df(rows, source_name)


def extract_chase_credit_monthly(file_bytes, source_name):
    rows, section = [], None
    for text in read_pdf_pages(file_bytes):
        for raw in text.split('\n'):
            line = compact(raw)
            if not line:
                continue
            if 'PAYMENTS AND OTHER CREDITS' in line:
                section = 'credit'
                continue
            if line.startswith('PURCHASE'):
                section = 'purchase'
                continue
            if line.startswith('FEES CHARGED'):
                section = 'fee'
                continue
            if any(skip in line for skip in ['ACCOUNT ACTIVITY', 'Date of Transaction Merchant Name or Transaction Description Amount', 'TOTAL FEES FOR THIS PERIOD', '2026 Totals', 'Year-to-Date', 'INTEREST CHARGES', 'Order Number', 'REWARDS', 'RETURNS AND OTHER CREDITS', 'PURCHASES AND REDEMPTIONS']):
                continue
            m = CHASE_CC_LINE_RE.match(line)
            if not m:
                continue
            date4, desc, amt = m.groups()
            a = parse_amount(amt)
            year = 2025 if date4.startswith('12') else 2026
            date_txt = f"{date4[:2]}/{date4[2:]}/{year}"
            net = a
            if section == 'purchase' or section == 'fee':
                net = -abs(a)
            elif section == 'credit':
                net = abs(a) if a < 0 else a
            rows.append({'date': date_txt, 'description': compact(desc), 'net_amount': net, 'statement_category': None})
    if not rows:
        raise ValueError('No Chase monthly credit transactions extracted.')
    return finalize_df(rows, source_name)


def extract_chase_checking(file_bytes, source_name):
    rows = []
    for text in read_pdf_pages(file_bytes):
        for raw in text.split('\n'):
            line = compact(raw)
            if not line or 'Beginning Balance' in line or 'Ending Balance' in line:
                continue
            if any(skip in line for skip in ['TRANSACTION DETAIL', 'DATE DESCRIPTION AMOUNT BALANCE', 'CHECKING SUMMARY', 'Electronic Withdrawals', 'Deposits and Additions']):
                continue
            m = CHASE_CHECKING_RE.match(line)
            if not m:
                continue
            date4, desc, amt, bal = m.groups()
            year = 2025 if date4.startswith('12') else 2026
            date_txt = f"{date4[:2]}/{date4[2:]}/{year}"
            rows.append({'date': date_txt, 'description': compact(desc), 'net_amount': parse_amount(amt), 'statement_category': None})
    if not rows:
        raise ValueError('No Chase checking transactions extracted.')
    return finalize_df(rows, source_name)


def extract_citi_monthly(file_bytes, source_name):
    rows = []
    in_account = False
    for text in read_pdf_pages(file_bytes):
        for raw in text.split('\n'):
            line = compact(raw)
            if 'ACCOUNT SUMMARY Sale Date Post Date Description Amount' in line:
                in_account = True
                continue
            if not in_account:
                continue
            if any(skip in line for skip in ['Fees Charged', 'Interest Charged', '2026 totals', 'Interest charge calculation', 'Account messages', 'Costco Cash Back Rewards Summary']):
                continue
            if line.startswith('Payments, Credits and Adjustments') or line.startswith('SURESH THATI Standard Purchases'):
                continue
            m = CITI_MONTHLY_RE.match(line)
            if not m:
                continue
            sale, post, desc, amt = m.groups()
            year = 2025 if sale.startswith('12') else 2026
            date_txt = f"{sale[:2]}/{sale[2:]}/{year}"
            a = parse_amount(amt)
            rows.append({'date': date_txt, 'description': compact(desc), 'net_amount': a, 'statement_category': None})
    if not rows:
        raise ValueError('No Citi monthly transactions extracted.')
    return finalize_df(rows, source_name)


def extract_amex_monthly(file_bytes, source_name):
    rows, mode = [], None
    for text in read_pdf_pages(file_bytes):
        for raw in text.split('\n'):
            line = compact(raw)
            if 'Payments Details Indicates posting date' in line or 'Credits Details Indicates posting date' in line:
                mode = 'credit'
                continue
            if 'New Charges Details' in line:
                mode = 'charge'
                continue
            if any(skip in line for skip in ['Date Description Type Amount', 'Payments and Credits Summary', 'New Charges Summary', 'Fees Total Fees for this Period', 'Interest Charged Total Interest Charged for this Period']):
                continue
            m = AMEX_LINE_RE.match(line)
            if not m:
                continue
            date_raw, desc, amt = m.groups()
            if len(date_raw) == 8:
                mm, dd, yyyy = date_raw[:2], date_raw[2:4], date_raw[4:]
                date_txt = f'{mm}/{dd}/{yyyy}'
            else:
                mm, dd = date_raw[:2], date_raw[2:4]
                year = 2026 if mm in ['01', '02', '03', '04'] else 2025
                date_txt = f'{mm}/{dd}/{year}'
            a = parse_amount(amt)
            net = a
            if mode == 'charge':
                net = -abs(a)
            elif mode == 'credit':
                net = abs(a) if a < 0 else a
            rows.append({'date': date_txt, 'description': compact(desc), 'net_amount': net, 'statement_category': None})
    if not rows:
        raise ValueError('No Amex monthly transactions extracted.')
    return finalize_df(rows, source_name)


def extract_generic_bank(file_bytes, source_name):
    text = '\n'.join(read_pdf_pages(file_bytes))
    rows = []
    for raw in text.split('\n'):
        line = compact(raw)
        m = re.search(r'(\d{2}/\d{2}/\d{2,4})\s+(.+?)\s+(-?[\d,]+\.\d{2})(?:\s+(-?[\d,]+\.\d{2}))?$', line)
        if m:
            date_txt, desc, amt, _ = m.groups()
            rows.append({'date': date_txt, 'description': desc, 'net_amount': parse_amount(amt), 'statement_category': None})
    if not rows:
        raise ValueError('No generic PDF transactions extracted.')
    return finalize_df(rows, source_name)


def parse_pdf_statement(uploaded_file):
    file_bytes = uploaded_file.read()
    pages = read_pdf_pages(file_bytes)
    text = '\n'.join(pages)
    layout = detect_layout(text, uploaded_file.name)
    if layout == 'chase_annual':
        return extract_chase_annual(file_bytes, uploaded_file.name)
    if layout == 'citi_annual':
        return extract_citi_annual(file_bytes, uploaded_file.name)
    if layout == 'chase_credit_monthly':
        return extract_chase_credit_monthly(file_bytes, uploaded_file.name)
    if layout == 'chase_checking':
        return extract_chase_checking(file_bytes, uploaded_file.name)
    if layout == 'citi_monthly':
        return extract_citi_monthly(file_bytes, uploaded_file.name)
    if layout == 'amex_monthly':
        return extract_amex_monthly(file_bytes, uploaded_file.name)
    return extract_generic_bank(file_bytes, uploaded_file.name)


def build_dataset(files):
    frames, errors = [], []
    for f in files:
        try:
            name = f.name.lower()
            if name.endswith('.pdf'):
                frames.append(parse_pdf_statement(f))
            elif name.endswith('.csv'):
                df = pd.read_csv(f)
                frames.append(df)
            elif name.endswith('.xlsx') or name.endswith('.xls'):
                df = pd.read_excel(f)
                frames.append(df)
            else:
                errors.append(f'{f.name}: unsupported file type')
        except Exception as e:
            errors.append(f'{f.name}: {e}')
    if not frames:
        return pd.DataFrame(), errors
    df = pd.concat(frames, ignore_index=True)
    return df, errors


def normalize_categories(df):
    df = df.copy()
    mapping = {
        'BILLSANDUTILITIES':'Utilities','FOODANDDRINK':'Dining','GAS':'Transport','GROCERIES':'Groceries','HEALTHANDWELLNESS':'Health','SHOPPING':'Shopping','TRAVEL':'Travel','ENTERTAINMENT':'Entertainment','EDUCATION':'Education','FEESANDADJUSTMENTS':'Fees','AUTOMOTIVE':'Transport','HOME':'Housing','PERSONAL':'Other','PROFESSIONALSERVICES':'Other','GIFTSANDDONATIONS':'Other',
        'Merchandise':'Shopping','Restaurants':'Dining','Services':'Other','Vehicle Services':'Transport','Organizations':'Education','Entertainment':'Entertainment'
    }
    df['category'] = [mapping.get(sc, classify_category(d, a)) for sc, d, a in zip(df.get('statement_category', [None]*len(df)), df['description'], df['net_amount'])]
    return df


def analyze(df):
    df = normalize_categories(df)
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['income'] = np.where(df['net_amount'] > 0, df['net_amount'], 0.0)
    df['expense'] = np.where(df['net_amount'] < 0, -df['net_amount'], 0.0)
    monthly = df.groupby('month', as_index=False).agg(income=('income','sum'), expenses=('expense','sum'), transactions=('net_amount','count'))
    monthly['net'] = monthly['income'] - monthly['expenses']
    monthly['savings_rate'] = np.where(monthly['income'] > 0, (monthly['net']/monthly['income'])*100, np.nan)
    cat = df[df['expense'] > 0].groupby('category', as_index=False)['expense'].sum().sort_values('expense', ascending=False)
    return df.sort_values('date'), monthly.sort_values('month'), cat


def to_excel(monthly, cat, txns):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        monthly.to_excel(writer, index=False, sheet_name='Monthly Summary')
        cat.to_excel(writer, index=False, sheet_name='Categories')
        txns.to_excel(writer, index=False, sheet_name='Transactions')
    return buf.getvalue()

st.title('💼 Finance Statement Analyzer')
st.write('This app now supports the statement families you uploaded: Chase credit monthly and annual reports, Chase checking, Citi Costco monthly and annual summaries, and Amex monthly statements.')
files = st.file_uploader('Upload statements', type=['pdf','csv','xlsx','xls'], accept_multiple_files=True)
if not files:
    st.info('Upload your statements to analyze earnings and expenses.')
    st.stop()

df, errors = build_dataset(files)
if errors:
    for e in errors:
        st.error(e)
if df.empty:
    st.stop()

df, monthly, cat = analyze(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric('Files parsed', len(set(df['source_file'])))
c2.metric('Transactions', len(df))
c3.metric('Income', f"${monthly['income'].sum():,.0f}")
c4.metric('Expenses', f"${monthly['expenses'].sum():,.0f}")

fig = go.Figure()
fig.add_bar(x=monthly['month'], y=monthly['income'], name='Income', marker_color='#16a34a')
fig.add_bar(x=monthly['month'], y=monthly['expenses'], name='Expenses', marker_color='#dc2626')
fig.add_scatter(x=monthly['month'], y=monthly['net'], name='Net', mode='lines+markers', line=dict(color='#0f172a', width=3))
fig.update_layout(height=420, barmode='group')
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader('Monthly summary')
    st.dataframe(monthly.round(2), use_container_width=True, hide_index=True)
with right:
    st.subheader('Top categories')
    if not cat.empty:
        bar = px.bar(cat.head(12).sort_values('expense'), x='expense', y='category', orientation='h', text='expense')
        bar.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        bar.update_layout(height=420)
        st.plotly_chart(bar, use_container_width=True)

st.subheader('Transactions')
show = df.copy()
show['date'] = show['date'].dt.strftime('%Y-%m-%d')
show['net_amount'] = show['net_amount'].round(2)
st.dataframe(show[['date','description','statement_category','category','net_amount','source_file']], use_container_width=True, hide_index=True, height=420)

st.download_button('Download workbook', to_excel(monthly, cat, show), 'finance_analysis.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
st.caption('Files are processed in-session by the app; this Streamlit app is designed not to intentionally persist uploaded statement contents.')
