import io
import re
import unicodedata
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Finance Statement Analyzer', page_icon='💼', layout='wide')

MONTHS = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12}

CATEGORY_RULES = {
    'Income': [r'payroll', r'direct deposit', r'refund', r'credit', r'interest', r'dividend'],
    'Housing': [r'rent', r'mortgage', r'hoa', r'clickpay', r'firstservice'],
    'Utilities': [r'utility', r'firstenergy', r'pse&g', r'att bill payment', r'cricket wireless', r'phone', r'internet', r'ezpass'],
    'Groceries': [r'costco', r'stop shop', r'shoprite', r'trader joe', r'patel brothers', r'acme', r'whole foods', r'wegmans', r'wendys?'],
    'Dining': [r'restaurant', r'dhaba', r'panera', r'starbucks', r'bagel', r'wonder', r'burger', r'taco', r'pizza', r'paris baguette', r'sushi', r'dunkin', r'chipotle', r'mcdonald'],
    'Transport': [r'gas', r'quick chek', r'wawa', r'bp', r'exxon', r'uber', r'lyft', r'parking', r'tesla supercharger', r'nj ezpass'],
    'Shopping': [r'amazon', r'target', r'macys', r'lacoste', r'temu', r'underarmour', r'home depot', r'great clips', r'old navy', r'shein', r'zappos', r'kohls'],
    'Health': [r'walgreens', r'pharmacy', r'orthop', r'peds', r'diagnostics', r'optical', r'eyecare', r'ouraring', r'dent', r'fitness'],
    'Insurance': [r'geico', r'insurance'],
    'Subscriptions': [r'patreon', r'uber one', r'youtube', r'disney plus', r'hp instant ink', r'linkedin', r'tesla subscription', r'apple\\.combill', r'trendspider', r'x corp\\. paid features', r'quizlet'],
    'Travel': [r'airbnb', r'united', r'american air', r'alaska air', r'chase travel', r'cruise', r'dcl', r'frontier', r'parking auth'],
    'Transfers': [r'payment thank you', r'online payment', r'payment to chase card', r'online realtime transfer', r'online transfer to sav', r'zelle payment to', r'transfer to bofa'],
    'Fees': [r'fee', r'annual membership fee', r'late fee', r'interest charge'],
    'Education': [r'outschool', r'board of educa', r'taekwondo', r'swim school', r'quizlet', r'kids united'],
    'Entertainment': [r'sixflags', r'amc', r'urban air', r'patreon membership', r'glacier park', r'lake george', r'carnival cruise line res'],
}

def compact(s): return re.sub(r'\s+', ' ', str(s)).strip()
def norm(text): return unicodedata.normalize('NFKC', text or '').replace('','')

def parse_amount(v):
    s = str(v).strip().replace(',', '').replace('$', '').replace('O','0')
    if s.startswith('(') and s.endswith(')'): s = '-' + s[1:-1]
    try: return float(s)
    except: return np.nan

def pages(file_bytes):
    out=[]
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            out.append(norm(p.extract_text(x_tolerance=1, y_tolerance=1) or ''))
    return out

def text_all(file_bytes):
    return '\n'.join(pages(file_bytes))

def lines(text):
    return [compact(x) for x in text.splitlines() if compact(x)]

def classify(desc, amt):
    t = desc.lower()
    for cat,pats in CATEGORY_RULES.items():
        if any(re.search(p,t) for p in pats): return cat
    return 'Income' if amt > 0 else 'Other'

def finalize(rows, source):
    df = pd.DataFrame(rows)
    if df.empty: raise ValueError('No transactions extracted.')
    df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
    df = df[df['date'].notna() & df['net_amount'].notna()].copy()
    if df.empty: raise ValueError('No transactions extracted.')
    df['source_file'] = source
    if 'statement_category' not in df.columns: df['statement_category'] = 'GENERIC'
    return df[['date','description','net_amount','source_file','statement_category']]

def extract_chase_monthly(file_bytes, source):
    rows=[]
    for t in pages(file_bytes):
        for line in lines(t):
            m = re.match(r'^(\d{4})\s+(.+?)\s+(-?\$?\d[\d,]*\.\d{2})$', line)
            if m:
                mmdd, desc, amt = m.groups(); a=parse_amount(amt)
                if not np.isnan(a): rows.append({'date': f'{mmdd[:2]}/{mmdd[2:]}/2026', 'description': compact(desc), 'net_amount': a, 'statement_category': 'CHASE'})
                continue
            m = re.match(r'^(\d{2}/\d{2}(?:/\d{2,4})?)\s+(.+?)\s+(-?\$?\d[\d,]*\.\d{2})$', line)
            if m:
                d, desc, amt = m.groups(); a=parse_amount(amt)
                if not np.isnan(a): rows.append({'date': d if d.count('/')==2 else f'{d}/2026', 'description': compact(desc), 'net_amount': a, 'statement_category': 'CHASE'})
    if not rows: raise ValueError('No Chase monthly transactions extracted.')
    return finalize(rows, source)

def extract_chase_annual(file_bytes, source):
    rows=[]
    for t in pages(file_bytes):
        for line in lines(t):
            m = re.match(r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}.*?(-?\$?\d[\d,]*\.\d{2})$', line)
            if m:
                amt = m.group(1); a=parse_amount(amt)
                if not np.isnan(a):
                    d = re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}', line).group(0)
                    desc = compact(line.replace(d,'').rsplit(amt,1)[0])
                    rows.append({'date': d, 'description': desc, 'net_amount': -abs(a), 'statement_category': 'CHASE'})
    if not rows: raise ValueError('No Chase annual transactions extracted.')
    return finalize(rows, source)

def extract_citi_monthly(file_bytes, source):
    rows=[]
    for t in pages(file_bytes):
        for line in lines(t):
            if any(k in line.upper() for k in ['BILLING PERIOD','ACCOUNT SUMMARY','REWARDS','TOTAL','COSTCO ANYWHERE VISA CARD BY CITI']):
                continue
            m = re.match(r'^(\d{2}/\d{2}(?:/\d{2,4})?)\s+(.+?)\s+(-?\$?\d[\d,]*\.\d{2})$', line)
            if m:
                d, desc, amt = m.groups(); a=parse_amount(amt)
                if not np.isnan(a): rows.append({'date': d, 'description': compact(desc), 'net_amount': a, 'statement_category': 'CITI'})
                continue
            m = re.match(r'^(\d{4})\s+(\d{4})\s+(.+?)\s+(-?\$?\d[\d,]*\.\d{2})$', line)
            if m:
                sale, post, desc, amt = m.groups(); a=parse_amount(amt)
                if not np.isnan(a): rows.append({'date': f'{sale[:2]}/{sale[2:]}/2026', 'description': compact(desc), 'net_amount': a, 'statement_category': 'CITI'})
    if not rows: raise ValueError('No Citi monthly transactions extracted.')
    return finalize(rows, source)

def extract_citi_annual(file_bytes, source):
    rows=[]
    for t in pages(file_bytes):
        for line in lines(t):
            m = re.match(r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\s+(.+?)\s+(-?\$?\d[\d,]*\.\d{2})$', line)
            if m:
                desc, amt = m.groups(); a=parse_amount(amt)
                if not np.isnan(a):
                    d = re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}', line).group(0)
                    rows.append({'date': d, 'description': compact(desc), 'net_amount': a, 'statement_category': 'CITI'})
    if not rows: raise ValueError('No Citi annual transactions extracted.')
    return finalize(rows, source)

def extract_amex_monthly(file_bytes, source):
    rows=[]
    for t in pages(file_bytes):
        for line in lines(t):
            m = re.match(r'^(\d{2}/\d{2}(?:/\d{2,4})?)\s+(.+?)\s+(-?\$?\d[\d,]*\.\d{2})$', line)
            if m:
                d, desc, amt = m.groups(); a=parse_amount(amt)
                if not np.isnan(a): rows.append({'date': d, 'description': compact(desc), 'net_amount': a, 'statement_category': 'AMEX'})
    if not rows: raise ValueError('No Amex monthly transactions extracted.')
    return finalize(rows, source)

def extract_amex_annual(file_bytes, source):
    rows=[]
    for t in pages(file_bytes):
        for line in lines(t):
            m = re.match(r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\s+(.+?)\s+(-?\$?\d[\d,]*\.\d{2})$', line)
            if m:
                desc, amt = m.groups(); a=parse_amount(amt)
                if not np.isnan(a):
                    d = re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}', line).group(0)
                    rows.append({'date': d, 'description': compact(desc), 'net_amount': a, 'statement_category': 'AMEX'})
    if not rows: raise ValueError('No Amex annual transactions extracted.')
    return finalize(rows, source)

def extract_chase_checking(file_bytes, source):
    rows=[]
    for t in pages(file_bytes):
        for line in lines(t):
            m = re.match(r'^(\d{2}/\d{2})\s+(.+?)\s+(-?\d[\d,]*\.\d{2})$', line)
            if m:
                d, desc, amt = m.groups(); a=parse_amount(amt)
                if not np.isnan(a): rows.append({'date': f'{d}/2026', 'description': compact(desc), 'net_amount': a, 'statement_category': 'CHECKING'})
                continue
            m = re.match(r'^(\d{4})\s+(.+?)\s+(-?\$?\d[\d,]*\.\d{2})$', line)
            if m:
                mmdd, desc, amt = m.groups(); a=parse_amount(amt)
                if not np.isnan(a): rows.append({'date': f'{mmdd[:2]}/{mmdd[2:]}/2026', 'description': compact(desc), 'net_amount': a, 'statement_category': 'CHECKING'})
    if not rows: raise ValueError('No Chase checking transactions extracted.')
    return finalize(rows, source)

def extract_bofa(file_bytes, source):
    rows=[]
    for t in pages(file_bytes):
        for line in lines(t):
            m = re.match(r'^(\d{2}/\d{2}(?:/\d{2,4})?)\s+(.+?)\s+(-?\$?\d[\d,]*\.\d{2})$', line)
            if m:
                d, desc, amt = m.groups(); a=parse_amount(amt)
                if not np.isnan(a): rows.append({'date': d, 'description': compact(desc), 'net_amount': a, 'statement_category': 'BOFA'})
    if not rows: raise ValueError('No generic PDF transactions extracted.')
    return finalize(rows, source)

def extract_generic(file_bytes, source):
    rows=[]
    for t in pages(file_bytes):
        for line in lines(t):
            m = re.match(r'^(\d{2}/\d{2}(?:/\d{2,4})?)\s+(.+?)\s+(-?\d[\d,]*\.\d{2})$', line)
            if m:
                d, desc, amt = m.groups(); a=parse_amount(amt)
                if not np.isnan(a): rows.append({'date': d, 'description': compact(desc), 'net_amount': a, 'statement_category': 'GENERIC'})
    if not rows: raise ValueError('No generic PDF transactions extracted.')
    return finalize(rows, source)

def parse_pdf_statement(uploaded_file):
    file_bytes = uploaded_file.read()
    txt = text_all(file_bytes)
    up = txt.upper()
    nm = uploaded_file.name.lower()
    if ('COSTCO ANYWHERE VISA CARD BY CITI' in up) or ('CITI' in up and 'COSTCO' in up):
        try: return extract_citi_monthly(file_bytes, uploaded_file.name)
        except: pass
    if 'AMERICAN EXPRESS' in up and 'ANNUAL' in up:
        try: return extract_amex_annual(file_bytes, uploaded_file.name)
        except: pass
    if 'ANNUAL ACCOUNT SUMMARY' in up or ('SAPPHIRE' in up and 'ANNUAL' in up):
        try: return extract_chase_annual(file_bytes, uploaded_file.name)
        except: pass
    if 'CHECKING SUMMARY' in up or 'CHECKING' in up:
        try: return extract_chase_checking(file_bytes, uploaded_file.name)
        except: pass
    if ('SAPPHIRE' in up or 'CHASE' in up or 'AMAZON' in up or 'AMAZON' in nm or 'SAPPHIRE' in nm or 'CARDMEMBER SERVICE' in up):
        try: return extract_chase_monthly(file_bytes, uploaded_file.name)
        except: pass
    if 'AMERICAN EXPRESS' in up or 'AMEX' in up:
        try: return extract_amex_monthly(file_bytes, uploaded_file.name)
        except: pass
    if 'CITI' in up or 'COSTCO' in up:
        try: return extract_citi_annual(file_bytes, uploaded_file.name)
        except: pass
    if 'BANK OF AMERICA' in up or 'BANKOFAMERICA' in up:
        try: return extract_bofa(file_bytes, uploaded_file.name)
        except: pass
    return extract_generic(file_bytes, uploaded_file.name)

def normalize_categories(df):
    df = df.copy()
    df['category'] = [classify(d,a) for d,a in zip(df['description'], df['net_amount'])]
    return df

def analyze(df):
    df = normalize_categories(df)
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['income'] = np.where(df['net_amount']>0, df['net_amount'], 0.0)
    df['expense'] = np.where(df['net_amount']<0, -df['net_amount'], 0.0)
    monthly = df.groupby('month', as_index=False).agg(income=('income','sum'), expenses=('expense','sum'), transactions=('net_amount','count'))
    monthly['net'] = monthly['income'] - monthly['expenses']
    cat = df[df['expense']>0].groupby('category', as_index=False)['expense'].sum().sort_values('expense', ascending=False)
    return df.sort_values('date'), monthly.sort_values('month'), cat

def build_dataset(files):
    frames=[]; errors=[]
    for f in files:
        try:
            if f.name.lower().endswith('.pdf'): frames.append(parse_pdf_statement(f))
            elif f.name.lower().endswith('.csv'): frames.append(pd.read_csv(f))
            elif f.name.lower().endswith(('.xlsx','.xls')): frames.append(pd.read_excel(f))
            else: errors.append(f'{f.name}: unsupported file type')
        except Exception as e:
            errors.append(f'{f.name}: {e}')
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), errors)

def to_excel(monthly, cat, txns):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        monthly.to_excel(writer, index=False, sheet_name='Monthly Summary')
        cat.to_excel(writer, index=False, sheet_name='Categories')
        txns.to_excel(writer, index=False, sheet_name='Transactions')
    return buf.getvalue()

st.title('💼 Finance Statement Analyzer')
files = st.file_uploader('Upload statements', type=['pdf','csv','xlsx','xls'], accept_multiple_files=True)
if not files: st.stop()
df, errors = build_dataset(files)
for e in errors: st.error(e)
if df.empty: st.stop()
df, monthly, cat = analyze(df)
st.metric('Transactions', len(df))
st.metric('Files parsed', len(set(df['source_file'])))
fig = go.Figure()
fig.add_bar(x=monthly['month'], y=monthly['income'], name='Income')
fig.add_bar(x=monthly['month'], y=monthly['expenses'], name='Expenses')
fig.add_scatter(x=monthly['month'], y=monthly['net'], name='Net', mode='lines+markers')
fig.update_layout(barmode='group', height=420)
st.plotly_chart(fig, use_container_width=True)
st.dataframe(df.head(50), use_container_width=True)
st.download_button('Download workbook', to_excel(monthly, cat, df), 'finance_analysis.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
