import io
import re
import unicodedata
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title='Finance Statement Analyzer', page_icon='💼', layout='wide')

CATEGORY_RULES = {
    'Income': [r'payroll', r'direct deposit', r'payment from', r'zelle payment from', r'interest', r'dividend', r'refund', r'credit'],
    'Housing': [r'rent', r'mortgage', r'hoa', r'clickpay', r'firstservice'],
    'Utilities': [r'utility', r'firstenergy', r'public services electric', r'pse&g', r'att bill payment', r'cricket wireless', r'phone', r'internet', r'ezpass'],
    'Groceries': [r'costco', r'stop shop', r'shoprite', r'trader joe', r'patel brothers', r'acme', r'whole foods', r'wegmans', r'apna bazar'],
    'Dining': [r'restaurant', r'dhaba', r'panera', r'starbucks', r'bagel', r'wonder', r'burger', r'taco', r'pizza', r'paris baguette', r'sushi'],
    'Transport': [r'gas', r'quick chek', r'wawa', r'bp', r'exxon', r'uber', r'lyft', r'parking', r'tesla supercharger', r'crown car wash', r'nj ezpass'],
    'Shopping': [r'amazon', r'target', r'macys', r'lacoste', r'costco.com', r'temu', r'underarmour', r'apple\.combill', r'home depot', r'great clips'],
    'Health': [r'walgreens', r'pharmacy', r'orthopaedic', r'peds', r'diagnostics', r'optical', r'eyecare', r'ouraring'],
    'Insurance': [r'geico', r'insurance', r'firstservice'],
    'Subscriptions': [r'patreon', r'uber one', r'youtube', r'disney plus', r'hp instant ink', r'linkedin', r'tesla subscription', r'apple\.combill'],
    'Travel': [r'airbnb', r'united', r'american air', r'alaska air', r'chase travel', r'cruise', r'dcl', r'frontier'],
    'Transfers': [r'payment thank you', r'online payment', r'payment to chase card', r'online realtime transfer', r'online transfer to sav', r'zelle payment to', r'robinhood debits', r'transfer to bofa'],
    'Fees': [r'fee', r'interest charge', r'annual membership fee', r'late fee'],
    'Education': [r'outschool', r'quizlet', r'board of educa', r'taekwondo', r'swim school'],
    'Entertainment': [r'sixflags', r'amc', r'urban air'],
}

MONTHS = '(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'


def parse_amount(v):
    s = str(v).strip().replace('$', '').replace(',', '').replace('O', '0')
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]
    try:
        return float(s)
    except Exception:
        return np.nan


def compact(s):
    return ' '.join(str(s).split())


def normalize_ocr(text):
    text = unicodedata.normalize('NFKC', text)
    fixes = {
        'MMaannaaggee': 'Manage', 'MMoobbiillee': 'Mobile', 'DDoowwnnllooaadd': 'Download',
        'LLaattaee': 'Late', 'LLatae': 'Late', 'Pptaeym enPt': 'Payment', 'Ppaym ent': 'Payment',
        'Ptaeym': 'Payment', 'wwww.chase.com': 'www.chase.com', 'w w w.chase.com': 'www.chase.com',
        'PPUURRCCHHAASSEE': 'PURCHASE', 'AACCCCOOUUNNTT': 'ACCOUNT', 'aaccttiivviittyy': 'activity'
    }
    for a, b in fixes.items():
        text = text.replace(a, b)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace(' .', '.').replace(' ,', ',')
    return text


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
            pages.append(normalize_ocr(p.extract_text(x_tolerance=1, y_tolerance=1) or ''))
    return pages


def detect_layout(text, filename):
    t = (text + ' ' + filename).lower()
    if 'spending report' in t and 'transaction date posted date description amount' in t:
        return 'chase_annual'
    if 'chase.com/amazon' in t or 'www.chase.com/amazon' in t or 'amazon chase' in t:
        return 'chase_monthly_amazon'
    if 'cardhelp' in t and 'ultimate rewards' in t:
        return 'chase_monthly_sapphire'
    if 'annual account summary' in t and 'citi' in t:
        return 'citi_annual'
    if 'costco anywhere visa card by citi' in t and 'cardholder summary' in t:
        return 'citi_monthly'
    if 'checking summary' in t and 'transaction detail' in t and 'jpmorgan chase bank' in t:
        return 'chase_checking'
    if 'platinum card' in t and 'american express' in t:
        return 'amex_monthly'
    return 'generic_bank'


def finalize_df(rows, source_name):
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('No transactions extracted.')
    df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
    df = df[df['date'].notna() & df['net_amount'].notna()].copy()
    df['source_file'] = source_name
    return df[['date', 'description', 'net_amount', 'source_file', 'statement_category']]


def split_ocr_lines(text):
    text = text.replace('ACCOUNT ACTIVITY', '\nACCOUNT ACTIVITY\n')
    text = text.replace('ACCOUNT ACTIVITY (CONTINUED)', '\nACCOUNT ACTIVITY (CONTINUED)\n')
    text = text.replace('PAYMENTS AND OTHER CREDITS', '\nPAYMENTS AND OTHER CREDITS\n')
    text = text.replace('PURCHASE ', '\nPURCHASE\n')
    text = text.replace('PURCHASES AND REDEMPTIONS', '\nPURCHASES AND REDEMPTIONS\n')
    text = text.replace('RETURNS AND OTHER CREDITS', '\nRETURNS AND OTHER CREDITS\n')
    return [compact(x) for x in text.split('\n') if compact(x)]


def stitch_chase_lines(lines):
    stitched = []
    buf = ''
    for line in lines:
        if re.match(r'^\d{4}\s', line) or re.match(r'^\d{2}/\d{2}/\d{2}\s', line) or re.match(r'^\w{3}\s+\d{1,2},\s+\d{4}\s', line):
            if buf:
                stitched.append(buf)
            buf = line
        else:
            if buf:
                buf = buf + ' ' + line
            else:
                stitched.append(line)
    if buf:
        stitched.append(buf)
    return stitched


def extract_chase_annual(file_bytes, source_name):
    txt = '\n'.join(read_pdf_pages(file_bytes))
    rows = []
    in_table = False
    current_cat = None
    date_pat = re.compile(r'^(?:[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})\s+(?:[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})\s+(.+?)\s+(-?[\d,]+\.\d{2})$')
    for raw in txt.split('\n'):
        line = compact(raw)
        if not line:
            continue
        if line in {'AUTOMOTIVE','BILLS_AND_UTILITIES','ENTERTAINMENT','EDUCATION','FOOD_AND_DRINK','GAS','GROCERIES','HEALTH_AND_WELLNESS','HOME','PERSONAL','SHOPPING','TRAVEL','GIFTS_AND_DONATIONS','FEES_AND_ADJUSTMENTS','PROFESSIONAL_SERVICES'}:
            current_cat = line
            continue
        if line.startswith('Transaction Date Posted Date Description Amount'):
            in_table = True
            continue
        if line.startswith('Total ') and in_table:
            in_table = False
            continue
        if not in_table:
            continue
        m = date_pat.match(line)
        if not m:
            continue
        desc, amt = m.groups()
        a = parse_amount(amt)
        if np.isnan(a):
            continue
        d = re.search(r'([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})', line)
        if not d:
            continue
        rows.append({'date': d.group(1), 'description': compact(desc), 'net_amount': -abs(a) if a > 0 else abs(a), 'statement_category': current_cat})
    if not rows:
        blocks = re.findall(r'(?ms)^([A-Z_ &/-]+)\nTransaction Date Posted Date Description Amount\n(.*?)(?=\n[A-Z_ &/-]+\nTransaction Date Posted Date Description Amount|\Z)', txt)
        for cat, block in blocks:
            for m in re.finditer(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4})\s+(?:[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})?\s*(.+?)\s+(-?[\d,]+\.\d{2})(?=\n|$)', block, flags=re.S):
                d, desc, amt = m.groups()
                a = parse_amount(amt)
                if np.isnan(a):
                    continue
                rows.append({'date': d, 'description': compact(desc), 'net_amount': -abs(a) if a > 0 else abs(a), 'statement_category': compact(cat)})
    if not rows:
        raise ValueError('No Chase annual transactions extracted.')
    return finalize_df(rows, source_name)


def extract_chase_monthly(file_bytes, source_name):
    rows = []
    for text in read_pdf_pages(file_bytes):
        lines = stitch_chase_lines(split_ocr_lines(text))
        mode = None
        for line in lines:
            if 'PAYMENTS AND OTHER CREDITS' in line:
                mode = 'credit'
                continue
            if line == 'PURCHASE':
                mode = 'purchase'
                continue
            if 'RETURNS AND OTHER CREDITS' in line:
                mode = 'credit'
                continue
            if 'PURCHASES AND REDEMPTIONS' in line:
                mode = 'points'
                continue
            if any(skip in line for skip in ['ACCOUNT SUMMARY', 'ACCOUNT ACTIVITY', 'INTEREST CHARGES', 'YOUR PRIME VISA POINTS', 'ULTIMATE REWARDS', 'SHOP WITH POINTS ACTIVITY', 'YEAR-TO-DATE', 'REWARDS', 'SUMMARY', 'Statement Date']):
                continue
            m = re.match(r'^(\d{4})\s+(.+?)\s+(-?[\d,]+\.\d{2})(?:\s+Order Number.*)?$', line)
            if m:
                mmdd, desc, amt = m.groups()
                a = parse_amount(amt)
                if np.isnan(a):
                    continue
                mm, dd = mmdd[:2], mmdd[2:]
                year = 2025 if mm == '12' else 2026
                rows.append({'date': f'{mm}/{dd}/{year}', 'description': compact(desc), 'net_amount': a, 'statement_category': mode})
                continue
            m = re.match(r'^(\d{2}/\d{2}/\d{2})\s+(.+?)\s+(-?[\d,]+\.\d{2})(?:\s+Order Number.*)?$', line)
            if m:
                d, desc, amt = m.groups()
                a = parse_amount(amt)
                if np.isnan(a):
                    continue
                rows.append({'date': d, 'description': compact(desc), 'net_amount': a, 'statement_category': mode})
    if not rows:
        raise ValueError('No Chase monthly transactions extracted.')
    return finalize_df(rows, source_name)


def extract_citi_annual(file_bytes, source_name):
    txt = '\n'.join(read_pdf_pages(file_bytes))
    rows = []
    for m in re.finditer(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4})\s+(.+?)\s+(-?[\d,]+\.\d{2})(?=\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}|\s*$)', txt, flags=re.S):
        d, desc, amt = m.groups()
        a = parse_amount(amt)
        if np.isnan(a):
            continue
        rows.append({'date': d, 'description': compact(desc), 'net_amount': -abs(a) if a > 0 else abs(a), 'statement_category': None})
    if not rows:
        raise ValueError('No Citi annual transactions extracted.')
    return finalize_df(rows, source_name)


def extract_citi_monthly(file_bytes, source_name):
    rows = []
    for text in read_pdf_pages(file_bytes):
        for raw in text.split('\n'):
            line = compact(raw)
            m = re.match(r'^(\d{4})\s+(\d{4})\s+(.+?)\s+(-?[\d,]+\.\d{2})$', line)
            if not m:
                continue
            sale, post, desc, amt = m.groups()
            a = parse_amount(amt)
            if np.isnan(a):
                continue
            year = 2025 if sale.startswith('12') else 2026
            rows.append({'date': f'{sale[:2]}/{sale[2:]}/{year}', 'description': compact(desc), 'net_amount': a, 'statement_category': None})
    if not rows:
        raise ValueError('No Citi monthly 
