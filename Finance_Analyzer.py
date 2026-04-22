import io
import re
import unicodedata

import numpy as np
import pandas as pd
import pdfplumber

CATEGORY_RULES = {
    "Income": [r"payroll", r"direct deposit", r"payment from", r"zelle payment from", r"interest", r"dividend", r"refund", r"credit"],
    "Housing": [r"rent", r"mortgage", r"hoa", r"clickpay", r"firstservice"],
    "Utilities": [r"utility", r"firstenergy", r"public services electric", r"pse&g", r"att bill payment", r"cricket wireless", r"phone", r"internet", r"ezpass"],
    "Groceries": [r"costco", r"stop shop", r"shoprite", r"trader joe", r"patel brothers", r"acme", r"whole foods", r"wegmans", r"apna bazar"],
    "Dining": [r"restaurant", r"dhaba", r"panera", r"starbucks", r"bagel", r"wonder", r"burger", r"taco", r"pizza", r"paris baguette", r"sushi"],
    "Transport": [r"gas", r"quick chek", r"wawa", r"bp", r"exxon", r"uber", r"lyft", r"parking", r"tesla supercharger", r"crown car wash", r"nj ezpass"],
    "Shopping": [r"amazon", r"target", r"macys", r"lacoste", r"costco.com", r"temu", r"underarmour", r"apple\.combill", r"home depot", r"great clips"],
    "Health": [r"walgreens", r"pharmacy", r"orthopaedic", r"peds", r"diagnostics", r"optical", r"eyecare", r"ouraring"],
    "Insurance": [r"geico", r"insurance", r"firstservice"],
    "Subscriptions": [r"patreon", r"uber one", r"youtube", r"disney plus", r"hp instant ink", r"linkedin", r"tesla subscription", r"apple\.combill"],
    "Travel": [r"airbnb", r"united", r"american air", r"alaska air", r"chase travel", r"cruise", r"dcl", r"frontier"],
    "Transfers": [r"payment thank you", r"online payment", r"payment to chase card", r"online realtime transfer", r"online transfer to sav", r"zelle payment to", r"robinhood debits", r"transfer to bofa"],
    "Fees": [r"fee", r"interest charge", r"annual membership fee", r"late fee"],
    "Education": [r"outschool", r"quizlet", r"board of educa", r"taekwondo", r"swim school"],
    "Entertainment": [r"sixflags", r"amc", r"urban air"],
}

MONTH_NAMES = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
FULL_MONTH_NAMES = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
AMOUNT = r"\(?-?\$?\d[\d,]*\.\d{2}\)?"


def parse_amount(v):
    s = str(v).strip().replace("$", "").replace(",", "").replace("O", "0")
    compacted = s
    ocr_short_thousands = re.match(r"^(\(?-?\d{1,2})(\d{2})\.(\d{2}\)?)$", compacted)
    original = str(v).strip().replace("$", "").replace("O", "0")
    if "," in original and re.match(r"^\(?-?\d{1,2},\d{2}\.\d{2}\)?$", original):
        s = re.sub(r",", "", original)
        s = re.sub(r"(\d{2})\.", r"\g<1>0.", s)
    elif ocr_short_thousands:
        s = compacted
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except Exception:
        return np.nan


def compact(s):
    return " ".join(str(s).split())


def normalize_ocr(text):
    """Clean OCR text while preserving line breaks for table parsers."""
    text = unicodedata.normalize("NFKC", text or "")
    fixes = {
        "MMaannaaggee": "Manage",
        "MMoobbiillee": "Mobile",
        "DDoowwnnllooaadd": "Download",
        "LLaattaee": "Late",
        "LLatae": "Late",
        "Pptaeym enPt": "Payment",
        "Ppaym ent": "Payment",
        "Ptaeym": "Payment",
        "wwww.chase.com": "www.chase.com",
        "w w w.chase.com": "www.chase.com",
        "PPUURRCCHHAASSEE": "PURCHASE",
        "AACCCCOOUUNNTT": "ACCOUNT",
        "aaccttiivviittyy": "activity",
    }
    for a, b in fixes.items():
        text = text.replace(a, b)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t\f\v]+", " ", line).strip()
        line = line.replace(" .", ".").replace(" ,", ",")
        if line:
            lines.append(line)
    return "\n".join(lines)


def classify_category(desc, amount):
    text = str(desc).lower()
    for cat, pats in CATEGORY_RULES.items():
        if any(re.search(p, text) for p in pats):
            return cat
    return "Income" if amount > 0 else "Other"


def read_pdf_pages(file_bytes):
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
            pages.append(normalize_ocr(text))
    return pages


def detect_layout(text, filename):
    t = compact(text + " " + filename).lower()
    if "spending report" in t and "transaction date posted date description amount" in t:
        return "chase_annual"
    if "chase.com/amazon" in t or "www.chase.com/amazon" in t or "amazon chase" in t:
        return "chase_monthly_amazon"
    if "cardhelp" in t and "ultimate rewards" in t:
        return "chase_monthly_sapphire"
    if "annual account summary" in t and "citi" in t:
        return "citi_annual"
    if "costco anywhere visa card by citi" in t and "cardholder summary" in t:
        return "citi_monthly"
    if "checking summary" in t and "transaction detail" in t and "jpmorgan chase bank" in t:
        return "chase_checking"
    if "year-end summary" in t and "platinum card" in t:
        return "amex_annual"
    if "platinum card" in t and "american express" in t:
        return "amex_monthly"
    return "generic_bank"


def statement_year(text, fallback=2026):
    slash_range = re.search(r"\b\d{2}/\d{2}/(\d{2,4})\s*-\s*\d{2}/\d{2}/(\d{2,4})\b", text)
    if slash_range:
        end = slash_range.group(2)
        return int(end) + 2000 if len(end) == 2 else int(end)
    word_range = re.search(rf"\b(?:{MONTH_NAMES}|{FULL_MONTH_NAMES})\s+\d{{1,2}},\s+(20\d{{2}})\s+through\s+(?:{MONTH_NAMES}|{FULL_MONTH_NAMES})\s+\d{{1,2}},\s+(20\d{{2}})\b", text)
    if word_range:
        return int(word_range.group(2))
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]
    return max(set(years), key=years.count) if years else fallback


def infer_year(mmdd, text, fallback=2026):
    mmdd = mmdd.replace("/", "")
    month = int(mmdd[:2])
    slash_range = re.search(r"\b(\d{2})/\d{2}/(\d{2,4})\s*-\s*(\d{2})/\d{2}/(\d{2,4})\b", text)
    if slash_range:
        start_month, start_year, end_month, end_year = slash_range.groups()
        start_year = int(start_year) + 2000 if len(start_year) == 2 else int(start_year)
        end_year = int(end_year) + 2000 if len(end_year) == 2 else int(end_year)
        if start_year != end_year:
            return start_year if month >= int(start_month) else end_year
        return end_year
    word_range = re.search(rf"\b({MONTH_NAMES}|{FULL_MONTH_NAMES})\s+\d{{1,2}},\s+(20\d{{2}})\s+through\s+({MONTH_NAMES}|{FULL_MONTH_NAMES})\s+\d{{1,2}},\s+(20\d{{2}})\b", text)
    if word_range:
        start_name, start_year, end_name, end_year = word_range.groups()
        months = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
        start_month = months[start_name[:3]]
        start_year, end_year = int(start_year), int(end_year)
        if start_year != end_year:
            return start_year if month >= start_month else end_year
        return end_year
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]
    if not years:
        return fallback - 1 if month == 12 and fallback == 2026 else fallback
    current = max(set(years), key=years.count)
    return current - 1 if month == 12 and any(m in text for m in ("01/", "0101", "Jan")) else current


def finalize_df(rows, source_name):
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No transactions extracted.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df["net_amount"] = pd.to_numeric(df["net_amount"], errors="coerce")
    df = df[df["date"].notna() & df["net_amount"].notna()].copy()
    if df.empty:
        raise ValueError("Transactions were found, but dates or amounts could not be parsed.")
    df["source_file"] = source_name
    if "statement_category" not in df:
        df["statement_category"] = None
    return df[["date", "description", "net_amount", "source_file", "statement_category"]]


def split_ocr_lines(text):
    markers = [
        "ACCOUNT ACTIVITY",
        "ACCOUNT ACTIVITY (CONTINUED)",
        "PAYMENTS AND OTHER CREDITS",
        "PURCHASES AND REDEMPTIONS",
        "RETURNS AND OTHER CREDITS",
    ]
    for marker in markers:
        text = text.replace(marker, f"\n{marker}\n")
    text = re.sub(r"\bPURCHASE\b", "\nPURCHASE\n", text)
    return [compact(x) for x in text.splitlines() if compact(x)]


def stitch_chase_lines(lines):
    stitched = []
    buf = ""
    date_start = re.compile(r"^(?:\d{4}|\d{2}/\d{2}/\d{2}|[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})\s")
    for line in lines:
        if date_start.match(line):
            if buf:
                stitched.append(buf)
            buf = line
        elif buf and not re.search(rf"\s{AMOUNT}(?:\s|$)", buf):
            buf = buf + " " + line
        else:
            if buf:
                stitched.append(buf)
                buf = ""
            stitched.append(line)
    if buf:
        stitched.append(buf)
    return stitched


def extract_chase_annual(file_bytes, source_name):
    txt = "\n".join(read_pdf_pages(file_bytes))
    rows = []
    current_cat = None
    in_table = False
    cats = {
        "AUTOMOTIVE", "BILLS_AND_UTILITIES", "ENTERTAINMENT", "EDUCATION", "FOOD_AND_DRINK",
        "GAS", "GROCERIES", "HEALTH_AND_WELLNESS", "HOME", "PERSONAL", "SHOPPING", "TRAVEL",
        "GIFTS_AND_DONATIONS", "FEES_AND_ADJUSTMENTS", "PROFESSIONAL_SERVICES",
    }
    row_pat = re.compile(rf"^({MONTH_NAMES}\s+\d{{1,2}},\s+\d{{4}})\s+{MONTH_NAMES}\s+\d{{1,2}},\s+\d{{4}}\s+(.+?)\s+({AMOUNT})$")
    for raw in txt.splitlines():
        line = compact(raw)
        if line in cats:
            current_cat = line
            continue
        if line.startswith("Transaction Date Posted Date Description Amount"):
            in_table = True
            continue
        if line.startswith("Total ") and in_table:
            in_table = False
            continue
        if not in_table:
            continue
        m = row_pat.match(line)
        if m:
            d, desc, amt = m.groups()
            a = parse_amount(amt)
            rows.append({"date": d, "description": compact(desc), "net_amount": -abs(a) if a > 0 else abs(a), "statement_category": current_cat})
    if not rows:
        raise ValueError("No Chase annual transactions extracted.")
    return finalize_df(rows, source_name)


def extract_chase_monthly(file_bytes, source_name):
    rows = []
    pages = read_pdf_pages(file_bytes)
    full_text = "\n".join(pages)
    fallback_year = statement_year(full_text)
    for text in pages:
        lines = stitch_chase_lines(split_ocr_lines(text))
        mode = None
        for line in lines:
            if "PAYMENTS AND OTHER CREDITS" in line or "RETURNS AND OTHER CREDITS" in line:
                mode = "credit"
                continue
            if line == "PURCHASE":
                mode = "purchase"
                continue
            if "PURCHASES AND REDEMPTIONS" in line:
                mode = "points"
                continue
            if any(skip in line for skip in ["ACCOUNT SUMMARY", "ACCOUNT ACTIVITY", "INTEREST CHARGES", "YOUR PRIME VISA POINTS", "ULTIMATE REWARDS", "SHOP WITH POINTS ACTIVITY", "YEAR-TO-DATE", "REWARDS", "SUMMARY", "Statement Date"]):
                continue
            m = re.match(rf"^(\d{{4}}|\d{{2}}/\d{{2}})\s+(.+?)\s+({AMOUNT})(?:\s+Order Number.*)?$", line)
            if m:
                mmdd, desc, amt = m.groups()
                a = parse_amount(amt)
                year = infer_year(mmdd, full_text, fallback_year)
                date_txt = f"{mmdd[:2]}/{mmdd[2:]}/{year}" if "/" not in mmdd else f"{mmdd}/{year}"
                net = -abs(a) if mode in ("purchase", "points") else abs(a)
                rows.append({"date": date_txt, "description": compact(desc), "net_amount": net, "statement_category": mode})
                continue
            m = re.match(rf"^(\d{{2}}/\d{{2}}/\d{{2}})\s+(.+?)\s+({AMOUNT})(?:\s+Order Number.*)?$", line)
            if m:
                d, desc, amt = m.groups()
                rows.append({"date": d, "description": compact(desc), "net_amount": parse_amount(amt), "statement_category": mode})
    if not rows:
        raise ValueError("No Chase monthly transactions extracted.")
    return finalize_df(rows, source_name)


def extract_citi_annual(file_bytes, source_name):
    txt = "\n".join(read_pdf_pages(file_bytes))
    rows = []
    current_cat = None
    line_pat = re.compile(rf"^({MONTH_NAMES}\s+\d{{1,2}},\s+\d{{4}})\s+(.+?)\s+({AMOUNT})$")
    for raw in txt.splitlines():
        line = compact(raw)
        if not line or line.startswith(("This report was generated", "If you earn rewards", "on your rewards", "Subtotal", "Total ")):
            continue
        if line in {"Entertainment", "Merchandise", "Organizations", "Restaurants", "Services", "Vehicle Services"}:
            current_cat = line
            continue
        m = line_pat.match(line)
        if not m:
            continue
        d, desc, amt = m.groups()
        a = parse_amount(amt)
        rows.append({"date": d, "description": compact(desc), "net_amount": -abs(a) if a > 0 else abs(a), "statement_category": current_cat})
    if not rows:
        raise ValueError("No Citi annual transactions extracted.")
    return finalize_df(rows, source_name)


def extract_amex_annual(file_bytes, source_name):
    txt = "\n".join(read_pdf_pages(file_bytes))
    rows = []
    current_cat = None
    line_pat = re.compile(rf"^(\d{{2}}/\d{{2}}/\d{{4}})\s+([A-Za-z]+)\s+(.+?)\s+({AMOUNT})$")
    skip_prefixes = (
        "2025 Year-End Summary",
        "Includes charges",
        "Prepared for",
        "Card Member",
        "Date Month Billed",
        "Subtotal",
        "Monthly Totals",
        "Total Spending",
    )
    for raw in txt.splitlines():
        line = compact(raw)
        if not line or line.startswith(skip_prefixes):
            continue
        if re.fullmatch(r"[A-Za-z &]+", line) and len(line.split()) <= 4:
            current_cat = line
            continue
        m = line_pat.match(line)
        if not m:
            continue
        d, _billed_month, desc, amt = m.groups()
        amount = parse_amount(amt)
        desc_l = desc.lower()
        is_credit = any(word in desc_l for word in ("credit", "refund", "reimbursement", "payment thank you"))
        net = abs(amount) if is_credit or amount < 0 else -abs(amount)
        rows.append({"date": d, "description": compact(desc), "net_amount": net, "statement_category": current_cat})
    if not rows:
        raise ValueError("No Amex annual transactions extracted.")
    return finalize_df(rows, source_name)


def extract_citi_monthly(file_bytes, source_name):
    rows = []
    pages = read_pdf_pages(file_bytes)
    full_text = "\n".join(pages)
    fallback_year = statement_year(full_text)
    for text in pages:
        for raw in text.splitlines():
            line = compact(raw)
            m = re.match(rf"^(\d{{4}})\s+(\d{{4}})\s+(.+?)\s+({AMOUNT})$", line)
            if not m:
                continue
            sale, _post, desc, amt = m.groups()
            year = infer_year(sale, full_text, fallback_year)
            rows.append({"date": f"{sale[:2]}/{sale[2:]}/{year}", "description": compact(desc), "net_amount": parse_amount(amt), "statement_category": None})
    if not rows:
        raise ValueError("No Citi monthly transactions extracted.")
    return finalize_df(rows, source_name)


def extract_amex_monthly(file_bytes, source_name):
    rows, mode = [], None
    pages = read_pdf_pages(file_bytes)
    full_text = "\n".join(pages)
    fallback_year = statement_year(full_text)
    for text in pages:
        for raw in text.splitlines():
            line = compact(raw)
            if "Payments Details" in line or "Credits Details" in line:
                mode = "credit"
                continue
            if "New Charges Details" in line:
                mode = "charge"
                continue
            m = re.match(rf"^(\d{{2}}/\d{{2}}/\d{{2,4}})\*?\s+(.+?)\s+(?:Pay In Full|Pay Over Time(?:\s+and/or Cash)?|Cash Advance)?\s*({AMOUNT})$", line)
            if not m:
                m = re.match(rf"^(\d{{4,8}})\s+(.+?)\s+({AMOUNT})$", line)
            if not m:
                continue
            date_raw, desc, amt = m.groups()
            a = parse_amount(amt)
            if "/" in date_raw:
                date_txt = date_raw
            elif len(date_raw) == 8:
                date_txt = f"{date_raw[:2]}/{date_raw[2:4]}/{date_raw[4:]}"
            else:
                year = infer_year(date_raw, full_text, fallback_year)
                date_txt = f"{date_raw[:2]}/{date_raw[2:4]}/{year}"
            net = -abs(a) if mode == "charge" else abs(a)
            rows.append({"date": date_txt, "description": compact(desc), "net_amount": net, "statement_category": None})
    if not rows:
        raise ValueError("No Amex monthly transactions extracted.")
    return finalize_df(rows, source_name)


def extract_chase_checking(file_bytes, source_name):
    rows = []
    pages = read_pdf_pages(file_bytes)
    full_text = "\n".join(pages)
    fallback_year = statement_year(full_text)
    for text in pages:
        for raw in text.splitlines():
            line = compact(raw)
            line = re.sub(r"^\*end\*transac\d*tion detail", "", line).strip()
            m = re.match(rf"^(\d{{4}}|\d{{2}}/\d{{2}})(?:\s+(?:\d{{4}}|\d{{2}}/\d{{2}}))?\s+(.+?)\s+({AMOUNT})\s+({AMOUNT})$", line)
            if m:
                mmdd, desc, amt, _bal = m.groups()
                year = infer_year(mmdd, full_text, fallback_year)
                date_txt = f"{mmdd[:2]}/{mmdd[2:]}/{year}" if "/" not in mmdd else f"{mmdd}/{year}"
                rows.append({"date": date_txt, "description": compact(desc), "net_amount": parse_amount(amt), "statement_category": None})
    if not rows:
        raise ValueError("No Chase checking transactions extracted.")
    return finalize_df(rows, source_name)


def extract_generic_bank(file_bytes, source_name):
    rows = []
    pages = read_pdf_pages(file_bytes)
    full_text = "\n".join(pages)
    fallback_year = statement_year(full_text)
    line_patterns = [
        re.compile(rf"^(\d{{1,2}}/\d{{1,2}}/\d{{2,4}})\s+(.+?)\s+({AMOUNT})(?:\s+{AMOUNT})?$"),
        re.compile(rf"^(\d{{4}})\s+(.+?)\s+({AMOUNT})(?:\s+{AMOUNT})?$"),
        re.compile(rf"^({MONTH_NAMES}\s+\d{{1,2}},?\s+\d{{4}})\s+(.+?)\s+({AMOUNT})(?:\s+{AMOUNT})?$", re.I),
    ]
    for text in pages:
        for raw in text.splitlines():
            line = compact(raw)
            for pat in line_patterns:
                m = pat.match(line)
                if not m:
                    continue
                date_txt, desc, amt = m.groups()
                if re.fullmatch(r"\d{4}", date_txt):
                    year = infer_year(date_txt, full_text, fallback_year)
                    date_txt = f"{date_txt[:2]}/{date_txt[2:]}/{year}"
                rows.append({"date": date_txt, "description": compact(desc), "net_amount": parse_amount(amt), "statement_category": None})
                break
    if not rows:
        sample = "\n".join(compact(x) for page in pages for x in page.splitlines()[:8])
        raise ValueError(f"No generic PDF transactions extracted. Detected text sample: {sample[:700]}")
    return finalize_df(rows, source_name)


def parse_pdf_statement(uploaded_file):
    file_bytes = uploaded_file.read()
    text = "\n".join(read_pdf_pages(file_bytes))
    layout = detect_layout(text, uploaded_file.name)
    if layout == "chase_annual":
        return extract_chase_annual(file_bytes, uploaded_file.name), layout
    if layout in ("chase_monthly_amazon", "chase_monthly_sapphire"):
        return extract_chase_monthly(file_bytes, uploaded_file.name), layout
    if layout == "citi_annual":
        return extract_citi_annual(file_bytes, uploaded_file.name), layout
    if layout == "citi_monthly":
        return extract_citi_monthly(file_bytes, uploaded_file.name), layout
    if layout == "amex_annual":
        return extract_amex_annual(file_bytes, uploaded_file.name), layout
    if layout == "amex_monthly":
        return extract_amex_monthly(file_bytes, uploaded_file.name), layout
    if layout == "chase_checking":
        return extract_chase_checking(file_bytes, uploaded_file.name), layout
    return extract_generic_bank(file_bytes, uploaded_file.name), layout


def build_dataset(files):
    frames, errors, layouts = [], [], []
    for f in files:
        try:
            name = f.name.lower()
            if name.endswith(".pdf"):
                parsed, layout = parse_pdf_statement(f)
                frames.append(parsed)
                layouts.append({"file": f.name, "detected_layout": layout, "rows": len(parsed)})
            elif name.endswith(".csv"):
                df = pd.read_csv(f)
                frames.append(df)
                layouts.append({"file": f.name, "detected_layout": "csv", "rows": len(df)})
            elif name.endswith(".xlsx") or name.endswith(".xls"):
                df = pd.read_excel(f)
                frames.append(df)
                layouts.append({"file": f.name, "detected_layout": "excel", "rows": len(df)})
            else:
                errors.append(f"{f.name}: unsupported file type")
        except Exception as e:
            errors.append(f"{f.name}: {e}")
    if not frames:
        return pd.DataFrame(), errors, pd.DataFrame(layouts)
    return pd.concat(frames, ignore_index=True), errors, pd.DataFrame(layouts)


def normalize_categories(df):
    df = df.copy()
    mapping = {
        "AUTOMOTIVE": "Transport",
        "BILLS_AND_UTILITIES": "Utilities",
        "ENTERTAINMENT": "Entertainment",
        "EDUCATION": "Education",
        "FOOD_AND_DRINK": "Dining",
        "GAS": "Transport",
        "GROCERIES": "Groceries",
        "HEALTH_AND_WELLNESS": "Health",
        "HOME": "Housing",
        "PERSONAL": "Other",
        "SHOPPING": "Shopping",
        "TRAVEL": "Travel",
        "GIFTS_AND_DONATIONS": "Other",
        "FEES_AND_ADJUSTMENTS": "Fees",
        "PROFESSIONAL_SERVICES": "Other",
    }
    df["category"] = [
        mapping.get(sc, classify_category(d, a))
        for sc, d, a in zip(df.get("statement_category", [None] * len(df)), df["description"], df["net_amount"])
    ]
    return df


def analyze(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df["net_amount"] = pd.to_numeric(df["net_amount"], errors="coerce")
    df = df[df["date"].notna() & df["net_amount"].notna()].copy()
    if "description" not in df:
        df["description"] = ""
    if "source_file" not in df:
        df["source_file"] = "uploaded file"
    if "statement_category" not in df:
        df["statement_category"] = None
    df = normalize_categories(df)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["income"] = np.where(df["net_amount"] > 0, df["net_amount"], 0.0)
    df["expense"] = np.where(df["net_amount"] < 0, -df["net_amount"], 0.0)
    monthly = df.groupby("month", as_index=False).agg(income=("income", "sum"), expenses=("expense", "sum"), transactions=("net_amount", "count"))
    monthly["net"] = monthly["income"] - monthly["expenses"]
    monthly["savings_rate"] = np.where(monthly["income"] > 0, (monthly["net"] / monthly["income"]) * 100, np.nan)
    cat = df[df["expense"] > 0].groupby("category", as_index=False)["expense"].sum().sort_values("expense", ascending=False)
    return df.sort_values("date"), monthly.sort_values("month"), cat


def to_excel(monthly, cat, txns):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        monthly.to_excel(writer, index=False, sheet_name="Monthly Summary")
        cat.to_excel(writer, index=False, sheet_name="Categories")
        txns.to_excel(writer, index=False, sheet_name="Transactions")
    return buf.getvalue()


def main():
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="Finance Statement Analyzer", page_icon="💼", layout="wide")
    st.title("💼 Finance Statement Analyzer")
    files = st.file_uploader("Upload statements", type=["pdf", "csv", "xlsx", "xls"], accept_multiple_files=True)
    if not files:
        st.stop()

    df, errors, layouts = build_dataset(files)
    if not layouts.empty:
        with st.expander("Parser diagnostics", expanded=bool(errors)):
            st.dataframe(layouts, use_container_width=True)
    for e in errors:
        st.error(e)
    if df.empty:
        st.stop()

    df, monthly, cat = analyze(df)
    col1, col2 = st.columns(2)
    col1.metric("Transactions", len(df))
    col2.metric("Files parsed", df["source_file"].nunique())

    fig = go.Figure()
    fig.add_bar(x=monthly["month"], y=monthly["income"], name="Income")
    fig.add_bar(x=monthly["month"], y=monthly["expenses"], name="Expenses")
    fig.add_scatter(x=monthly["month"], y=monthly["net"], name="Net", mode="lines+markers")
    fig.update_layout(barmode="group", height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df.head(50), use_container_width=True)
    st.download_button(
        "Download workbook",
        to_excel(monthly, cat, df),
        "finance_analysis.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
