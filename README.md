# Finance Statement Analyzer

A Streamlit app for analyzing bank, savings, and credit card statements from CSV, Excel, and PDF files.

## Features
- Upload multiple statement files at once
- Supports CSV, XLSX, XLS, and PDF
- Custom PDF parsing logic for Chase-style and Citi-style statements
- Monthly income, expenses, and net savings summary
- Automatic category tagging
- Spending charts and category breakdowns
- Savings suggestions based on discretionary spending patterns
- Downloadable Excel workbook of results

## Privacy
- The app processes uploaded files during the session.
- The app is written to avoid intentionally saving uploaded source files.
- Avoid adding custom logs or third-party analytics that may capture sensitive financial data.

## Supported inputs
### Tabular files
Use files with columns like:
- Date
- Description
- Amount

Or:
- Date
- Description
- Debit
- Credit

### PDF files
- Best for digital PDFs with selectable text
- Includes custom parsing logic for Chase and Citi layouts
- Some issuers and scanned PDFs may still need parser tuning or OCR

## Local setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `.gitignore`.
3. Go to https://streamlit.io/cloud
4. Create a new app.
5. Select your GitHub repository and branch.
6. Set the main file path to `app.py`.
7. Deploy.

## Recommended next improvements
- Add bank-specific templates for more issuers
- Add OCR support for scanned PDFs
- Add a parser review screen so users can approve extracted transactions before analysis
- Add recurring subscription detection and budgeting features
