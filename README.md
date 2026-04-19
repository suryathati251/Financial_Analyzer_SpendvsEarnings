# Private Finance Statement Analyzer

A privacy-first Streamlit app that lets users upload checking, savings, and credit card statements and view:
- Monthly income and expenses
- Expense categories
- Net savings and savings rate
- Suggestions on where to reduce spending

## Privacy model
- Uploaded files are processed in memory during the active session.
- The app does not intentionally save uploaded statement files.
- Do not add logging of uploaded file contents.

## Expected input
Upload CSV or Excel files with columns such as:
- Date
- Description
- Amount

Or:
- Date
- Description
- Debit
- Credit

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Push this folder to a GitHub repository.
2. Go to https://streamlit.io/cloud
3. Create a new app.
4. Select your GitHub repo.
5. Set the main file path to `app.py`.
6. Deploy.

## Important note
This app is designed to avoid persistent storage of uploaded files in app logic, but you should still review Streamlit hosting behavior and avoid adding analytics or logs that could capture sensitive data.
