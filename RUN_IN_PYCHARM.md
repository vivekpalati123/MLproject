# ▶️ How to Run SecureNet in PyCharm

## Step 1 — Open Project
1. Extract `SecureNet.zip` to your Desktop
2. Open **PyCharm** → **File** → **Open** → select the `SecureNet` folder → OK

## Step 2 — Set Up Python Interpreter
1. Go to **File** → **Settings** → **Project: SecureNet** → **Python Interpreter**
2. Click ⚙️ gear → **Add** → **Virtual Environment** → **New** → OK
3. PyCharm creates a `venv/` folder automatically

## Step 3 — Install Dependencies
Open **Terminal** (bottom of PyCharm) and run:
```
pip install -r requirements.txt
```

## Step 4 — Run the Dashboard (everything is already trained!)
In PyCharm Terminal:
```
python -m streamlit run app.py
```
Then open **http://localhost:8501** in your browser.

## That's it! All 20 pages will show real data immediately.

---

## Optional: Retrain Models
```
python src/train_pipeline.py
```

## Optional: Run Tests
```
python -m pytest tests/test_pipeline.py -v
```
