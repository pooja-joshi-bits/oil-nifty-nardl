# Data and code for "Oil Price and Exchange Rate Shocks, Monetary Transmission, and Stock Market Dynamics in India: Evidence from a Nonlinear ARDL Framework"

Pooja Joshi, Vishruti Gupta, Pooja Sharma (2026). Submitted to *Resources Policy*.

## Files
- `DF_with_globals.xlsx` — monthly data, 2000:01–2025:07. Nifty 50, WTI oil, INR/USD, 10-year bond yield, WPI, IIP growth, M2, S&P 500, VIX, US dollar index, Brent.
- `replicate.py` — reproduces every table and figure in the paper.

## How to run
```
pip install pandas numpy statsmodels scipy matplotlib openpyxl
python replicate.py
```
Outputs: `results_main.json`, `results_global.json`, `results_supwald.json`, and the four figure PDFs. Runtime: a few minutes (the bootstraps are the slow part).

## Data sources
Nifty 50: NSE. Oil (WTI, month-end close): Yahoo Finance, CL=F. Exchange rate, bond yield, M2: RBI. WPI: Office of the Economic Adviser, DPIIT. IIP: MoSPI. S&P 500: Yahoo Finance, ^GSPC. VIX: FRED, VIXCLS (end of period). Brent: FRED, DCOILBRENTEU (end of period). US dollar index: month-end close.
