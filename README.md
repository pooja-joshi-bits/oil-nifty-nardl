# Replication files

Data and code for "Why Do Indian Stocks Rise with Oil Prices? Global Demand Shocks, the Rupee, and Fuel-Price Deregulation".

## Files
- `DF_with_globals.xlsx`: monthly data, 2000:01-2025:07 (Nifty 50, WTI, INR/USD, 10-year G-sec yield, WPI, IIP growth, M2, S&P 500, VIX, US dollar index, Brent).
- `oil_market.csv`: monthly world crude oil production (EIA INTL.57-1-WORL-TBPD.M), Kilian global real activity index (FRED IGREA), US CPI (FRED CPIAUCSL), and WTI spot (FRED WTISPLC), 1973:01-2025:07.
- `replicate.py`: Tables 1-9 and Figures 2-5.
- `replicate_revision.py`: Table 10, wild-bootstrap bands, and the joint Toda-Yamamoto tests.
- `replicate_oilshocks.py`: Section 5.9, Table 11, and Figure 6 (structural oil-market shocks).
- `bounds_mc.py`: finite-sample critical values and p-values for the bounds test (Section 5.2).

## How to run
```
pip install pandas numpy statsmodels scipy matplotlib openpyxl
python replicate.py DF_with_globals.xlsx
python replicate_revision.py DF_with_globals.xlsx .
python replicate_oilshocks.py DF_with_globals.xlsx oil_market.csv .
```
