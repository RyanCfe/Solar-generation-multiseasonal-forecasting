<div align="center">

# ☀️ Solar Generation Forecasting
### Capturing daily and yearly seasonality with Facebook Prophet

**Applied Time Series Analysis · Germany · Hourly data · 2015–2020**

*From raw observations and autocorrelation to interpretable forecasts, model comparison, and residual diagnostics.*

[**Report (PDF)**](ATSA_Project_Report.pdf) · [**Notebook**](ATSA_Project.ipynb) · [**Python Script**](ATSA_Project.py) · [**Dataset Notes**](DATASET.md)

</div>

![Historical solar-generation series](assets/historical-generation.png)

## The question

Solar power generation has a strong day–night rhythm, but summer and winter do not look alike. **How much do Prophet forecasts improve when we model both daily and yearly seasonality?** We use real hourly German generation data to test that question rather than assuming a complex model must be superior to a simple baseline.

## Key results

| Forecasting method | MAE (MW) ↓ | RMSE (MW) ↓ |
| :-- | --: | --: |
| Seasonal naive — previous day, same hour | **1,136.62** | **2,477.54** |
| Prophet — trend only | 5,227.64 | 8,411.00 |
| Prophet — trend + daily | 5,794.46 | 7,077.20 |
| **Prophet — trend + daily + yearly** | **2,907.83** | **3,762.52** |

**46.8% lower Prophet RMSE** when yearly seasonality is added to daily seasonality. The seasonal-naive rule still records the lowest errors overall **in this evaluation**. That is not a like-for-like contest: the baseline uses actual previous-day observations throughout 2019, whereas Prophet is fitted before 2019 and not updated during testing.

![Model RMSE comparison](assets/rmse-comparison.png)

## The investigation, from signals to forecasts

| Stage | What we did | Why it mattered |
| --- | --- | --- |
| **01 · Prepare** | Loaded OPSD hourly timestamps and German solar output; checked 104 missing generation values, zero duplicate timestamps and consistent hourly spacing | Preserved the meaning of 24-hour lags and distinguished missing readings from valid nighttime zeros |
| **02 · Discover** | Visualised the complete series, a June 2019 week, hourly/monthly averages and four seasonal daily profiles | Found a strong 24-hour cycle and broad seasonal changes |
| **03 · Measure** | Examined hourly ACF through 168 lags, highlighting 24, 48 and 72 hours | Confirmed strong daily-lag dependence |
| **04 · Benchmark** | Predicted each hour using the observation 24 hours earlier | Established a transparent baseline |
| **05 · Model** | Compared trend-only, daily and daily+yearly Prophet on the same pre-2019 training and 2019 test periods | Quantified the contribution of seasonal components |
| **06 · Diagnose** | Investigated component plots, residual ACF and mean absolute error by hour | Located predictable structure the final model still missed |

### The daily shape changes with the season

![Average daily solar profiles across four months](assets/seasonal-months.png)

These profiles show why an average 24-hour curve alone cannot fully describe the year. In January the typical production window is shorter and lower; summer produces broader, higher curves.

### ACF gives the 24-hour benchmark a reason to exist

![Hourly autocorrelation](assets/acf-daily-lags.png)

Daily-lag peaks motivate the seasonal-naive rule `prediction[t] = actual[t - 24]`. We retain the full hourly index **before** shifting so 24 rows really mean 24 hours.

### What the final Prophet model captured—and missed

![Prophet components](assets/prophet-components.png)

![Residual autocorrelation](assets/residual-acf.png)

The residual ACF retains daily structure; time-of-day seasonality alone is not enough to explain every solar fluctuation. Weather, capacity changes and model specification are possible contributors not directly tested here.

## Run the project

Python 3.10+ recommended. Place the CSV beside the `.py` and `.ipynb` files.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python ATSA_Project.py
```

For the annotated analysis and saved charts, open `ATSA_Project.ipynb` in Jupyter or VS Code and execute cells in order. Prophet fitting can take time; outputs in the notebook reflect the saved run.

## Dataset

- **Source:** [Open Power System Data — Time Series, 2020-10-06](https://data.open-power-system-data.org/time_series/2020-10-06/)
- **Target:** `DE_solar_generation_actual` (aggregated German solar generation, MW)
- **Time index:** `utc_timestamp`, hourly UTC
- **Extent:** 50,401 timestamps, late 2014 through September 2020
- **Quality:** 104 missing target values (0.21%); no duplicated timestamps or unexpected timestamp gaps

See [DATASET.md](DATASET.md) for provenance and handling notes.

## Project files

```text
.
├── ATSA_Project.ipynb                    # Executed and annotated analysis
├── ATSA_Project.py                       # Runnable Python version
├── ATSA_Project_Report.pdf               # Research report with figures
├── ATSA_Project_Report.docx              # Editable report
├── time_series_60min_singleindex_filtered.csv
├── assets/                               # Figures exported from the notebook
├── DATASET.md
├── requirements.txt
└── README.md
```

## Limitations and next steps

This study deliberately isolates time-based seasonal components. It does **not** add cloud cover, irradiance, temperature or installed-capacity covariates, and it does **not** claim that the baseline and Prophet use equal information during evaluation. A stronger follow-up would use rolling-origin **day-ahead** comparisons and weather regressors.

## Authors and references

**Ryan (S041) · Yugraj (S043) · Aleric (S050)**  \nMBA Tech Data Science, Semester V · NMIMS MPSTME · 2026–27

- Open Power System Data. [Time Series 2020-10-06](https://data.open-power-system-data.org/time_series/2020-10-06/).
- Taylor, S. J. & Letham, B. (2018). [Forecasting at Scale](https://doi.org/10.1080/00031305.2017.1380080).
- [Prophet documentation](https://facebook.github.io/prophet/docs/quick_start.html).
