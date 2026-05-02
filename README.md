# Flight Price Prediction — Databricks

A PySpark-based machine learning pipeline for predicting US domestic flight fares using the Expedia Itineraries dataset (~82M rows). The project covers end-to-end data engineering, exploratory analysis, and a progression of linear regression models, all running on Databricks with data stored in Azure Blob Storage.

---

## Overview

Airline pricing is notoriously complex, driven by route demand, seasonality, booking lead time, seat class, and carrier strategy. This project builds a reproducible pipeline to explore those drivers and establish a regression baseline, culminating in a model that explains ~63% of fare variance on a held-out test set.

---

## Dataset

| Property | Detail |
|---|---|
| Source | Expedia Itineraries (Kaggle) |
| Storage | Azure Blob Storage (`myflightdata` container) |
| Format | CSV → converted to Parquet after preprocessing |
| Scale | ~82 million rows |

Key columns used: `flightDate`, `searchDate`, `startingAirport`, `destinationAirport`, `travelDuration`, `isBasicEconomy`, `isRefundable`, `isNonStop`, `baseFare`, `totalFare`, `seatsRemaining`, `totalTravelDistance`, `segmentsAirlineName`, `segmentsEquipmentDescription`

---

## Pipeline

### 1. Data Ingestion
Mounts Azure Blob Storage via `wasbs://` and reads the CSV with schema inference using Spark.

### 2. Feature Engineering
| Feature | Description |
|---|---|
| `taxes` | `totalFare - baseFare` |
| `daysFrom` | Days between search date and flight date (booking lead time) |
| `flightMonth` / `flightYear` | Extracted from `flightDate` for seasonality analysis |
| `travelDuration` | Parsed from ISO 8601 format (e.g. `PT2H30M`) to decimal hours |
| Boolean columns | Cast to integers (`isBasicEconomy`, `isRefundable`, `isNonStop`) |

### 3. Data Cleaning
- **Duplicates:** Dropped exact duplicate rows
- **Nulls in `totalTravelDistance`:** Imputed with median per `(startingAirport, destinationAirport)` pair using a window function
- **Nulls in `segmentsEquipmentDescription`:** Filled with `"Missing"`

### 4. Storage
Cleaned data is saved as Parquet for faster subsequent reads.

### 5. Exploratory Data Analysis
A 1.2% random sample (~1M rows) is pulled into Pandas for visualisation. Charts produced:
- Flights by month
- Busiest US airports
- Top 15 most frequent routes
- Market share by airline
- Average fare by month (seasonal trends)
- Mean travel duration by departure airport
- Fare distribution by seat type (basic economy vs standard)
- Spearman correlation heatmap

### 6. Modelling

Six progressively richer linear regression models are evaluated, all built with scikit-learn Pipelines:

| Model | Features | R² |
|---|---|---|
| 8a — Simple LR | Distance only | ~0.23 |
| 8b — Multiple LR | Distance + duration | ~0.25 |
| 8c — Interaction term | Distance × duration | ~0.25 |
| 8d — Categorical features | Route, airline, month, seat type, lead time | ~0.49 |
| 8e — Log-transformed target | Same as 8d, `log1p(totalFare)` | ~0.60 |
| 8f — Best model | Categorical features + interaction terms on route/month | ~0.63 |

Log-transforming the target variable (`totalFare`) was the single biggest improvement, addressing the heteroscedasticity visible in residual plots.

---

## Key Findings

- **Route and seasonality** are the strongest predictors of fare — far more important than distance or duration alone.
- **Booking lead time** (`daysFrom`) adds meaningful signal when combined with categorical route features.
- **Log-transforming the fare** resolves heteroscedasticity and improves R² from ~0.49 to ~0.60.
- **Linear models hit a ceiling at ~0.63 R²** — non-linear relationships in airline pricing (e.g. last-minute fare spikes) are better captured by tree-based models.

---

## Requirements

### Databricks / Spark
- Databricks Runtime with PySpark
- Access to an Azure Blob Storage account with the itineraries CSV

### Python Libraries
```
pyspark
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## Setup

1. **Configure storage credentials** in the first cell:
   ```python
   storage_account = "your_storage_account"
   access_key = "your_access_key"   # or use dbutils.secrets.get()
   container = "your_container"
   file_name = "itineraries.csv"
   ```
   > ⚠️ Never commit access keys directly. Use `dbutils.secrets` in production.

2. **Run cells in order** — the notebook is structured sequentially (mount → load → engineer → clean → save → EDA → model).

3. **Adjust sample fraction** for EDA and modelling if memory is a constraint:
   ```python
   sample_df = data.sample(fraction=0.012, seed=42).toPandas()
   ```

---

## Next Steps

- **Tree-based models:** XGBoost or LightGBM to capture non-linear pricing dynamics
- **Hyperparameter tuning:** Grid search over polynomial degrees and regularisation (Ridge/Lasso)
- **MLlib at scale:** Replace the scikit-learn sample-based approach with Spark MLlib for full 82M-row training
- **Feature expansion:** Incorporate day of week, time of day, holidays, and competitor pricing signals
