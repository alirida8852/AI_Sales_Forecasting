# AI Sales Forecasting and Pipeline Analytics

ML-first foundation for deal win prediction and revenue forecasting. The initial milestone loads your CRM opportunity tables, trains a win-probability model, and saves evaluation artifacts for the future FastAPI service and React dashboard.

## Project structure

```text
src/sales_forecasting/data/raw/CRM+Sales+Opportunities/
						   CRM CSV tables
models/                    Trained model artifacts
src/sales_forecasting/     ML pipeline
scripts/                   Developer commands
```

## Setup

Requires Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

The project expects these CRM tables under `src/sales_forecasting/data/raw/CRM+Sales+Opportunities/`:

```text
sales_pipeline.csv        Opportunities and deal outcomes
accounts.csv               Account enrichment
products.csv               Product prices and series
sales_teams.csv            Sales-agent enrichment
data_dictionary.csv        Dataset reference
```

Train the baseline model:

```powershell
python scripts\train_model.py
```

Start the API after training:

```powershell
uvicorn sales_forecasting.api:app --reload
```

Initial endpoints:

```text
GET  /health
GET  /api/v1/pipeline/summary
POST /api/v1/predictions/win-probability
```

Run the tests:

```powershell
pytest
```

The model trains on closed opportunities (`Won` and `Lost`). Open opportunities (`Engaging` and `Prospecting`) remain available for future pipeline scoring and forecasting work.

## Design direction

The future dashboard will take visual cues from SalesCentri: dark command-center surfaces, high-contrast typography, vivid lime actions, compact KPI blocks, pipeline-stage views, and clear automation states. The implementation will remain an original interface rather than copying the reference site.