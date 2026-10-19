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

Start the dashboard in a second terminal:

```powershell
npm --prefix frontend install
npm --prefix frontend run dev
```

Open the dashboard at `http://localhost:5173/`. It reads the pipeline summary through the FastAPI proxy and falls back to a clearly labelled demo state if the API is offline.

Initial endpoints:

```text
GET  /health
GET  /api/v1/pipeline/summary
GET  /api/v1/pipeline/opportunities
POST /api/v1/predictions/win-probability
GET  /api/v1/forecast/revenue
```

The revenue forecast combines recent monthly won revenue with the probability-weighted value of open deals. In the current CRM export, open deals do not contain `close_value`, so the open-pipeline contribution is zero until deal values are supplied for those stages.

## Sales intelligence

The dashboard includes a grounded sales-intelligence panel. It answers pipeline questions from the imported SQLite sales database and searches approved company knowledge documents through SQLite FTS5. It does not invent answers when no supporting data is available.

Import an approved Markdown or text document:

```powershell
python scripts\ingest_document.py path\to\sales_playbook.md --type playbook
```

The importer accepts `.md`, `.markdown`, and `.txt` files up to 2 MB. Documents are chunked, stored in the ignored local SQLite database, and returned with source excerpts.

The optional OpenAI refinement layer is disabled unless configured through environment variables:

```powershell
$env:SALES_LLM_API_KEY = "your-key-in-the-shell-only"
$env:SALES_LLM_MODEL = "gpt-4o-mini"
$env:SALES_LLM_BASE_URL = "https://api.openai.com/v1/chat/completions"
```

The model receives only the verified analytics result and retrieved document evidence. If the key is absent or the provider fails, the deterministic grounded answer remains available.

The intelligence endpoint is:

```text
POST /api/v1/intelligence/ask
{"question": "How many open opportunities do we have?"}
```

Run the tests:

```powershell
pytest
```

The model trains on closed opportunities (`Won` and `Lost`). Open opportunities (`Engaging` and `Prospecting`) remain available for future pipeline scoring and forecasting work.

## Design direction

The future dashboard will take visual cues from SalesCentri: dark command-center surfaces, high-contrast typography, vivid lime actions, compact KPI blocks, pipeline-stage views, and clear automation states. The implementation will remain an original interface rather than copying the reference site.