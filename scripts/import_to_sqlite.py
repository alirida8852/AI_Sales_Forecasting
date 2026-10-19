"""Import the current CRM CSV snapshot into the local SQLite database."""

from sales_forecasting.data.database import import_crm_data


if __name__ == "__main__":
    print(f"Imported {import_crm_data()} opportunities into SQLite.")