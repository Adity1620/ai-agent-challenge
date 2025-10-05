import pandas as pd
import pdfplumber
import re

def parse(pdf_path: str) -> pd.DataFrame:
    all_transactions = []
    columns = ["Date", "Description", "Debit Amt", "Credit Amt", "Balance"]

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()

            for table in tables:
                header_row_idx = -1
                for i, row_data in enumerate(table):
                    if row_data:
                        # Clean cells in the current row_data list for header detection
                        cleaned_cells_in_row = [c.strip() if c else "" for c in row_data]
                        # Check if all required column names are present in this row's cells
                        if all(col_name in cleaned_cells_in_row for col_name in columns):
                            header_row_idx = i
                            break
                
                if header_row_idx == -1:
                    continue # No valid table header found, skip this table

                # Get column indices based on the identified header
                header = [cell.strip() if cell else "" for cell in table[header_row_idx]]
                
                # If header is malformed after strip, skip it. This is a robust check.
                if not all(col in header for col in columns):
                    continue 

                col_indices = {col: header.index(col) for col in columns}

                # Helper to safely get and convert numeric values
                def safe_float_convert(row_data_list, col_index):
                    if col_index < len(row_data_list):
                        val = row_data_list[col_index]
                        if val and str(val).strip() != '':
                            try:
                                return float(str(val).strip())
                            except ValueError:
                                return None # Return None for values that can't be converted
                    return None # Return None if index out of bounds or value is empty/None

                # Process data rows (skip header and any rows before it)
                for row_idx in range(header_row_idx + 1, len(table)):
                    row_data = table[row_idx]
                    
                    # Basic validation: ensure row has enough cells and starts with a date pattern
                    if not row_data or col_indices["Date"] >= len(row_data):
                        continue

                    date_cell = row_data[col_indices["Date"]]
                    if not date_cell or not re.match(r"\d{2}-\d{2}-\d{4}", date_cell.strip()):
                        continue # Skip non-data rows (e.g., footers, page numbers, empty rows)
                    
                    transaction = {}
                    transaction["Date"] = date_cell.strip()
                    
                    # Description can be multi-word and contains spaces, so take it as is
                    transaction["Description"] = (row_data[col_indices["Description"]].strip() 
                                             if col_indices["Description"] < len(row_data) and row_data[col_indices["Description"]] 
                                             else None)

                    transaction["Debit Amt"] = safe_float_convert(row_data, col_indices["Debit Amt"])
                    transaction["Credit Amt"] = safe_float_convert(row_data, col_indices["Credit Amt"])
                    transaction["Balance"] = safe_float_convert(row_data, col_indices["Balance"])
                    
                    all_transactions.append(transaction)

    df = pd.DataFrame(all_transactions)

    # Convert data types and ensure correct types for an empty DataFrame
    if not df.empty:
        # Keep 'Date' as string (object dtype) as per target CSV
        # df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y') # Removed this conversion
        df['Debit Amt'] = pd.to_numeric(df['Debit Amt'], errors='coerce')
        df['Credit Amt'] = pd.to_numeric(df['Credit Amt'], errors='coerce')
        df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
    else:
        # Create an empty DataFrame with correct column names and dtypes
        df = pd.DataFrame(columns=columns)
        df['Date'] = df['Date'].astype(str)
        df['Description'] = df['Description'].astype(str)
        df['Debit Amt'] = pd.Series(dtype='float64')
        df['Credit Amt'] = pd.Series(dtype='float64')
        df['Balance'] = pd.Series(dtype='float64')

    # Ensure column order matches target CSV
    df = df[columns]

    return df