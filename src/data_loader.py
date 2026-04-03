import pandas as pd

def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # Strip any hidden spaces from column names
        df.columns = df.columns.str.strip()
        
        # Explicit mapping based on your raw dataset structure
        rename_map = {
            'Age': 'Age',
            'Annual Income (k$)': 'AnnualIncome',
            'Spending Score (1-100)': 'SpendingScore'
        }
        
        df = df.rename(columns=rename_map)
        
        # Check if the renaming actually worked
        print(f"📊 Columns found in dataset: {df.columns.tolist()}")
        
        # Validate that our required columns exist
        required = ['Age', 'AnnualIncome', 'SpendingScore']
        for col in required:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found. Check your CSV header!")

        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None