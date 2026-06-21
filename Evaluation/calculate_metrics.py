import pandas as pd
import numpy as np

# --- Configuration ---
INPUT_FILE = "ragas_eval_results_v3.csv"

def calculate_final_metrics():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"❌ Error: Could not find the file '{INPUT_FILE}'. Please ensure it is in the same directory as this script.")
        return
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    # --- Data Validation ---
    print("Validating metrics...")
    
    # Check for any remaining NaN values in the key metric columns
    key_metrics = ['faithfulness', 'answer_relevancy', 'context_precision']
    
    for metric in key_metrics:
        if metric not in df.columns:
             print(f"❌ Error: Column '{metric}' not found in the CSV.")
             return
             
        missing_count = df[metric].isna().sum()
        if missing_count > 0:
            print(f"⚠️ Warning: Found {missing_count} missing values in the '{metric}' column.")
            print(f"   These rows will be dropped from the {metric} average calculation.")
            
    # --- Calculation & Output ---
    print("\n" + "="*40)
    print("🟢 FINAL PIPELINE METRICS 🟢")
    print("="*40)
    
    # Calculate the mean, dropping NaNs automatically to keep the average pure
    final_faithfulness = df['faithfulness'].mean()
    final_relevancy = df['answer_relevancy'].mean()
    final_precision = df['context_precision'].mean()

    print(f"Total Questions Evaluated: {len(df)}")
    print("-" * 40)
    print(f"Faithfulness:      {final_faithfulness:.4f}")
    print(f"Answer Relevancy:  {final_relevancy:.4f}")
    print(f"Context Precision: {final_precision:.4f}")
    print("="*40)
    
    #distribution check
    print("\n--- Metric Distribution Summary ---")
    print(df[key_metrics].describe().loc[['min', '25%', '50%', '75%', 'max']])

if __name__ == "__main__":
    calculate_final_metrics()