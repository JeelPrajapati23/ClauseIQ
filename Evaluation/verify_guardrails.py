import pandas as pd

# Load your final merged dataset
INPUT_FILE = "ragas_eval_results_v3.csv" 

def verify_guardrail_refusals():
    print(f"📂 Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_FILE} not found.")
        return

    # Filter for rows where answer_relevancy or context_precision is NaN
    # These represent the 4 rows where the guardrail kicked in
    guardrail_rows = df[df['answer_relevancy'].isna()]

    if len(guardrail_rows) == 0:
        print("✅ No guardrail rows found.")
        return

    print(f"🛡️ Total Guardrail Refusal rows found: {len(guardrail_rows)}")
    print("Displaying the rows for manual verification...\n")

    for idx, row in guardrail_rows.iterrows():
        print("="*80)
        print(f"🔍 ROW INDEX: {idx}")
        print("="*80)
        print(f"❓ QUESTION:\n{row['user_input']}\n")
        print(f"📄 RETRIEVED CONTEXTS:\n{row['retrieved_contexts']}\n")
        print(f"🤖 GENERATED ANSWER:\n{row['response']}\n")
        
        # If you have an evaluation_error column tracking the refusal logic, print it
        if 'evaluation_error' in df.columns:
            print(f"⚠️ EVALUATION LOG:\n{row['evaluation_error']}\n")
        
        input("Press Enter to view the next refusal...")

    print("\n🎉 Verification complete!")

if __name__ == "__main__":
    verify_guardrail_refusals()