import pandas as pd

# Load your final merged dataset
INPUT_FILE = "ragas_eval_results_v3.csv" # or whatever your latest file is named

def verify_genuine_perfect_scores():
    df = pd.read_csv(INPUT_FILE)

    # 1. Isolate rows with a perfect 1.0 Faithfulness score
    perfect_faithfulness = df[df['faithfulness'] == 1.0]

    # 2. FILTER OUT the Guardrail Refusals
    # We know refusals dropped Context Precision to NaN, so we only keep rows with real Context Precision scores
    genuine_perfects = perfect_faithfulness[perfect_faithfulness['context_precision'].notna()]

    print(f"📊 Total genuine 1.0 Faithfulness scores (excluding guardrails): {len(genuine_perfects)}")
    print("🎲 Pulling 3 random samples for your manual review...\n")

    # 3. Randomly sample 3 rows (or however many you want)
    sample_size = min(3, len(genuine_perfects))
    samples = genuine_perfects.sample(n=sample_size)

    for idx, row in samples.iterrows():
        print("="*80)
        print(f"🔍 ROW INDEX: {idx}")
        print("="*80)
        print(f"❓ QUESTION:\n{row['user_input']}\n")
        print(f"📄 RETRIEVED CONTEXTS:\n{row['retrieved_contexts']}\n")
        print(f"🤖 GENERATED ANSWER:\n{row['response']}\n")
        
        # Pause so you can read it before printing the next one
        input("Press Enter to view the next sample...")

if __name__ == "__main__":
    verify_genuine_perfect_scores()