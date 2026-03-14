from datasets import load_dataset
import json

dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train[:200]")

records = []
for row in dataset:
    print(f"Processing pubid: {row['pubid']}")
    records.append({
        "pubid": row["pubid"],
        "question": row["question"],
        "long_answer": row["long_answer"],
        "final_decision": row["final_decision"]
    })

with open("pubmedqa_sample.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("Saved pubmedqa_sample.json")
