"""
Avaliação offline.
Formato JSONL:
{"question":"...", "answer":"...", "expected_sources":["Contrato.pdf"]}
"""
import json,sys
from services.evaluation import evaluate_answer

def main(path):
    rows=[json.loads(x) for x in open(path,encoding="utf-8") if x.strip()]
    print("Registros:",len(rows))
    # Use este script como ponto de integração para dataset/LLM-as-judge.
    for r in rows:
        print(r["question"])

if __name__=="__main__":
    main(sys.argv[1])
