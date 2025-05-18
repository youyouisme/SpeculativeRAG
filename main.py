import argparse
from QAG.methods import run_method

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["zero_shot", "rag", "rag_cot", "rat", "rap", "qag_speculator", "qag_answer"])
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--question_model", type=str, default="7B")  # only for qag_answer

    args = parser.parse_args()
    run_method(args)