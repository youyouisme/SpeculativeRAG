import json
import re
import pandas as pd
from .data import load_data
from .models import map_model_type, initialize_model, generate_answer
from .retriever import create_vector_store, load_vector_store
from .utils import exact_match, save_results_to_csv, save_accuracy_to_txt, extract_between_tags, safe_extract_json_answer
from .prompt import *

def prepare_pipeline(args):
    faiss = load_vector_store()
    questions, gold_answers, ids = load_data(args.dataset, "test")
    model_type = map_model_type(args.model_name)
    model, tokenizer = initialize_model(model_type, args.model_name)
    return faiss, questions, gold_answers, ids, model_type, model, tokenizer

## QAG METHODS ##
########################################################
### Question Speculator
def run_qag_speculator(args):
    faiss, questions, gold_answers, ids, model_type, model, tokenizer = prepare_pipeline(args)
    results, correct = [], 0
    for q, a, id_ in zip(questions, gold_answers, ids):
        q_clean = re.sub(r"(Question:\s*|\s*Options:\s*|\s*[A-D]\)\s*|\s+)", " ", q).strip()

        retrieved_results = faiss.similarity_search_with_score(q_clean, k=args.k)
        retrieved_text = "".join([f"Reference {i+1}: {doc.page_content}\n" for i, (doc, _) in enumerate(retrieved_results)])
        scores = sum([score for _, score in retrieved_results]) / args.k

        messages = [
            {"role": "system", "content": QAG_SPECULATOR_PROMPT["system"]},
            {"role": "user", "content": QAG_SPECULATOR_PROMPT["user"].format(retrieved_text=retrieved_text, question=q)}
        ]
        gen = generate_answer(messages, model, tokenizer, model_type, 600) 

        results.append({
            "id": id_,
            "retrieved_text": retrieved_text,
            "question": q,
            "real_answer": a,
            "generated_answer": gen,
            "scores": scores
        })

        if exact_match(gen, a): correct += 1

    acc = correct / len(questions)
    save_results_to_csv(args.model_name, results,
        ["id", "retrieved_text", "question", "real_answer", "generated_answer", "scores"],
        "./results/QAG/Question_Speculator", args.dataset, "Question_Speculator", args.k)
    save_accuracy_to_txt(args.model_name, acc, args.dataset, "Question_Speculator", args.k)

### Answer Generator
def extract_list(pattern, text):
    match = re.search(pattern, text, re.DOTALL)
    if match:
        extracted_text = match.group(1)
        items = re.findall(r'"(.*?)"', extracted_text)  # Extract quoted strings
        return items
    return []

def run_qag_answer(args):
    _, questions, gold_answers, ids, model_type, model, tokenizer = prepare_pipeline(args)
    results, correct = [], 0
    for q, a, id_ in zip(questions, gold_answers, ids):
        df = pd.read_csv(f"./results/QAG/Question_Speculator/Qwen2.5-{args.question_model}_{args.dataset}_Question_Speculator_k={args.k}.csv")
        
        row = df[df["id"] == id_]
        if row.empty: continue
        gen = row["generated_answer"].values[0]

        try:
            data = json.loads(gen)
        except json.JSONDecodeError:
            sub_qs = extract_list(r'"sub_questions"\s*:\s*(\[[^\]]*\])', gen)
            sub_as = extract_list(r'"sub_answers"\s*:\s*(\[[^\]]*\])', gen)
            min_len = min(len(sub_qs), len(sub_as))
            data = {
                "sub_questions": sub_qs[:min_len],
                "sub_answers": sub_as[:min_len]
            }

        qa_str = ""
        for i, (subq, suba) in enumerate(zip(data["sub_questions"], data["sub_answers"])):
            qa_str += f"sub_question {i+1}: {subq}\nAnswer: {suba}\n\n"

        messages = [
            {"role": "system", "content": QAG_ANSWER_PROMPT["system"]},
            {"role": "user", "content": QAG_ANSWER_PROMPT["user"].format(qa_string=qa_str, question=q)}
        ]
        print(messages)
        gen_answer = generate_answer(messages, model, tokenizer, model_type, 600) 

        results.append({
            "id": id_,
            "sub_qa": qa_str,
            "question": q,
            "real_answer": a,
            "generated_answer": gen_answer
        })

        if exact_match(gen_answer, a): correct += 1

    acc = correct / len(questions)
    save_results_to_csv(args.model_name, results,
        ["id", "sub_qa", "question", "real_answer", "generated_answer"],
        f"./results/QAG/Answer_Generator/{args.question_model}QA", args.dataset, "Answer_Generator", args.k)
    save_accuracy_to_txt(args.model_name, acc, args.dataset, "Answer_Generator", args.k, args.question_model)


## BASELINE METHODS##
########################################################
### Zero-shot baseline
def run_zero_shot(args):
    if args.k != 0:
        print("Zero-shot baseline does not support k > 0")
        return
    _, questions, gold_answers, ids, model_type, model, tokenizer = prepare_pipeline(args)
    results = []
    correct = 0
    for q, a, id_ in zip(questions, gold_answers, ids):
        messages = [
            {"role": "system", "content": ZERO_SHOT_PROMPT["system"]},
            {"role": "user", "content": ZERO_SHOT_PROMPT["user"].format(question=q)}
        ]
        gen = generate_answer(messages, model, tokenizer, model_type, 20)
        results.append({"id": id_, "question": q, "real_answer": a, "generated_answer": gen})
        if exact_match(gen, a): correct += 1

    acc = correct / len(questions)
    save_results_to_csv(args.model_name, results, ["id", "question", "real_answer", "generated_answer"], "./results/zero-shot", args.dataset, "zero_shot", 0)
    save_accuracy_to_txt(args.model_name, acc, args.dataset, "zero_shot", 0)


### RAG baseline
def run_rag(args):
    faiss, questions, gold_answers, ids, model_type, model, tokenizer = prepare_pipeline(args)
    results, correct = [], 0
    for q, a, id_ in zip(questions, gold_answers, ids):
        # Clean question
        q_clean = re.sub(r"(Question:\s*|\s*Options:\s*|\s*[A-D]\)\s*|\s+)", " ", q).strip()
        retrieved = faiss.similarity_search_with_score(q_clean, k=args.k)
        references = "".join([f"Reference {i+1}: {doc.page_content}\n" for i, (doc, _) in enumerate(retrieved)])
        scores = sum([score for _, score in retrieved]) / args.k
        user = RAG_PROMPT["user"].format(retrieved_text=references, question=q)
        messages = [
            {"role": "system", "content": RAG_PROMPT["system"]},
            {"role": "user", "content": user}
        ]
        gen = generate_answer(messages, model, tokenizer, model_type, 20)
        results.append({"id": id_, "retrieved_text": references, "question": q, "real_answer": a, "generated_answer": gen, "scores": scores})
        if exact_match(gen, a): correct += 1

    acc = correct / len(questions)
    save_results_to_csv(args.model_name, results, ["id", "retrieved_text", "question", "real_answer", "generated_answer", "scores"], "./results/RAG", args.dataset, "RAG", args.k)
    save_accuracy_to_txt(args.model_name, acc, args.dataset, "RAG", args.k)


### RAG + Chain of Thought
def run_rag_cot(args):
    faiss, questions, gold_answers, ids, model_type, model, tokenizer = prepare_pipeline(args)
    results, correct = [], 0
    for q, a, id_ in zip(questions, gold_answers, ids):
        # Clean question
        q_clean = re.sub(r"(Question:\s*|\s*Options:\s*|\s*[A-D]\)\s*|\s+)", " ", q).strip()
        retrieved = faiss.similarity_search_with_score(q_clean, k=args.k)
        references = "".join([f"Reference {i+1}: {doc.page_content}\n" for i, (doc, _) in enumerate(retrieved)])
        scores = sum([score for _, score in retrieved]) / args.k
        user = RAG_COT_PROMPT["user"].format(retrieved_text=references, question=q)
        messages = [
            {"role": "system", "content": RAG_COT_PROMPT["system"]},
            {"role": "user", "content": user}
        ]
        gen = generate_answer(messages, model, tokenizer, model_type, 500)

        results.append({"id": id_, "retrieved_text": references, "question": q, "real_answer": a, "generated_answer": gen, "scores": scores})
        if exact_match(gen, a): correct += 1

    acc = correct / len(questions)
    save_results_to_csv(args.model_name, results, ["id", "retrieved_text", "question", "real_answer", "generated_answer", "scores"], "./results/RAG_CoT", args.dataset, "RAG_CoT", args.k)
    save_accuracy_to_txt(args.model_name, acc, args.dataset, "RAG_CoT", args.k)

### RAT
def query_and_answer_subquestions(faiss, model, tokenizer, model_type, queries, args):
    query_answer_string, retrieved_text = "", ""
    cannot_answer_count = 0
    for query in queries:
        if query == "NA":
            continue
        retrieved_results = faiss.similarity_search_with_score(query, k=args.k)
        text = "".join([f"Reference {i+1}: {doc.page_content}\n" for i, (doc, _) in enumerate(retrieved_results)])
        messages = [
            {"role": "system", "content": RAT_QUERY_REASONING_PROMPT["system"]},
            {"role": "user", "content": RAT_QUERY_REASONING_PROMPT["user"].format(retrieved_text=text, query=query)}
        ]
        qa = generate_answer(messages, model, tokenizer, model_type, 300)
        answer_content = safe_extract_json_answer(qa)
        if "CANNOT_ANSWER" in answer_content.upper():
            cannot_answer_count += 1
        else:
            query_answer_string += f"sub-question: {query} Answer: {answer_content}\n"
            retrieved_text += text
    return query_answer_string, retrieved_text, cannot_answer_count

def run_rat(args):
    faiss, questions, gold_answers, ids, model_type, model, tokenizer = prepare_pipeline(args)
    results, correct = [], 0
    for q, a, id_ in zip(questions, gold_answers, ids):
        # Step 1: Generate analysis
        messages = [
            {"role": "system", "content": RAT_ANALYSIS_PROMPT["system"]},
            {"role": "user", "content": RAT_ANALYSIS_PROMPT["user"].format(question=q)}
        ]
        analysis_output = generate_answer(messages, model, tokenizer, model_type, 200)
        analysis_list = extract_between_tags(analysis_output, "analysis")
        analysis = analysis_list[0] if analysis_list else ""

        # Step 2: Generate queries
        messages = [
            {"role": "system", "content": RAT_QUERY_GENERATION_PROMPT["system"]},
            {"role": "user", "content": RAT_QUERY_GENERATION_PROMPT["user"].format(analysis=analysis, question=q)}
        ]
        query_output = generate_answer(messages, model, tokenizer, model_type, 300)
        queries = extract_between_tags(query_output, "query")


        # Step 3: Answer sub-questions
        query_answer_string, retrieved_text, cannot_answer_count = query_and_answer_subquestions(
            faiss, model, tokenizer, model_type, queries, args
        )

        # Step 4: Final answer
        messages = [
            {"role": "system", "content": RAT_ANSWER_GENERATION_PROMPT["system"]},
            {"role": "user", "content": RAT_ANSWER_GENERATION_PROMPT["user"].format(
                question=q, analysis=analysis, qa_string=query_answer_string)}
        ]
        final_answer = generate_answer(messages, model, tokenizer, model_type, 500)

        results.append({
            "id": id_, "retrieved_text": retrieved_text, "analysis": analysis,
            "query_answer_string": query_answer_string, "question": q, "real_answer": a,
            "generated_answer": final_answer, "query_count": len(queries),
            "cannot_answer_count": cannot_answer_count, "cannot_answer_list": [],
        })
        if exact_match(final_answer, a): correct += 1

    acc = correct / len(questions)
    save_results_to_csv(
        args.model_name, results,
        ["id", "retrieved_text", "analysis", "query_answer_string", "question", "real_answer", "generated_answer",
         "query_count", "cannot_answer_count", "cannot_answer_list"],
        "./results/RAT", args.dataset, "RAT", args.k
    )
    save_accuracy_to_txt(args.model_name, acc, args.dataset, "RAT", args.k)


### RAP
def run_rap(args):
    faiss, questions, gold_answers, ids, model_type, model, tokenizer = prepare_pipeline(args)
    results, correct, total_time = [], 0, 0
    for q, a, id_ in zip(questions, gold_answers, ids):
        # Clean question
        q_clean = re.sub(r"(Question:\s*|\s*Options:\s*|\s*[A-D]\)\s*|\s+)", " ", q).strip()
        retrieved_results = faiss.similarity_search_with_score(q_clean, k=args.k)
        retrieved_text = "".join([f"Reference {i+1}: {doc.page_content}\n" for i, (doc, _) in enumerate(retrieved_results)])
        scores = sum([score for _, score in retrieved_results]) / args.k
        messages = [
            {"role": "system", "content": RAP_PROMPT["system"]},
            {"role": "user", "content": RAP_PROMPT["user"].format(retrieved_text=retrieved_text, question=q)}
        ]
        print(messages)
        gen = generate_answer(messages, model, tokenizer, model_type, 500) 

        results.append({
            "id": id_, "retrieved_text": retrieved_text, "question": q,
            "real_answer": a, "generated_answer": gen, "scores": scores
        })
        if exact_match(gen, a): correct += 1

    acc = correct / len(questions)
    save_results_to_csv(args.model_name, results,
        ["id", "retrieved_text", "question", "real_answer", "generated_answer", "scores"],
        "./results/RAP", args.dataset, "RAP", args.k)
    save_accuracy_to_txt(args.model_name, acc, args.dataset, "RAP", args.k)


def run_method(args):
    method_map = {
        "zero_shot": run_zero_shot,
        "rag": run_rag,
        "rag_cot": run_rag_cot,
        "rat": run_rat,
        "rap": run_rap,
        "qag_speculator": run_qag_speculator,
        "qag_answer": run_qag_answer,
    }
    method_map[args.method](args)