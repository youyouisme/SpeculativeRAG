import pandas as pd
import re
import os
import glob

def process_generated_answer(generated_answer):
    if not isinstance(generated_answer, str):
        return ""
    else:
        cleaned = re.findall(r'\b[A-D]\b|\b[A-D](?=\s|$|\)|\])', generated_answer)
        return cleaned[0] if cleaned else ""

def load_csv_files(model_name, dataset_name):
    base_path = "./results"
    zero_path = os.path.join(base_path, "zero-shot", f"{model_name}_{dataset_name}_zero_shot_k=0.csv")
    rag_path = os.path.join(base_path, "RAG", f"{model_name}_{dataset_name}_RAG_k=5.csv")
    qrag_paths = {
        "7BQRAG": os.path.join(base_path, "QAG/Answer_Generator/7BQAG", f"{model_name}_{dataset_name}_Answer_Generator_k=5.csv"),
        "32BQRAG": os.path.join(base_path, "QAG/Answer_Generator/32BQAG", f"{model_name}_{dataset_name}_Answer_Generator_k=5.csv"),
        "72BQRAG": os.path.join(base_path, "QAG/Answer_Generator/72BQAG", f"{model_name}_{dataset_name}_Answer_Generator_k=5.csv"),
    }
    
    zero_df = pd.read_csv(zero_path) if os.path.exists(zero_path) else None
    rag_df = pd.read_csv(rag_path) if os.path.exists(rag_path) else None
    qrag_dfs = {name: pd.read_csv(path) for name, path in qrag_paths.items() if os.path.exists(path)}
    print(model_name, dataset_name)
    if zero_df is None or rag_df is None or not qrag_dfs:
        print(f"Error: One or more files are missing for {model_name} on {dataset_name}")
        return None, None, {}
    return zero_df, rag_df, qrag_dfs

def classify_results(zero_df, rag_df, qrag_dfs):
    if "real_answer" not in zero_df.columns or "real_answer" not in rag_df.columns:
        raise KeyError("Missing 'real_answer' column in zero_df or rag_df")
    
    merged = zero_df.merge(rag_df, on="id", suffixes=("_zero", "_rag"))
    merged["zero_label"] = merged["generated_answer_zero"].apply(process_generated_answer)
    merged["rag_label"] = merged["generated_answer_rag"].apply(process_generated_answer)
    merged["zero_correct"] = merged["zero_label"] == merged["real_answer_zero"]
    merged["rag_correct"] = merged["rag_label"] == merged["real_answer_rag"]
    
    merged["A"] = merged["zero_correct"]
    merged["B"] = ~merged["zero_correct"]
    merged["A1"] = merged["A"] & merged["rag_correct"]
    merged["A2"] = merged["A"] & ~merged["rag_correct"]
    merged["B1"] = merged["B"] & merged["rag_correct"]
    merged["B2"] = merged["B"] & ~merged["rag_correct"]
    
    for qrag_name, qrag_df in qrag_dfs.items():
        if "real_answer" not in qrag_df.columns:
            raise KeyError(f"Missing 'real_answer' column in {qrag_name}")
            
        qrag_df = qrag_df.rename(columns={
            "generated_answer": f"generated_answer_{qrag_name}",
            "real_answer": "real_answer"
        })
        
        merged = merged.merge(qrag_df, on="id", suffixes=("", f"_{qrag_name}"))
        
        merged[f"qrag_label_{qrag_name}"] = merged[f"generated_answer_{qrag_name}"].apply(process_generated_answer)
        merged[f"qrag_correct_{qrag_name}"] = merged[f"qrag_label_{qrag_name}"] == merged["real_answer"]
        
        merged[f"A11_{qrag_name}"] = merged["A1"] & merged[f"qrag_correct_{qrag_name}"]
        merged[f"A12_{qrag_name}"] = merged["A1"] & ~merged[f"qrag_correct_{qrag_name}"]
        merged[f"A21_{qrag_name}"] = merged["A2"] & merged[f"qrag_correct_{qrag_name}"]
        merged[f"A22_{qrag_name}"] = merged["A2"] & ~merged[f"qrag_correct_{qrag_name}"]
        merged[f"B11_{qrag_name}"] = merged["B1"] & merged[f"qrag_correct_{qrag_name}"]
        merged[f"B12_{qrag_name}"] = merged["B1"] & ~merged[f"qrag_correct_{qrag_name}"]
        merged[f"B21_{qrag_name}"] = merged["B2"] & merged[f"qrag_correct_{qrag_name}"]
        merged[f"B22_{qrag_name}"] = merged["B2"] & ~merged[f"qrag_correct_{qrag_name}"]
    
    return merged

def compute_proportions(df, qrag_models):
    A = df["A"].sum()
    B = df["B"].sum()
    A1 = df["A1"].sum()
    A2 = df["A2"].sum()
    B1 = df["B1"].sum()
    B2 = df["B2"].sum()
    
    results = {
        "zero->RAG": (A2 / (A+B))  if A > 0 else 0,
        "zero->RAG (wrong to correct)": (B1 / (A+B))  if B > 0 else 0,
    }
    
    for qrag in qrag_models:
        A12 = df[f"A12_{qrag}"].sum()
        A22 = df[f"A22_{qrag}"].sum()
        B11 = df[f"B11_{qrag}"].sum()
        B21 = df[f"B21_{qrag}"].sum()
        A21 = df[f"A21_{qrag}"].sum()
        B12 = df[f"B12_{qrag}"].sum()

        results[f"zero->{qrag}"] = ((A12 + A22) / (A+B))  if (A+B) > 0 else 0
        results[f"zero->{qrag} (wrong to correct)"] = ((B11 + B21) / (A+B))  if (A+B) > 0 else 0
        results[f"RAG->{qrag}"] = ((A12 + B12) / (A+B))  if (A+B) > 0 else 0
        results[f"RAG->{qrag} (wrong to correct)"] = ((A21 + B21) / (A+B))  if (A+B) > 0 else 0
    
    return results

def process_all_models():
    datasets = ["medqa", "medmcqa", "pubmedqa", "MMLU_Col_Med", "MMLU_Col_Bio", "MMLU_Pro_Med", "MMLU_Anatomy", "MMLU_Gene", "MMLU_Clinic"]
    models = ["gpt-4o", "gpt-3.5-turbo", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-72B", "LLaMA/llama-3-8B-Instruct", "LLaMA/llama-3-70B-Instruct"]
    dataframes = []
    
    for model in models:
        for dataset in datasets:
            zero_df, rag_df, qrag_dfs = load_csv_files(model, dataset)
            
            if isinstance(zero_df, pd.DataFrame) and isinstance(rag_df, pd.DataFrame) and qrag_dfs:
                merged_df = classify_results(zero_df, rag_df, qrag_dfs)
                proportions = compute_proportions(merged_df, qrag_dfs.keys())
                
                for metric, value in proportions.items():
                    dataframes.append([model, metric, dataset, value])
    
    final_df = pd.DataFrame(dataframes, columns=["Model", "Metric", "Category", "Value"])
    
    # Create directory if it doesn't exist
    base_dir = "./correction_analysis"
    os.makedirs(base_dir, exist_ok=True)
    
    # Split into wrong_to_correct and correct_to_wrong
    wrong_mask = final_df['Metric'].str.contains('(wrong to correct)', regex=False)
    wrong_to_correct_df = final_df[wrong_mask].copy()
    correct_to_wrong_df = final_df[~wrong_mask].copy()
    
    # Save each DataFrame
    wrong_to_correct_path = os.path.join(base_dir, f"wrong_to_correct.csv")
    correct_to_wrong_path = os.path.join(base_dir, f"correct_to_wrong.csv")
    
    wrong_to_correct_df.to_csv(wrong_to_correct_path, index=False)
    correct_to_wrong_df.to_csv(correct_to_wrong_path, index=False)
    
    print(f"Saved wrong_to_correct results to {wrong_to_correct_path}")
    print(f"Saved correct_to_wrong results to {correct_to_wrong_path}")

# Run the processing
process_all_models()