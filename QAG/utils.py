import os
import re
import csv
import math
import json
import random
import pandas as pd
from datetime import datetime
from functools import reduce
from datasets import load_dataset

def exact_match(predicted_answer, correct_answer):
    cleaned_pred = clean_answer(predicted_answer)
    return cleaned_pred == correct_answer

def clean_answer(answer):
    if not isinstance(answer, str):
        return ""
    cleaned = re.findall(r'\b[A-D]\b|(?<=\s)[A-D](?=\s|$|\)|\])', answer)
    return cleaned[0] if cleaned else ""

def save_results_to_csv(model_name, results, field_name, folder_path, dataset, task, k):
    os.makedirs(folder_path, exist_ok=True)
    sanitized_model = model_name.split("/")[-1]
    output_file = os.path.join(folder_path, f"{sanitized_model}_{dataset}_{task}_k={k}.csv")
    with open(output_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_name)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Results saved to {output_file}")

def save_accuracy_to_txt(model, accuracy, dataset, task, k, question_model=None):
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = "./result.txt"
    with open(output_path, mode="a") as f:
        if question_model:
            f.write(f"Model: {model}, Dataset: {dataset}, prompt: {task}, k={k}, {question_model}QA, time: {time}, Accuracy: {accuracy * 100:.2f}%\n\n")
        else:
            f.write(f"Model: {model}, Dataset: {dataset}, prompt: {task}, k={k}, time: {time}, Accuracy: {accuracy * 100:.2f}%\n\n")
    print(f"Accuracy saved to {output_path}")

def extract_between_tags(text, tag):
    pattern = fr"<{tag}>(.*?)</{tag}>"
    return [match.strip() for match in re.findall(pattern, text, re.DOTALL)]

def safe_extract_json_answer(text):
    try:
        data = json.loads(text)
        return data.get("answer", "CANNOT_ANSWER")
    except:
        match = re.search(r'"answer":\s*"([^"]*)"', text)
        return match.group(1) if match else "CANNOT_ANSWER"
