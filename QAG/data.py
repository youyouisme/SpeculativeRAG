import re
import math
import random
from functools import reduce
import pandas as pd
from datasets import load_dataset

## Test Dataset Loading
def load_data(dataset_name: str, dataset_split: str):
    questions = []
    answers = []
    ids = []
    if dataset_name == 'medqa':
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split=dataset_split)

        def query_format(entry, with_answer=False):
            question = entry['question']
            options = entry['options']
            options_str = '\n'.join([f"{key}) {value}" for key, value in options.items()])
            q = f"Question: {question}\nOptions:\n{options_str}"
            return q

        for idx, entry in enumerate(dataset):
            q = query_format(entry)
            a = entry['answer_idx']
            id = 'temp_{}'.format(idx)

            questions.append(q)
            answers.append(a)
            ids.append(id)
    
    elif dataset_name == 'medmcqa':
        if dataset_split == 'test':
            data_split = 'validation'
        else:
            data_split = 'train'
        dataset = load_dataset("openlifescienceai/medmcqa", split= data_split)
        for idx, entry in enumerate(dataset):
            question = entry['question']
            options = {
                'A': entry['opa'],
                'B': entry['opb'],
                'C': entry['opc'],
                'D': entry['opd']
            }
            options_str = '\n'.join([f"{key}) {value}" for key, value in options.items()])
            q = f"Question: {question}\nOptions:\n{options_str}"
            
            cop_to_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            a = cop_to_label[entry['cop']]

            id = entry['id']

            questions.append(q)
            answers.append(a)
            ids.append(id)

    elif dataset_name.lower() == 'pubmedqa':
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split='train')
        # Convert dataset to a dictionary format for splitting
        dataset_dict = {
            entry['pubid']: {
                'question': entry['question'], 
                'context': entry['context']['contexts'], 
                'final_decision': entry['final_decision']
            } for entry in dataset
        }
        # Split the dataset for 500 test and 500 CV samples
        CV_set, testset = split(dataset_dict, 2)  # Split into 2 parts: CV and test
        if dataset_split == 'test':
            for pubid, entry in testset.items():
                question = entry['question']
                context = '\n'.join(entry['context'])
                q = f"Context:\n{context}\nQuestion:\n{question}\nChoices:\n A) Yes\n B) No\n C) Maybe"
                decision_map = {'yes': 'A', 'no': 'B', 'maybe': 'C'}
                a = decision_map[entry['final_decision']]
                id = f'pubmedqa_{pubid}'
                questions.append(q)
                answers.append(a)
                ids.append(id)
        elif dataset_split == 'train':
            for pubid, entry in CV_set.items():
                question = entry['question']
                context = '\n'.join(entry['context'])
                q = f"Context:\n{context}\nQuestion:\n{question}\nChoices:\n A) Yes\n B) No\n C) Maybe"
                decision_map = {'yes': 'A', 'no': 'B', 'maybe': 'C'}
                a = decision_map[entry['final_decision']]
                id = f'pubmedqa_{pubid}'
                questions.append(q)
                answers.append(a)
                ids.append(id)

    elif dataset_name.lower() == 'mmlu_col_med':
        dataset = load_dataset("cais/mmlu", "college_medicine",split=dataset_split)
        for idx, entry in enumerate(dataset):
            question = entry['question']
            options = entry['choices']
            options_str = '\n'.join([f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)])
            q = f"Question: {question}\nOptions:\n{options_str}"
            answer_to_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            a = answer_to_label[entry['answer']]
            id = f'college_med_{idx}'
            questions.append(q)
            answers.append(a)
            ids.append(id)

    elif dataset_name.lower() == 'mmlu_col_bio':
        dataset = load_dataset("cais/mmlu", "college_biology", split=dataset_split)
        for idx, entry in enumerate(dataset):
            question = entry['question']
            options = entry['choices']
            options_str = '\n'.join([f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)])
            q = f"Question: {question}\nOptions:\n{options_str}"
            answer_to_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            a = answer_to_label[entry['answer']]
            id = f'college_bio_{idx}'
            questions.append(q)
            answers.append(a)
            ids.append(id)

    elif dataset_name.lower() == 'mmlu_pro_med':
        dataset = load_dataset("cais/mmlu", "professional_medicine", split=dataset_split)
        for idx, entry in enumerate(dataset):
            question = entry['question']
            options = entry['choices']
            options_str = '\n'.join([f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)])
            q = f"Question: {question}\nOptions:\n{options_str}"
            answer_to_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            a = answer_to_label[entry['answer']]
            id = f'pro_med_{idx}'
            questions.append(q)
            answers.append(a)
            ids.append(id)

    elif dataset_name.lower() == 'mmlu_anatomy':
        dataset = load_dataset("cais/mmlu", "anatomy", split=dataset_split)
        for idx, entry in enumerate(dataset):
            question = entry['question']
            options = entry['choices']
            options_str = '\n'.join([f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)])
            q = f"Question: {question}\nOptions:\n{options_str}"
            answer_to_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            a = answer_to_label[entry['answer']]
            id = f'anatomy_{idx}'
            questions.append(q)
            answers.append(a)
            ids.append(id)


    elif dataset_name.lower() == 'mmlu_gene':
        dataset = load_dataset("cais/mmlu", "medical_genetics", split=dataset_split)
        for idx, entry in enumerate(dataset):
            question = entry['question']
            options = entry['choices']
            options_str = '\n'.join([f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)])
            q = f"Question: {question}\nOptions:\n{options_str}"
            answer_to_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            a = answer_to_label[entry['answer']]
            id = f'gene_{idx}'
            questions.append(q)
            answers.append(a)
            ids.append(id)
    
    
    elif dataset_name.lower() == 'mmlu_clinic':
        dataset = load_dataset("cais/mmlu", "clinical_knowledge", split=dataset_split)
        for idx, entry in enumerate(dataset):
            question = entry['question']
            options = entry['choices']
            options_str = '\n'.join([f"{chr(65 + i)}) {opt}" for i, opt in enumerate(options)])
            q = f"Question: {question}\nOptions:\n{options_str}"
            answer_to_label = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            a = answer_to_label[entry['answer']]
            id = f'clinic_{idx}'
            questions.append(q)
            answers.append(a)
            ids.append(id)

    elif dataset_name.lower() == 'medmcqa_exp':
        dataset = pd.read_csv('./data/medmcqa_exp.csv')
        for idx, row in dataset.iterrows():
            question = row['question']
            answer = row['answer']
            explanation = row['explanation']
            id = row['id']
            questions.append(question)
            answers.append(answer)
            ids.append(id)
    
    elif dataset_name.lower() == 'hle':
        dataset = load_dataset("cais/hle", split=dataset_split)
        for entry in dataset:
        # Filter for Biology/Medicine with no image and multiple-choice answer type
            if (
                entry.get('category') == 'Biology/Medicine'
                and not entry.get('image')
                    and entry.get("answer_type") == "multipleChoice"
                ):
                    question = entry['question']
                    q = f"Question: {question}"
                    a = entry['answer']
                    id = entry['id']
                    questions.append(q)
                    answers.append(a)
                    ids.append(id)
        print(f"Total number of questions: {len(questions)}")
    return questions[0:5], answers[0:5], ids[0:5]

## Dataset splitting following the PubMedQA dataset processing
def split(dataset, fold, seed=42):
    # Set random seed for reproducibility
    random.seed(seed)
    
    add = lambda x: reduce(lambda a, b: a+b, x)
    
    label2pmid = {'yes': [], 'no': [], 'maybe': []}
    for pmid, info in dataset.items():
        label2pmid[info['final_decision']].append(pmid)

    label2pmid = {k: split_label(v, fold) for k, v in label2pmid.items()} # splited

    output = []

    for i in range(fold):
        pmids = add([v[i] for _, v in label2pmid.items()])
        output.append({pmid: dataset[pmid] for pmid in pmids})

    if len(output[-1]) != len(output[0]): # imbalanced: [51, 51, 51, 51, 51, 51, 51, 51, 51, 41]
        # randomly pick one from each to the last
        for i in range(fold-1):
            pmids = list(output[i])
            picked = random.choice(pmids)
            output[-1][picked] = output[i][picked]
            output[i].pop(picked)

    return output

def split_label(pmids, fold, seed=42):
    # Set random seed for reproducibility 
    random.seed(seed)
    
    random.shuffle(pmids)

    num_all = len(pmids)
    num_split = math.ceil(num_all / fold)

    output = []
    for i in range(fold):
        if i == fold - 1:
            output.append(pmids[i*num_split: ])
        else:
            output.append(pmids[i*num_split: (i+1)*num_split])

    return output
