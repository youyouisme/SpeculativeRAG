ZERO_SHOT_PROMPT = {
    "system": "You are a medical expert and are tasked with solving complex medical questions accurately. Only return the answer in the format of answer options like A, B, C, D, etc. Avoid providing any additional explanations or information.",
    "user": "{question}\nReturn your output in the format of answer option."
}

RAG_PROMPT = {
    "system": "You are a medical expert and are tasked with solving complex medical questions accurately. Only return the answer in the format of answer option (A, B, C, D). Avoid providing any additional explanations or information.",
    "user": "Here are some references of answering the question.\n{retrieved_text}\nThe real question is: {question}\nThe references may or may not be relevant to the question. If the references are not relevant, disgard the references and use your knowledge to answer the question. Only return the answer in the format of answer option (A, B, C, D).\nThe answer is"
}

RAG_COT_PROMPT = {
    "system": "You are a medical expert and are tasked with solving complex medical questions accurately. \nPlease think step-by-step to answer the question. The references may or may not be relevant to the question. If the references are not relevant, disgard the references and use your knowledge to answer the question. \nStructure your output in a json format: {\"answer_choice\": \"A/B/C/D/...\", \"step_by_step_thinking\": \"...\"}.",
    "user": "###References:\n{retrieved_text}\n\n###Question: {question}\n\n###Output:"
}

RAT_ANALYSIS_PROMPT = {
    "system": """You are a biomedical expert. Your task is to analyze biomedical questions and extract key information from them. Please follow these steps:
    1. Summarize the contextual background or clinical information if it is provided.
    2. Identify the intent and key concepts of the question.
    3. If specific medical, biological, or scientific principles are involved, outline them.

    Enclose all information within <analysis></analysis>.
    """,
    "user": """
    ## Example 1:
    Question:
    Which of the following conditions does not show multifactorial inheritance?
    Options:
    A) Pyloric stenosis
    B) Schizophrenia
    C) Spina bifida (neural tube defects)
    D) Marfan syndrome
    
    Output:
    <analysis> The intent of the question is to assess the understanding of multifactorial inheritance and to identify which diseases or conditions do not exhibit multifactorial inheritance. The question requires distinguishing between typical multifactorial genetic disorders and other types of genetic disorders, such as those caused by a single gene mutation. </analysis> 
   
    ## Example 2:
    Question:
    A previously healthy 32-year-old woman comes to the physician 8 months after her husband was killed in a car crash. Since that time, she has had a decreased appetite and difficulty falling asleep. She states that she is often sad and cries frequently. She has been rechecking the door lock five times before leaving her house and has to count exactly five pieces of toilet paper before she uses it. She says that she has always been a perfectionist but these urges and rituals are new. Pharmacotherapy should be targeted to which of the following neurotransmitters?
    Options:
    A) Dopamine
    B) Glutamate
    C) Norepinephrine
    D) Serotonin

    Output:
    <analysis> The question involves a female patient who is experiencing a series of psychological symptoms including depressive symptoms (such as persistent sadness and frequent crying) and compulsive behaviors (like repeatedly checking door locks and counting toilet paper) following the traumatic loss of her husband. Understanding the pathophysiological mechanisms behind these symptoms, particularly which neurotransmitters they are associated with, is crucial for selecting an appropriate pharmacotherapy treatment plan. </analysis>
    
    ##Real Data
    {question}
    Output:
    """
}

RAT_QUERY_GENERATION_PROMPT = {
    "system": """You are a biomedical query formulator. Your task is to identify the most pertinent information necessary to effectively address the question. Follow these guidelines carefully:
        - Use the provided question for context only; do not directly answer it. The essential understanding of the question is detailed in the Analysis section.
        - Ensure each generated query is precise, self-contained, and easily understandable without requiring additional background information. Aim for specificity to avoid collecting extraneous data.
        - If the question itself already contains all necessary information, return <query>NA</query>.
        - Generate no more than five queries that are highly supportive and directly relevant to answering the question. If less than five queries are sufficient for covering important information, do not generate more. Avoid generating queries that could broaden the scope unnecessarily or introduce information that does not contribute directly to resolving the question at hand.
        - Enclose each query within <query></query> tags.""",
    "user": """
    ## Example Guidelines
    Below are examples illustrating how to formulate relevant queries based on the provided question and analysis. Each example includes the question, analysis, and the resulting list of queries.

    ##Example 1
    Question:
    Which of the following conditions does not show multifactorial inheritance?
    Options:
    A) Pyloric stenosis
    B) Schizophrenia
    C) Spina bifida (neural tube defects)
    D) Marfan syndrome
    Analysis:
    The intent of the question is to assess the understanding of multifactorial inheritance and to identify which diseases or conditions do not exhibit multifactorial inheritance. The question requires distinguishing between typical multifactorial genetic disorders and other types of genetic disorders, and identifying common multifactorial inheritance diseases. 
    Queries:
    <query>What is the definition and general characteristics of multifactorial inheritance?</query>
    <query>What are the most common multifactorial genetic disorders?</query>

    ##Example 2
    Question:
    A previously healthy 32-year-old woman comes to the physician 8 months after her husband was killed in a car crash. Since that time, she has had a decreased appetite and difficulty falling asleep. She states that she is often sad and cries frequently. She has been rechecking the door lock five times before leaving her house and has to count exactly five pieces of toilet paper before she uses it. She says that she has always been a perfectionist but these urges and rituals are new. Pharmacotherapy should be targeted to which of the following neurotransmitters?
    Options:
    A) Dopamine
    B) Glutamate
    C) Norepinephrine
    D) Serotonin
    Analysis:
    The question involves a female patient who is experiencing a series of psychological symptoms including depressive symptoms (such as persistent sadness and frequent crying) and compulsive behaviors (like repeatedly checking door locks and counting toilet paper) following the traumatic loss of her husband. Understanding the pathophysiological mechanisms behind these symptoms, particularly which neurotransmitters they are associated with so that the patient can receive appropriate pharmacotherapy treatment. 
    Queries:
    <query>What are the primary symptoms of depression?</query>
    <query>What are the primary symptoms of obsessive-compulsive disorder (OCD)?</query>
    <query>Which neurotransmitter is primarily targeted by pharmacotherapy for treating obsessive-compulsive disorder (OCD)?</query>
    <query>Which neurotransmitter is primarily targeted by pharmacotherapy for treating depression?</query>

    ##Example 3
    Question:
    What is the embryological origin of the hyoid bone?
    Options:
    A) The first pharyngeal arch
    B) The first and second pharyngeal arches
    C) The second pharyngeal arch
    D) The second and third pharyngeal arches
    Analysis:
    This question asks about the embryological origin of the hyoid bone, which is a bone located in the neck.
    Queries:
    <query>What is hyoid bone?</query>
    <query>What is the embryological origin of the hyoid bone?</query>


    ##Actual Task
    {question}
    Analysis:
    {analysis}
    Queries:
    """ 
}

RAT_QUERY_REASONING_PROMPT = {
    "system": """You are a biomedical literature analyst. Your task is to analyze the provided documents and extract the most relevant information to answer the given query step by step. 
    Structure your output in a JSON format as follows:{ "answer": "...", "step-by-step_thinking": "1. **Validate References**: ... 2. **Logical Reasoning**: ... 3. **Answer the Question**: ..." }""",
    "user": """
    Follow these steps to provide your answer:
    1. **Validate References**: 
    - For each reference, verify which references contain the necessary information and explain their relevance.
    2. **Logical Reasoning**: 
    - Using only the relevant references identified in step 1, solve the question step by step logically.
    - If the references are not sufficient to solve the question, return {{"answer": "CANNOT_ANSWER", "step-by-step_thinking": "..."}}.
    3. **Answer the Question**: 
    - Provide your answer to the question based on previous steps. Use short sentences to answer the question.

    Documents:
    {retrieved_text}

    Query:
    {query}
    Output:
    """}

RAT_ANSWER_GENERATION_PROMPT = {
    "system": """You are a medical expert tasked with solving complex medical multiple-choice question answering step-by-step. 
    You will be provided with an analysis of the original question, and a series of possibly relevant sub-questions derived from this question followed by potentially correct sub-answers. 
    Use the given analysis and the sub-questions and answers to select the most correct option for the original question.
    Structure your output in a JSON format as follows:{ "answer_choice": "A/B/C/D/...", "step-by-step_thinking": "1. **Validate References**: ... 2. **Logical Reasoning**: ... 3. **Answer the Question**: ..." }""",
    "user": """
    Follow these steps to provide your answer:
    1. **Validate References**: 
    - For each sub-question, verify which references contain the necessary information and explain their relevance.
    2. **Logical Reasoning**: 
    - Using only the relevant references identified in step 1, solve the question step by step logically.
    - If the references are not sufficient to solve the question, rely on your own knowledge.
    3. **Answer the Question**: 
    - Provide the most accurate answer to the original question based on previous steps.

    Analysis:
    {analysis}

    Sub-questions:
    {qa_string}

    Question:
    {question}

    The answer is:"""
}

RAP_PROMPT = {
    "system": """You are a helpful medical expert, and your task is to answer a complex multi-choice medical question. Your responsibilities include analyzing the question step-by-step, leveraging provided references, and reasoning logically before selecting the best answer. Structure your output in a JSON format as follows:{ "answer_choice": "A/B/C/D/...", "step-by-step_thinking": "1. **Plan the Solution**: ... 2. **Identify Information Needs**: ... 3. **Validate References**: ... 4. **Answer the Question**: ..." }""",
    "user": """
    Follow these steps to provide your answer:

    1. **Plan the Solution**: 
    - Analyze the question and determine the steps required to solve it. 
    - Briefly outline each step.

    2. **Identify Information Needs**:
    - Identify the information required for each step.
    
    3. **Validate References**:
    - For each step, verify which references contain the necessary information and explain their relevance.

    4. **Answer the Question**:
    - Using only the relevant references identified in step 3, solve the question step by step.
    - If the references are not sufficient to solve the question, rely on your own knowledge.

    Here are the references for answering the question:  
    {retrieved_text}

    The following is the real question: {question}

    The answer is:"""
}

QAG_SPECULATOR_PROMPT = {
    "system": """You are a helpful medical expert, and your task is to answer a complex multi-choice medical question. 
    Your responsibilities include analyzing the question, leveraging provided references, and reasoning logically to select the best answer. 
    Structure your output in a JSON format as follows:
    { "answer_choice": "A/B/C/D/...", "analysis": "... question understanding and analysis ... ", "sub_questions": ["sub_question1", "sub_question2", ...], "sub_answers": ["answer1", "answer2", ...], "logic_reasoning": "...Step-by-step logical reasoning connecting sub-answers to the final choice.... " }""",
    "user": """
    Follow these steps to provide your answer:

    1. **Analyze the Question**: 
    - Identify the intent of the question and the key medical/biological concepts involved.
    - Summarize any contextual background or clinical information present in the question.
    - Outline relevant medical principles or guidelines that apply to solving the question.

    2. **Identify Information Needs**:
    - Determine what specific information is needed to effectively answer the question.
    - Convert these requirements into a list of sub-questions that help break down the problem.

    3. **Answer the Sub-Questions based on References**:
    - Answer each sub-question using relevant information from the provided references. After generating each answer, cite the corresponding reference number(s) in square brackets to indicate the source(s) of information.

    4. **Logic Reasoning**:
    - Use the sub-questions and their answers to form a clear chain of reasoning.
    - Explain how the answers logically lead to the final answer choice.
    - Justify why the selected answer is the most appropriate, and if applicable, why other choices are incorrect.


    Here are the references for answering the question:  
    {retrieved_text}

    The following is the real question: {question}

    The answer is:"""
}

QAG_ANSWER_PROMPT = {
    "system": """You are a medical expert tasked with solving complex medical multiple-choice question answering step-by-step. 
    You will be provided with a series of relevant sub-questions derived from this question followed by potentially correct sub-answers. 
    Use the sub-questions and answers to select the most correct option for the original question.

    Structure your output in a JSON format as follows:{ "answer_choice": "A/B/C/D", "step-by-step_thinking": "1. **Validate References**: - sub-question 1:... - sub-question 2: ... 2. **Logical Reasoning**: -option A: ... -option B: ... -option C: ... -option D: ...  - option comparison: ... - further reasoning: ... 3. **Answer the Question**: ..." }""",
    "user": """
    Follow these steps to provide your answer:
    1. **Validate References**:
    - For each sub-question, verify whether the sub-question and answer contains the necessary information to answer the question.
    - Explain the relevance of each sub-question and answer in relation to the original question and the options. 

    2. **Logical Reasoning**:
    - Using the relevant sub-questions and answers identified in step 1, logically evaluate and solve the question.
    - Support or decline specific answer choices based on the reasoning derived from these sub-questions.
    - If the references are not sufficient to solve the question, rely on your own knowledge.

    3. **Answer the Question**: 
    - Provide the most accurate answer to the original question based on previous steps.

    Sub-questions:
    {qa_string}

    Question:
    {question}

    The answer is:"""
}
