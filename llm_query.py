import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

llm_dir = "./Qwen3-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(llm_dir)
model = AutoModelForCausalLM.from_pretrained(
    llm_dir,
    torch_dtype="auto",
    device_map="auto"
)

def create_chat_completion(messages, temperature, thinking, *, max_new_tokens=128):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        temperature = temperature,
        top_p = 0.9
    )
    reply = tokenizer.decode(
        output[0, inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    )

    if thinking and '</think>' in reply:
        final_output = reply.split('</think>')[-1].strip()
        return final_output

    return reply.strip()

def parse_intent(question, context):
    instruments = []
    mediator = []
    adjustment_set = []
    running_variable = "None"
    cutoff_value = "None"

    treatment = create_chat_completion(
        messages = [
            {"role": "system", "content": f"""
            Below is a CSV table containing variable names and sample data:

            {context}

            Task: Identify the TREATMENT variable — the variable whose causal effect is being studied on an outcome (e.g., earnings). The TREATMENT is the variable we are trying to estimate the effect of using instrumental variable techniques.

            Important: Do NOT confuse instrument variables with the treatment. An instrument is used to estimate the effect of the treatment. The treatment is the variable being instrumented.

            Clues:
            - If the question says "using X as an instrument," then X is NOT the treatment — it is an instrument for the actual treatment.
            - Questions like "How does education affect earnings using being near a 4-year college as an instrument?" → the treatment is EDUCATION.
            - Focus on what is having the causal effect, not what helps identify it.

            Instructions:
            1. Identify the variable whose causal effect is being studied (X → Y), regardless of how it's instrumented.
            2. Ignore mention of instruments like "using near_college as IV"
            3. Return ONLY the exact column name from the table that is the treatment variable.
            4. Do NOT return any instrument variables."""},
            {"role": "user", "content": question}
        ], temperature = 0.1, thinking = False
    )

    outcome = create_chat_completion(
        messages = [
            {"role": "system", "content": f"""
            You are given a CSV table containing variable names and sample data:

            {context}

            Your task is to identify the **outcome variable**: the variable that is influenced, affected, or predicted in the user's question. This is the result of a causal or correlational relationship — it's the effect, not the cause.

            Key concepts:
            - The outcome is the variable whose value changes *as a result of* the treatment (or predictor).
            - It may also be described as the variable we are trying to explain or model in the question.

            Important Rules:
            - Only output the **exact column name** from the data that best matches the outcome variable.
            - Do **not** include multiple column names, explanations, comments, or instrument variables.
            - Ignore how the treatment is estimated (e.g. via instruments); focus only on what the treatment is **affecting**.

            Example:
            If the question is: "What is the effect of exercise frequency on blood pressure, using distance to gym as an instrument?"
            → The outcome is the column related to blood pressure

            """},
            {"role": "user", "content": question}
        ], temperature = 0.1, thinking = False
    )
    time_variable = create_chat_completion(
        messages=[
            {"role": "system", "content": f"""
            You are given a CSV table containing variable names and sample data:

            {context}

            Your task is to identify the **time variable** used in Difference-in-Differences (DiD) analysis. The time variable represents the temporal dimension that distinguishes between pre-treatment and post-treatment periods.

            Key characteristics of a DiD time variable:
            - Often binary (0/1) indicating before/after treatment implementation
            - May be called "post", "after", "time", "period", or similar temporal indicators  
            - Changes value at the point when treatment begins, affecting ALL units (both treatment and control groups)
            - Represents calendar time, not treatment status

            Important distinctions:
            - Time variable ≠ Treatment variable (which identifies who gets treated)
            - Time variable ≠ Treatment interaction (which is treatment × time)
            - The time variable captures when the policy/intervention was implemented, not who received it

            Critical Rules:
            - Output ONLY the exact column name from the data that represents the time dimension
            - If NO appropriate time variable exists, respond with exactly "None"  
            - Do NOT return treatment variables, outcome variables, or interaction terms
            - Do NOT return multiple column names or add explanations

            Example:
            If analyzing the effect of a minimum wage change in 2015:
            → The time variable might be a column indicating years ≥2015 vs. years <2015

            """},
            {"role": "user", "content": "What is the time variable in the data? Reply None if there is no time variable. Remember to focus entirely on the variable NAME."}
        ],
        temperature = 0.1,
        thinking = False
    )
    running_variable = create_chat_completion(
        messages=[
            {"role": "system", "content": f"Below is a CSV table containing variable names and sample data:\n\n{context}, and attached is the causal question the user desires to answer:\n\n{question}. Does the user specify a specific variable to use as the running variable for RDD? If yes, return the name of the variable ONLY. Do NOT add any additional descriptions or make it a full sentence response. Else, return **None**."},
            {"role": "user", "content": "Is there a specific running variable the user wants to use?"}
        ], temperature = 0.1, thinking = False
    )
    cutoff_value = create_chat_completion(
        messages=[
            {"role": "system", "content": f"Below is a CSV table containing variable names and sample data:\n\n{context}, and attached is the causal question the user desires to answer:\n\n{question}. Does the user specify a specific value for the cutoff of the running variable for RDD? If yes, return the value ONLY. Do NOT add any additional descriptions or make it a full sentence response. Else, return **None**."},
            {"role": "user", "content": "Is there a specific cutoff value for the running variable?"}
        ], temperature = 0.1, thinking = False
    )
    group_variable = create_chat_completion(
            messages=[
                {"role": "system", "content": f"Below is a CSV table containing variable names and sample data:\n\n{context}. When the user asks a question, determine which column in the table corresponds to the **group variable** — the (often binary) variable used for DiD that represents whether the observation is part of the control or treatment group. Be sure to reply **None** if no variable seems appropriate, or choose the **single most appropriate column name** that best represents the outcome. Your response must be **only the exact column name** from the table. Do NOT add any explanation or extra words."},
                {"role": "user", "content": "What is the group variable in the data? Reply None if there is no group variable."}
            ], temperature = 0.1, thinking = False
    )
    inference_algorithm = create_chat_completion(
        messages=[
            {"role": "system", "content": f"""Attached is the causal question the user desires to answer:\n\n{question}. 
            Which causal inference algorithm should we use to answer this question? 
                
            If the user specifies a causal inference algorithm to use out of DiD, DML, Frontdoor Adjustment, G Computation, IV, OLS, Propensity Score, and RDD, return the name of the algorithm exactly how I spelled it when labelling the options, and your confidence in that decision in the format provided below. 
            Else, if the user does not specify an algorithm, use your best judgment to determine the most appropriate algorithm based on the question and the context provided, and provide an appropriate confidence in your decision.
            If the question does not specify an algorithm, and you don't have enough information to determine the best algorithm, answer None.
                
            Return your response as a valid JSON object in the following format:
            {{ 
            "inference_algorithm": "ALGORITHM_NAME" or "None",
            "confidence": "100" if the user specifies an algorithm in their question, "HIGH" if you are confident in the algorithm, "MID" if you are somewhat confident, "LOW" if you are not sure"
            }}
            
            """},
            {"role": "user", "content": "Is there a specific causal inference algorithm the user wants to use?"}
        ], temperature = 0.1, thinking = False
    )

    response = create_chat_completion(
        messages=[
            {"role": "system", "content": f"""
            You are a causal‐inference domain expert. Given:

            • A CSV header and sample data rows:
            {context}

            • A treatment variable: {treatment}
            • An outcome variable: {outcome}
            • The user’s causal question: "{question}"

            Definition:
            An *instrumental variable* must satisfy both:
            1. Relevance: it causes or predicts the treatment.
            2. Exclusion: it does not directly affect the outcome except through the treatment.

            Instructions:
            1. Use domain knowledge and the provided data context.
            2. Identify variables that unambiguously meet both criteria.
            3. Do not include any variables that measure time, individual demographics (e.g. age, race, gender), or pure interaction terms.
            However, access-based measures or regional instruments (e.g., distance to services or geographic variation) can qualify if they satisfy both criteria.
            4. Return **only** a comma-separated list of column names — e.g., `Z1,Z2` — or reply exactly `None`.
            5. Do not add any other text or symbols.

            Example:
            Question: "What is the effect of education on wage?"
            → proximity_to_school, parental_education

            """},
            {"role":"user", "content":"Identify the instruments. Make sure you only include the instruments you are certain are instruments, based off the domain knowledge regarding each of the variables. If the question explicitly mentions specific instruments, include the variables that best represents those instruments regardless of its validity."}
        ], temperature = 0.1, thinking = False
    )
    instrument_resp = response.strip()
    if instrument_resp != "None":
        for instrument in instrument_resp.split(","):
            instruments.append(instrument.strip())
        inference_algorithm = "IV"
        
    response = create_chat_completion(
        messages=[
            {"role":"system", "content":f"""
            You are a causal-inference expert. Given:

            • CSV header and 2 sample rows:
            {context}

            • Treatment: {treatment}
            • Outcome: {outcome}
            • Question core: "{question}"

            Definition:
            A *mediator* lies on the pathway from {treatment} to {outcome}.

            Example (unrelated):
            “How does hours studied affect test scores, mediated by sleep quality?”
            → mediator: sleep_quality

            Instructions:
            1. Rely on header and sample values for semantics.
            2. Return only column names (comma list) or "None".
            """},
            {"role":"user", "content":"Identify mediators. Make sure you only include the mediators you are certain are mediators, based off the domain knowledge regarding each of the variables. If the question explicitly mentions specific mediators, include the variables that best represents those mediators regardless of its validity."}
        ], temperature = 0.1, thinking = False
    )
    mediators_resp = response.strip()
    if mediators_resp != "None":
        for med in mediators_resp.split(","):
            mediator.append(med.strip())


    graph = create_chat_completion(
        messages=[
            {"role":"system", "content":f"""
            You are an expert in causal inference. Your task is to construct a causal graph to help answer a user query.

            Here are the treatment and outcome variables:
            Treatment: {treatment}
            Outcome: {outcome}

            Here are the available variables and an example row in the dataset:
            {context}

            Based on your background knowledge on the available variables, construct a causal graph that captures the relationships between the treatment, outcome, and other relevant variables.

            Use only variables present in the dataset. Do not invent or assume any variables. However, not all variables need to be included—only those that are relevant to the causal relationships should appear in the graph.
            If you are uncertain about a causal relationship, do not include it in the graph. ONLY include relationships you are confident about.

            Return the causal graph in DOT format. The DOT format should include:
            - Nodes for each included variable.
            - Directed edges representing causal relationships among variables.

            Also return the list of edges in the format "A -> B", where A and B are variable names.

            Here is an example of the DOT format:
            digraph G {{
                A -> B;
                B -> C;
                A -> C;
            }}

            And the corresponding list of edges:
            ["A -> B", "B -> C", "A -> C"]

            Return your response as a valid JSON object in the following format:
            {{ 
            "causal_graph": "DOT_FORMAT_STRING",
            "edges": ["EDGE_1", "EDGE_2", ...] 
            }}
            """}, {"role":"user", "content":"Construct an accurate causal graph based on the above information."}
        ], temperature=0.1, thinking=False
    )
    
    return treatment, outcome, time_variable, group_variable, inference_algorithm, instruments, mediator, adjustment_set, running_variable, cutoff_value, graph
