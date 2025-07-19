import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

llm_dir = "./Qwen3-4B"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(llm_dir)
model = AutoModelForCausalLM.from_pretrained(
    llm_dir,
    torch_dtype="auto",
    device_map="auto"
)

def create_chat_completion(messages, temperature, thinking, *, max_new_tokens=64):
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
    return reply.strip()

def parse_intent(question, context):
    instruments = []
    mediator = []
    adjustment_set = []
    running_variable = "None"
    cutoff_value = "None"

    treatment = create_chat_completion(
        messages = [
            {"role": "system", "content": f"""Below is a CSV table containing variable names and sample data:

    {context}

    Your task is to identify the TREATMENT variable - the main causal variable whose effect is being studied. The treatment variable is the primary cause in the causal question, NOT the instruments used to estimate its effect.

    Key distinctions:
    - Treatment variable: The main variable whose causal effect is being studied
    - Instrument variables: Variables used to help estimate the treatment effect (mentioned as "using X as instruments")
    - Outcome variable: The variable being affected by the treatment

    Instructions:
    1. Focus on the core causal relationship being studied (X affects Y)
    2. Ignore any mention of instruments or instrumental variables
    3. Return ONLY the exact column name of the treatment variable
    4. Do NOT return instrument variable names"""},
            {"role": "user", "content": question}
        ], temperature = 0.1, thinking = False
    )

    outcome = create_chat_completion(
        messages = [
            {"role": "system", "content": f"Below is a CSV table containing variable names and sample data:\n\n{context}. When the user asks a question, determine which column in the table corresponds to the **outcome variable** — the variable that is affected or influenced by the treatment in a question. Be sure to choose the **single most appropriate column name** that best represents the outcome, and your response must be **only the exact column name** from the table. Do NOT add any explanation or extra words. For example, if the question asks how does rain impact temperature using season and month as instruments, your response should be the column name most closely related to temperature."},
            {"role": "user", "content": question}
        ], temperature = 0.1, thinking = False
    )
    time_variable = create_chat_completion(
        messages=[
            {"role": "system", "content": f"Below is a CSV table containing variable names and sample data:\n\n{context}. When the user asks a question, determine which column in the table corresponds to the **time variable** — the (often binary) variable used for DiD that represents whether the observation is from before or after the implementation of the policy. Be sure to reply **None** if no variable seems appropriate, or choose the **single most appropriate column name** that best represents the outcome. Your response must be **only the exact column name** from the table. Do NOT add any explanation or extra words."},
            {"role": "user", "content": "What is the time variable in the data? Reply None if there is no time variable."}
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
            {"role": "system", "content": f"Attached is the causal question the user desires to answer:\n\n{question}. Does the user specify a causal inference algorithm to use out of DiD, DML, Frontdoor Adjustment, G Computation, IV, OLS, Propensity Score, and RDD? If yes, return the name of the algorithm ONLY, exactly how I spelled it when labelling the options. If the question mentions using an instrument, return **IV**. If the question mentions using a mediator, return **Frontdoor Adjustment**. If the question mentions using a running variable, return **RDD**. Else, return **None**"},
            {"role": "user", "content": "Is there a specific causal inference algorithm the user wants to use?"}
        ], temperature = 0.1, thinking = False
    )

    if inference_algorithm == "IV":
        resp = create_chat_completion(
            messages=[
                {"role": "system", "content": (f"Below is a CSV table containing variable names and sample data:\n\n{context}, and attached is the causal question the user desires to answer:\n\n{question}. Does the user specify a specific variables to use as instruments for IV? If yes, return the name of the variables ONLY in the format **A,B,C**. For example, if the question asks how does rain impact temperature using season and month as instruments, your response should be the column name most closely related to seasons and month in the foramt seasons,month. Do NOT add any additional descriptions or make it a full sentence response. Else, return **None**.")},
                {"role": "user", "content": "Is there a specific instrument the user wants to use?"}
            ], temperature=0.1, thinking=False
        )
        instruments_resp = resp.strip()
        if instruments_resp != "None":
            for instr in instruments_resp.split(","):
                instruments.append(instr.strip())
    elif inference_algorithm == "Frontdoor Adjustment":
        resp = create_chat_completion(
            messages=[
                {"role": "system", "content": (f"Below is a CSV table containing variable names and sample data:\n\n{context}, and attached is the causal question the user desires to answer:\n\n{question}. Does the user specify a specific variables to use as mediators for Frontdoor Adjustment? If yes, return the name of the variables ONLY in the format **A,B,C**. Do NOT add any additional descriptions or make it a full sentence response. Else, return **None**.")},
                {"role": "user", "content": "Is there a specific mediator the user wants to use?"}
            ], temperature=0.1, thinking=False
        )
        mediators_resp = resp.strip()
        if mediators_resp != "None":
            for med in mediators_resp.split(","):
                mediator.append(med.strip())

    elif inference_algorithm == "RDD":
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
    
    return treatment, outcome, time_variable, group_variable, inference_algorithm, instruments, mediator, running_variable, cutoff_value