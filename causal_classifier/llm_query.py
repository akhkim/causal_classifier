import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

llm_dir = "./Qwen3-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(llm_dir)

# Load model and ensure it's on the same device as we'll use for inputs
if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        llm_dir,
        torch_dtype=torch.float16,  # Use float16 for GPU efficiency
        device_map="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        llm_dir,
        torch_dtype="auto",
        device_map="cpu"
    )

# Get the actual device the model is on
if hasattr(model, 'device'):
    model_device = model.device
elif hasattr(model, 'parameters'):
    model_device = next(model.parameters()).device
else:
    model_device = device

print(f"LLM loaded on device: {model_device}")

def create_chat_completion(messages, temperature, thinking, *, max_new_tokens=128):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking
    )
    # Use the model's actual device instead of the global device variable
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
    
    try:
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
        
    except RuntimeError as e:
        if "device" in str(e).lower():
            print(f"Device error encountered: {e}")
            print(f"Model device: {model_device}, Input device: {inputs.input_ids.device}")
            # Try moving model to CPU as fallback
            model.to('cpu')
            inputs = inputs.to('cpu')
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
        else:
            raise e

def parse_json_response(response, key):
    """Parse JSON response from LLM and extract the specified key."""
    try:
        parsed = json.loads(response)
        return parsed.get(key, "None"), parsed.get("confidence", 0.5)
    except json.JSONDecodeError:
        # Fallback: return the raw response if JSON parsing fails
        return response.strip(), 0.3

def _query_did_variables(question, context):
    """Query for DiD-specific variables: time_variable and group_variable"""
    results = {}
    
    time_variable_response = create_chat_completion(
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
            - Identify the exact column name from the data that represents the time dimension
            - If NO appropriate time variable exists, return "None"  
            - Do NOT return treatment variables, outcome variables, or interaction terms
            - Do NOT return multiple column names or add explanations

            Example:
            If analyzing the effect of a minimum wage change in 2015:
            → The time variable might be a column indicating years ≥2015 vs. years <2015

            Return your response as a valid JSON object in the following format:
            {{ 
            "time_variable": "COLUMN_NAME" or "None",
            "confidence": YOUR_CONFIDENCE_SCORE
            }}
            
            Where confidence is a number between 0 and 1 indicating how certain you are about this identification (0 = not sure at all, 1 = completely certain).
            """},
            {"role": "user", "content": "What is the time variable in the data? Reply None if there is no time variable. Remember to focus entirely on the variable NAME."}
        ],
        temperature = 0.1,
        thinking = False
    )
    time_variable, time_variable_confidence = parse_json_response(time_variable_response, "time_variable")
    results['time_variable'] = {'value': time_variable, 'confidence': time_variable_confidence}
    
    group_variable_response = create_chat_completion(
            messages=[
                {"role": "system", "content": f"""Below is a CSV table containing variable names and sample data:

                {context}

                When the user asks a question, determine which column in the table corresponds to the **group variable** — the (often binary) variable used for DiD that represents whether the observation is part of the control or treatment group. 

                If no variable seems appropriate, return "None". Otherwise, choose the single most appropriate column name that best represents the group assignment.

                Return your response as a valid JSON object in the following format:
                {{ 
                "group_variable": "COLUMN_NAME" or "None",
                "confidence": YOUR_CONFIDENCE_SCORE
                }}

                Where confidence is a number between 0 and 1 indicating how certain you are about this identification (0 = not sure at all, 1 = completely certain)."""},
                {"role": "user", "content": "What is the group variable in the data? Reply None if there is no group variable."}
            ], temperature = 0.1, thinking = False
    )
    group_variable, group_variable_confidence = parse_json_response(group_variable_response, "group_variable")
    results['group_variable'] = {'value': group_variable, 'confidence': group_variable_confidence}
    
    return results

def _query_rdd_variables(question, context):
    """Query for RDD-specific variables: running_variable and cutoff_value"""
    results = {}
    
    running_variable_response = create_chat_completion(
        messages=[
            {"role": "system", "content": f"""Below is a CSV table containing variable names and sample data:

            {context}

            And attached is the causal question the user desires to answer:

            {question}

            Does the user specify a specific variable to use as the running variable for RDD? If yes, return the name of the variable. If no specific running variable is mentioned, return "None".

            Return your response as a valid JSON object in the following format:
            {{ 
            "running_variable": "VARIABLE_NAME" or "None",
            "confidence": YOUR_CONFIDENCE_SCORE
            }}

            Where confidence is a number between 0 and 1 indicating how certain you are about this identification (0 = not sure at all, 1 = completely certain)."""},
            {"role": "user", "content": "Is there a specific running variable the user wants to use?"}
        ], temperature = 0.1, thinking = False
    )
    running_variable, running_variable_confidence = parse_json_response(running_variable_response, "running_variable")
    results['running_variable'] = {'value': running_variable, 'confidence': running_variable_confidence}
    
    cutoff_value_response = create_chat_completion(
        messages=[
            {"role": "system", "content": f"""Below is a CSV table containing variable names and sample data:

            {context}

            And attached is the causal question the user desires to answer:

            {question}

            Does the user specify a specific value for the cutoff of the running variable for RDD? If yes, return the value. If no specific cutoff value is mentioned, return "None".

            Return your response as a valid JSON object in the following format:
            {{ 
            "cutoff_value": "VALUE" or "None",
            "confidence": YOUR_CONFIDENCE_SCORE
            }}

            Where confidence is a number between 0 and 1 indicating how certain you are about this identification (0 = not sure at all, 1 = completely certain)."""},
            {"role": "user", "content": "Is there a specific cutoff value for the running variable?"}
        ], temperature = 0.1, thinking = False
    )
    cutoff_value, cutoff_value_confidence = parse_json_response(cutoff_value_response, "cutoff_value")
    results['cutoff_value'] = {'value': cutoff_value, 'confidence': cutoff_value_confidence}
    
    return results

def _query_iv_variables(question, enhanced_context, treatment, outcome, simple_context=None):
    """Query for IV-specific variables: instruments"""
    results = {}
    
    # Use both contexts when available for better instrument detection
    context_for_llm = enhanced_context
    if simple_context:
        context_for_llm = f"SIMPLE DATA VIEW:\n{simple_context}\n\nDETAILED ANALYSIS:\n{enhanced_context}"
    
    response = create_chat_completion(
        messages=[
            {"role": "system", "content": f"""
            You are a causal inference expert specializing in instrumental variable identification. Given:

            • CSV data with variables and sample values:
            {context_for_llm}

            • Treatment variable: {treatment}
            • Outcome variable: {outcome}
            • Research question: "{question}"

            Task: Identify ALL variables that could serve as instrumental variables for estimating the causal effect of {treatment} on {outcome}.

            Instrumental Variable Definition:
            An instrument must satisfy two conditions:
            1. **Relevance**: It affects/predicts the treatment variable ({treatment})
            2. **Exclusion**: It does NOT directly affect the outcome ({outcome}) except through its effect on the treatment

            Common Types of Valid Instruments:
            - **Proximity/Access variables**: Distance to schools, hospitals, etc. (e.g., nearc2, nearc4 for education)
            - **Geographic variation**: Regional differences, weather patterns, soil quality
            - **Policy variation**: Different policies across regions/time periods
            - **Lottery/Random assignment**: Random allocation mechanisms
            - **Family background**: Parental characteristics that affect treatment choice but not directly outcome
            - **Supply-side factors**: Availability of services, infrastructure

            Special Focus for Education-Earnings Analysis:
            If the question involves education's effect on earnings/wages, look particularly for:
            - Proximity to colleges (nearc2, nearc4, etc.)
            - Compulsory schooling laws
            - Parental education (as it affects schooling choice but may not directly affect wages)
            - Geographic or policy variation in education access

            Instructions:
            1. Examine EVERY variable in the dataset for potential instrumental validity
            2. Consider domain knowledge about education, earnings, and policy contexts
            3. Look for variables that would logically affect treatment choice but not outcome directly
            4. Include variables with names like "near", "prox", "dist", "access", or geographic indicators
            5. Be INCLUSIVE rather than restrictive - if a variable could plausibly be an instrument, include it
            6. Assign confidence scores based on theoretical validity (0.7+ for strong instruments)

            Important: Even if you're not 100% certain about exclusion restriction, include variables that clearly affect treatment if they seem plausibly exogenous to the outcome.

            Return your response as a valid JSON object in the following format:
            {{ 
            "instruments": {{"VARIABLE_NAME": CONFIDENCE_SCORE, "VARIABLE_NAME2": CONFIDENCE_SCORE}} or "None",
            "overall_confidence": YOUR_OVERALL_CONFIDENCE_SCORE
            }}

            Where:
            - instruments: A dictionary mapping each instrument variable name to its individual confidence score (0-1)
            - overall_confidence: Your overall confidence in the instrument identification task (0-1)
            - Use "None" for instruments if no valid instruments exist

            """},
            {"role":"user", "content":"Identify ALL potential instrumental variables in the dataset. For education-earnings questions, pay special attention to proximity variables (nearc2, nearc4) and family background variables. Be inclusive and identify any variable that could plausibly serve as an instrument."}
        ], temperature = 0.1, thinking = False
    )
    
    try:
        parsed = json.loads(response)
        instruments_dict = parsed.get("instruments", "None")
        overall_confidence = parsed.get("overall_confidence", 0.5)
        
        if instruments_dict == "None":
            results['instruments'] = {'value': "None", 'confidence': overall_confidence}
        else:
            # Store both the dictionary format and a backward-compatible string format
            instrument_names = list(instruments_dict.keys())
            results['instruments'] = {
                'value': ",".join(instrument_names) if instrument_names else "None",
                'confidence': overall_confidence,
                'individual_confidences': instruments_dict
            }
    except (json.JSONDecodeError, AttributeError):
        # Fallback to old format
        instrument_resp, instruments_confidence = parse_json_response(response, "instruments")
        results['instruments'] = {'value': instrument_resp, 'confidence': instruments_confidence}
    
    return results

def _query_mediator_variables(question, enhanced_context, treatment, outcome, simple_context=None):
    """Query for Frontdoor Adjustment-specific variables: mediators"""
    results = {}
    
    # Use both contexts when available for better mediator detection
    context_for_llm = enhanced_context
    if simple_context:
        context_for_llm = f"SIMPLE DATA VIEW:\n{simple_context}\n\nDETAILED ANALYSIS:\n{enhanced_context}"
    
    response = create_chat_completion(
        messages=[
            {"role":"system", "content":f"""
            You are a causal-inference expert. Given:

            • CSV header and 2 sample rows:
            {context_for_llm}

            • Treatment: {treatment}
            • Outcome: {outcome}
            • Question core: "{question}"

            Definition:
            A *mediator* lies on the pathway from {treatment} to {outcome}.

            Example (unrelated):
            "How does hours studied affect test scores, mediated by sleep quality?"
            → mediator: sleep_quality

            Instructions:
            1. Rely on header and sample values for semantics.
            2. For each mediator you identify, provide a confidence score between 0 and 1.
            3. If no mediators exist, return "None" for mediators.
            
            Return your response as a valid JSON object in the following format:
            {{ 
            "mediators": {{"VARIABLE_NAME": CONFIDENCE_SCORE, "VARIABLE_NAME2": CONFIDENCE_SCORE}} or "None",
            "overall_confidence": YOUR_OVERALL_CONFIDENCE_SCORE
            }}

            Where:
            - mediators: A dictionary mapping each mediator variable name to its individual confidence score (0-1)
            - overall_confidence: Your overall confidence in the mediator identification task (0-1)
            - Use "None" for mediators if no valid mediators exist
            """},
            {"role":"user", "content":"Identify mediators with individual confidence scores. Make sure you only include the mediators you are certain are mediators, based off the domain knowledge regarding each of the variables. If the question explicitly mentions specific mediators, include the variables that best represents those mediators regardless of its validity."}
        ], temperature = 0.1, thinking = False
    )
    
    try:
        parsed = json.loads(response)
        mediators_dict = parsed.get("mediators", "None")
        overall_confidence = parsed.get("overall_confidence", 0.5)
        
        if mediators_dict == "None":
            results['mediators'] = {'value': "None", 'confidence': overall_confidence}
        else:
            # Store both the dictionary format and a backward-compatible string format
            mediator_names = list(mediators_dict.keys())
            results['mediators'] = {
                'value': ",".join(mediator_names) if mediator_names else "None",
                'confidence': overall_confidence,
                'individual_confidences': mediators_dict
            }
    except (json.JSONDecodeError, AttributeError):
        # Fallback to old format
        mediators_resp, mediators_confidence = parse_json_response(response, "mediators")
        results['mediators'] = {'value': mediators_resp, 'confidence': mediators_confidence}
    
    return results


def _query_adjustment_variables(question, enhanced_context, treatment, outcome, simple_context=None):
    """Query for adjustment set (confounders)"""
    results = {}
    
    # Use both contexts when available for better confounder detection  
    context_for_llm = enhanced_context
    if simple_context:
        context_for_llm = f"SIMPLE DATA VIEW:\n{simple_context}\n\nDETAILED ANALYSIS:\n{enhanced_context}"
    
    adjustment_set_response = create_chat_completion(
        messages=[
            {"role":"system", "content":f"""
            You are a causal-inference expert. Given:

            • CSV header and sample data:
            {context_for_llm}

            • Treatment: {treatment}
            • Outcome: {outcome}
            • Question: "{question}"

            Definition:
            An *adjustment set* (also called confounders or control variables) are variables that should be controlled for to estimate the causal effect of treatment on outcome. These are variables that:
            1. Affect both the treatment and the outcome (common causes)
            2. Are not on the causal pathway from treatment to outcome (not mediators)
            3. Are not affected by the treatment (not colliders or post-treatment variables)

            Instructions:
            1. Identify variables that are likely confounders based on domain knowledge.
            2. Include variables that might create spurious correlations if not controlled for.
            3. Exclude mediators, instruments, and post-treatment variables.
            4. For each confounder you identify, provide a confidence score between 0 and 1.
            5. If no confounders exist, return "None" for adjustment_set.

            Example:
            Question: "What is the effect of education on wage?"
            → {{"age": 0.9, "experience": 0.95, "ability": 0.7, "family_background": 0.8}}

            Return your response as a valid JSON object in the following format:
            {{ 
            "adjustment_set": {{"VARIABLE_NAME": CONFIDENCE_SCORE, "VARIABLE_NAME2": CONFIDENCE_SCORE}} or "None",
            "overall_confidence": YOUR_OVERALL_CONFIDENCE_SCORE
            }}

            Where:
            - adjustment_set: A dictionary mapping each confounder variable name to its individual confidence score (0-1)
            - overall_confidence: Your overall confidence in the confounder identification task (0-1)
            - Use "None" for adjustment_set if no valid confounders exist
            """},
            {"role":"user", "content":"Identify the adjustment set (confounders) with individual confidence scores. Make sure you only include variables you are certain are confounders based on domain knowledge."}
        ], temperature = 0.1, thinking = False
    )
    
    try:
        parsed = json.loads(adjustment_set_response)
        adjustment_dict = parsed.get("adjustment_set", "None")
        overall_confidence = parsed.get("overall_confidence", 0.5)
        
        if adjustment_dict == "None":
            results['adjustment_set'] = {'value': "None", 'confidence': overall_confidence}
        else:
            # Store both the dictionary format and a backward-compatible string format
            adjustment_names = list(adjustment_dict.keys())
            results['adjustment_set'] = {
                'value': ",".join(adjustment_names) if adjustment_names else "None",
                'confidence': overall_confidence,
                'individual_confidences': adjustment_dict
            }
    except (json.JSONDecodeError, AttributeError):
        # Fallback to old format
        adjustment_resp, adjustment_confidence = parse_json_response(adjustment_set_response, "adjustment_set")
        results['adjustment_set'] = {'value': adjustment_resp, 'confidence': adjustment_confidence}
    
    return results

def _query_gwas_variable_mapping(question, context, treatment, outcome, target_variable_type):
    """Query for GWAS-specific variable mapping and composite variable construction"""
    results = {}
    
    # Determine which variable is the target (treatment or outcome based on GWAS analysis type)
    target_variable = treatment if target_variable_type == "EXPOSURE" else outcome
    
    gwas_mapping_response = create_chat_completion(
        messages=[
            {"role": "system", "content": f"""
            You are a GWAS (Genome-Wide Association Studies) and UK Biobank expert. Given:

            • CSV header and sample data:
            {context}

            • Treatment: {treatment}
            • Outcome: {outcome}
            • Target Variable for GWAS: {target_variable} (type: {target_variable_type})
            • Research Question: "{question}"

            Your task is to identify the specific questionnaire fields that should be used to construct the target phenotype for GWAS analysis, similar to how complex traits like insomnia are constructed from multiple questionnaire responses.

            Example (Insomnia GWAS):
            - TTS (time to sleep), WAKE (night waking), EARLY (early waking) → sleep trouble
            - FREQ (frequency) → regular occurrence  
            - MOOD, CONC, GEN (impact variables) → functional impairment
            - DUR (duration) → chronicity
            Final phenotype: insomnia = (sleep_trouble & freq & impact & dur)

            Instructions:
            1. Identify ALL questionnaire fields in the dataset that relate to the target variable "{target_variable}"
            2. Group these fields by their conceptual role (e.g., symptoms, frequency, severity, duration, impact)
            3. Suggest how to combine these fields to create a robust phenotype definition
            4. Include confidence scores for each field's relevance
            5. Consider standard GWAS practices for phenotype construction

            Key considerations:
            - Look for direct symptom measures
            - Frequency/occurrence indicators  
            - Severity/intensity measures
            - Duration/chronicity indicators
            - Functional impact measures
            - Quality of life measures
            - Any related diagnostic or clinical variables

            Return your response as a valid JSON object in the following format:
            {{
            "target_phenotype": "{target_variable}",
            "questionnaire_fields": {{
                "FIELD_NAME1": {{
                    "category": "symptoms|frequency|severity|duration|impact|other",
                    "confidence": CONFIDENCE_SCORE,
                    "description": "brief description of what this field measures"
                }},
                "FIELD_NAME2": {{
                    "category": "symptoms|frequency|severity|duration|impact|other", 
                    "confidence": CONFIDENCE_SCORE,
                    "description": "brief description"
                }}
            }},
            "phenotype_construction": {{
                "method": "binary|continuous|ordinal",
                "logic": "detailed description of how to combine the fields",
                "threshold_suggestions": "any threshold recommendations for binary conversion"
            }},
            "covariates": [
                "LIST", "OF", "COVARIATE", "FIELD", "NAMES"
            ],
            "overall_confidence": YOUR_OVERALL_CONFIDENCE_SCORE
            }}

            Where:
            - questionnaire_fields: Map each relevant field to its category and confidence
            - phenotype_construction: Detailed instructions for creating the final phenotype
            - covariates: Standard covariates needed for GWAS (age, sex, PCs, etc.)
            - overall_confidence: Your confidence in this variable mapping (0-1)
            """},
            {"role": "user", "content": f"Identify all questionnaire fields related to '{target_variable}' and explain how to construct this phenotype for GWAS analysis."}
        ],
        temperature=0.1,
        thinking=False,
        max_new_tokens=512
    )
    
    try:
        parsed = json.loads(gwas_mapping_response)
        results['gwas_variable_mapping'] = {
            'target_phenotype': parsed.get('target_phenotype', target_variable),
            'questionnaire_fields': parsed.get('questionnaire_fields', {}),
            'phenotype_construction': parsed.get('phenotype_construction', {}),
            'covariates': parsed.get('covariates', []),
            'confidence': parsed.get('overall_confidence', 0.5)
        }
    except (json.JSONDecodeError, AttributeError):
        # Fallback to simpler mapping
        results['gwas_variable_mapping'] = {
            'target_phenotype': target_variable,
            'questionnaire_fields': {},
            'phenotype_construction': {'method': 'unknown', 'logic': 'Could not parse field mapping'},
            'covariates': [],
            'confidence': 0.3
        }
    
    return results

def _query_causal_graph(question, context, treatment, outcome):
    """Query for causal graph construction"""
    results = {}
    
    graph_response = create_chat_completion(
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
            "edges": ["EDGE_1", "EDGE_2", ...],
            "confidence": YOUR_CONFIDENCE_SCORE
            }}
            
            Where confidence is a number between 0 and 1 indicating how certain you are about the causal relationships in this graph (0 = not sure at all, 1 = completely certain).
            """}, {"role":"user", "content":"Construct an accurate causal graph based on the above information."}
        ], temperature=0.1, thinking=False
    )
    graph, graph_confidence = parse_json_response(graph_response, "causal_graph")
    results['graph'] = {'value': graph, 'confidence': graph_confidence}
    
    return results

def _query_opengwas_id(variable_name, variable_description, variable_type, question_context):
    """
    Query LLM to find the most appropriate OpenGWAS ID for a variable not in the dataset
    
    Args:
        variable_name: Name of the variable (e.g., "BMI", "coronary_artery_disease")
        variable_description: Description of what the variable represents
        variable_type: "EXPOSURE" or "OUTCOME" 
        question_context: The original research question for context
    
    Returns:
        dict: Contains OpenGWAS ID, confidence, and reasoning
    """
    
    opengwas_response = create_chat_completion(
        messages=[
            {"role": "system", "content": f"""You are an expert in GWAS databases and the OpenGWAS platform (https://gwas.mrcieu.ac.uk/). 

            Your task is to identify the most appropriate OpenGWAS study ID for a given phenotype/trait.

            Variable to find: {variable_name}
            Variable description: {variable_description}
            Variable type: {variable_type}
            Research context: {question_context}

            OpenGWAS contains thousands of GWAS summary statistics. Common study ID patterns include:
            - ieu-a-XXXX: Early IEU studies
            - ieu-b-XXXX: IEU OpenGWAS studies  
            - ukb-b-XXXX: UK Biobank studies
            - ebi-a-XXXX: EBI studies
            - finn-b-XXXX: FinnGen studies
            - bbj-a-XXXX: Biobank Japan studies

            Example mappings:
            - BMI: "ieu-a-2" or "ukb-b-19953"
            - Height: "ieu-a-89" or "ukb-b-16576" 
            - Coronary artery disease: "ieu-a-7" or "finn-b-I9_CORONER"
            - Type 2 diabetes: "ieu-a-26" or "finn-b-E4_DM2"
            - Educational attainment: "ieu-a-1239"
            - Smoking initiation: "ieu-a-954"
            - Depression: "ieu-a-1187" or "finn-b-F5_MOOD_ANX"
            - Schizophrenia: "ieu-a-22"
            - Blood pressure: "ieu-a-1031" (systolic) or "ieu-a-1032" (diastolic)

            Instructions:
            1. Based on the variable name and description, identify the most likely OpenGWAS study ID
            2. Consider the sample size and population ancestry (prefer larger, more diverse studies when possible)
            3. For binary traits, prefer case-control studies
            4. For continuous traits, prefer quantitative trait studies
            5. Consider recent/updated versions of studies when available
            6. If multiple good options exist, provide the most commonly used/cited one

            Return your response as a valid JSON object:
            {{
                "opengwas_id": "study-id-here",
                "study_description": "Brief description of the study/trait",
                "sample_size_estimate": "approximate sample size if known",
                "population": "population ancestry (e.g., European, East Asian, Mixed)",
                "trait_type": "binary/continuous/ordinal",
                "confidence": 0.0-1.0,
                "reasoning": "Why this study was selected",
                "alternative_ids": ["alt-id-1", "alt-id-2"],
                "notes": "Any important considerations or limitations"
            }}

            Where confidence reflects how certain you are this is the correct/best study for this phenotype."""},
            {"role": "user", "content": f"Find the most appropriate OpenGWAS study ID for: {variable_name} ({variable_description})"}
        ],
        temperature=0.1,
        thinking=False,
        max_new_tokens=400
    )
    
    try:
        parsed = json.loads(opengwas_response)
        return {
            'opengwas_id': parsed.get('opengwas_id', 'unknown'),
            'study_description': parsed.get('study_description', ''),
            'sample_size_estimate': parsed.get('sample_size_estimate', 'unknown'),
            'population': parsed.get('population', 'unknown'),
            'trait_type': parsed.get('trait_type', 'unknown'),
            'confidence': parsed.get('confidence', 0.5),
            'reasoning': parsed.get('reasoning', ''),
            'alternative_ids': parsed.get('alternative_ids', []),
            'notes': parsed.get('notes', ''),
            'search_successful': True
        }
    except (json.JSONDecodeError, AttributeError):
        # Fallback parsing
        opengwas_id, confidence = parse_json_response(opengwas_response, "opengwas_id")
        return {
            'opengwas_id': opengwas_id,
            'study_description': 'Parsed with fallback method',
            'sample_size_estimate': 'unknown',
            'population': 'unknown', 
            'trait_type': 'unknown',
            'confidence': confidence,
            'reasoning': 'Fallback parsing used',
            'alternative_ids': [],
            'notes': 'Limited information due to parsing issues',
            'search_successful': opengwas_id != 'unknown'
        }

def parse_intent(question, context, data, variable_analysis, relationships, causal_patterns):
    """Enhanced intent parsing with statistical analysis and simple context"""
    from .variable_analysis import create_enhanced_context
    
    # Create enhanced context with statistical patterns
    enhanced_context = create_enhanced_context(data, variable_analysis, relationships, causal_patterns)
    
    # Use the simple context passed in as parameter (this is the traditional data.head(3) format)
    simple_context = context
    
    # Dictionary to store all value-confidence pairs
    results = {}

    # Enhanced prompt for placeholder variables
    enhanced_system_prompt = f"""You are an expert in causal inference analysis. You will be given a research question and dataset information that may include variables with placeholder names (like X1, X2, var1, col_1, etc.) that don't describe their meaning.

CRITICAL INSTRUCTIONS FOR PLACEHOLDER VARIABLES:
- Use the statistical patterns, data types, variable relationships, and causal suggestions provided
- Binary variables (2 unique values) are often treatments or group indicators
- Variables with high variance and right-skewed distributions might be income/wage outcomes
- Variables with many strong relationships to others might be outcomes or important confounders
- Age-like variables (0-120 range) are typically confounders
- Education-like variables (ordered, 5-25 levels) are typically confounders
- Time-increasing variables are likely time indicators
- Consider the suggested causal relationships provided in the analysis

ENHANCED CONTEXT:
{enhanced_context}

Your task: Given the research question "{question}", identify the most appropriate variables for causal analysis based on BOTH semantic understanding AND statistical patterns.

When variable names are meaningless, rely heavily on:
1. Statistical patterns and distributions
2. Variable relationships and correlations  
3. Suggested causal relationships from the analysis
4. Domain knowledge about typical causal structures

Return the same JSON format as always, but base your decisions on statistical evidence when variable names are uninformative."""

    # Core variables needed for all analyses
    treatment_response = create_chat_completion(
        messages = [
            {"role": "system", "content": f"""
            Below is a CSV table containing variable names and sample data with enhanced statistical analysis:

            {enhanced_context}

            Task: Identify the TREATMENT variable — the variable whose causal effect is being studied on an outcome (e.g., earnings). The TREATMENT is the variable we are trying to estimate the effect of using instrumental variable techniques.

            Important: Do NOT confuse instrument variables with the treatment. An instrument is used to estimate the effect of the treatment. The treatment is the variable being instrumented.

            Clues:
            - If the question says "using X as an instrument," then X is NOT the treatment — it is an instrument for the actual treatment.
            - Questions like "How does education affect earnings using being near a 4-year college as an instrument?" → the treatment is EDUCATION.
            - Focus on what is having the causal effect, not what helps identify it.

            Instructions:
            1. Identify the variable whose causal effect is being studied (X → Y), regardless of how it's instrumented.
            2. Ignore mention of instruments like "using near_college as IV"
            3. Return the exact column name from the table that is the treatment variable.
            4. Do NOT return any instrument variables.
            
            Return your response as a valid JSON object in the following format:
            {{ 
            "treatment_variable": "COLUMN_NAME",
            "confidence": YOUR_CONFIDENCE_SCORE
            }}
            
            Where confidence is a number between 0 and 1 indicating how certain you are about this identification (0 = not sure at all, 1 = completely certain)."""},
            {"role": "user", "content": question}
        ], temperature = 0.1, thinking = False
    )
    treatment, treatment_confidence = parse_json_response(treatment_response, "treatment_variable")
    results['treatment'] = {'value': treatment, 'confidence': treatment_confidence}

    outcome_response = create_chat_completion(
        messages = [
            {"role": "system", "content": f"""
            You are given a CSV table containing variable names and sample data with enhanced statistical analysis:

            {enhanced_context}

            Your task is to identify the **outcome variable**: the variable that is influenced, affected, or predicted in the user's question. This is the result of a causal or correlational relationship — it's the effect, not the cause.

            Key concepts:
            - The outcome is the variable whose value changes *as a result of* the treatment (or predictor).
            - It may also be described as the variable we are trying to explain or model in the question.

            Important Rules:
            - Only identify the exact column name from the data that best matches the outcome variable.
            - Do not include multiple column names, explanations, comments, or instrument variables.
            - Ignore how the treatment is estimated (e.g. via instruments); focus only on what the treatment is affecting.

            Example:
            If the question is: "What is the effect of exercise frequency on blood pressure, using distance to gym as an instrument?"
            → The outcome is the column related to blood pressure

            Return your response as a valid JSON object in the following format:
            {{ 
            "outcome_variable": "COLUMN_NAME",
            "confidence": YOUR_CONFIDENCE_SCORE
            }}
            
            Where confidence is a number between 0 and 1 indicating how certain you are about this identification (0 = not sure at all, 1 = completely certain).
            """},
            {"role": "user", "content": question}
        ], temperature = 0.1, thinking = False
    )
    outcome, outcome_confidence = parse_json_response(outcome_response, "outcome_variable")
    results['outcome'] = {'value': outcome, 'confidence': outcome_confidence}
    
    # Determine the inference algorithm first
    inference_algorithm_response = create_chat_completion(
        messages=[
            {"role": "system", "content": f"""You are a causal inference expert. Given the following:

            Research Question: {question}
            
            Available Data Variables and Sample:
            {enhanced_context}

            Your task is to recommend the most appropriate causal inference algorithm.

            Available algorithms:
            - DiD (Difference-in-Differences): For policy interventions with before/after and treatment/control groups
            - DML (Double Machine Learning): For high-dimensional data with confounders
            - Frontdoor Adjustment: When mediators are available and confounders are unobserved
            - G Computation: For complex causal pathways with multiple treatments/outcomes
            - IV (Instrumental Variables): When treatment assignment has exogenous variation
            - MR (Mendelian Randomization): For GWAS data using genetic variants as instruments
            - OLS (Ordinary Least Squares): For simple linear relationships with minimal confounding
            - Propensity Score: For matching treated and control units
            - RDD (Regression Discontinuity): For treatment assignment based on cutoff thresholds

            Instructions:
            1. If the user explicitly mentions a specific algorithm, return that algorithm name exactly as listed above
            2. If not explicitly mentioned, analyze the question and data to recommend the most appropriate method
            3. **IMPORTANT**: If the data contains genetic variants (SNPs, genetic markers) combined with phenotypic data, recommend "MR" (Mendelian Randomization)
            4. Consider the nature of the treatment assignment, available variables, and research design
            5. Default to "OLS" for simple causal questions where other methods aren't clearly indicated
            6. Only return "None" if the question is not about causal inference at all

            Key decision factors:
            - **Genetic data present**: Use MR for GWAS datasets with genetic variants and phenotypes
            - Experimental vs observational data
            - Time series vs cross-sectional
            - Binary vs continuous treatment
            - Availability of instruments, time periods, or cutoffs
            - Presence of confounders
                
            Return your response as a valid JSON object in the following format:
            {{ 
            "inference_algorithm": "ALGORITHM_NAME",
            "confidence": YOUR_CONFIDENCE_SCORE,
            "rationale": "Brief explanation of why this algorithm was chosen"
            }}
            
            Where confidence is a number between 0 and 1 (0 = not sure at all, 1 = completely certain).
            """},
            {"role": "user", "content": f"Based on the question '{question}' and the available data, what is the most appropriate causal inference algorithm?"}
        ], temperature = 0.1, thinking = False
    )
    try:
        parsed = json.loads(inference_algorithm_response)
        inference_algorithm = parsed.get("inference_algorithm", "OLS")
        inference_algorithm_confidence = parsed.get("confidence", 0.7)
        algorithm_rationale = parsed.get("rationale", "Default algorithm selection")
        
        # If still None, default to OLS for basic causal questions
        if inference_algorithm == "None":
            inference_algorithm = "OLS"
            inference_algorithm_confidence = 0.6
            algorithm_rationale = "Defaulted to OLS for basic causal analysis"
            
    except (json.JSONDecodeError, AttributeError):
        # Fallback parsing - try to extract algorithm name from response
        print(f"JSON parsing failed, attempting fallback parsing for: {inference_algorithm_response[:100]}...")
        
        # Check if the response itself is a JSON-like string that contains the algorithm
        if "inference_algorithm" in inference_algorithm_response:
            try:
                # Try to extract just the algorithm name using regex
                import re
                algorithm_match = re.search(r'"inference_algorithm":\s*"([^"]+)"', inference_algorithm_response)
                if algorithm_match:
                    inference_algorithm = algorithm_match.group(1)
                    inference_algorithm_confidence = 0.7
                    algorithm_rationale = "Extracted from JSON-like response"
                else:
                    # Try alternative format
                    inference_algorithm, inference_algorithm_confidence = parse_json_response(inference_algorithm_response, "inference_algorithm")
                    algorithm_rationale = "Parsed with fallback method"
            except:
                inference_algorithm = "OLS"
                inference_algorithm_confidence = 0.6
                algorithm_rationale = "Fallback to OLS due to parsing error"
        else:
            inference_algorithm, inference_algorithm_confidence = parse_json_response(inference_algorithm_response, "inference_algorithm")
            algorithm_rationale = "Parsed with fallback method"
        
        # Apply same None -> OLS logic for fallback
        if inference_algorithm == "None" or not inference_algorithm:
            inference_algorithm = "OLS"
            inference_algorithm_confidence = 0.6
            algorithm_rationale = "Defaulted to OLS for basic causal analysis"
    
    results['inference_algorithm'] = {
        'value': inference_algorithm, 
        'confidence': inference_algorithm_confidence,
        'rationale': algorithm_rationale
    }

    gwas_response = create_chat_completion(
        messages=[
            {"role":"system", "content":f"""
            You are a genetics and genomics expert. Given:

            • CSV header and sample data:
            {enhanced_context}

            • Treatment/Exposure: {treatment}
            • Outcome: {outcome}
            • Question: "{question}"

            Task 1: Determine if this dataset appears to be questionnaire/survey data that includes genetic information suitable for GWAS analysis.

            Definition:
            Questionnaire data suitable for GWAS (Genome-Wide Association Studies) contains:
            1. Survey/questionnaire responses (behavioral, lifestyle, health, demographic questions)
            2. Genetic information (SNPs, genetic variants, genomic markers)
            3. Individual/participant identifiers
            4. Mix of self-reported phenotypes and genetic data
            5. Population characteristics for genetic analysis

            Look for indicators such as:
            - Survey or questionnaire response variables (behavioral, lifestyle, health questions)
            - Self-reported measures (smoking, drinking, exercise, diet, medical history)
            - Demographic information (age, gender, education, income)
            - Combined with genetic markers (SNPs, rs numbers, genetic variants)
            - Participant/subject IDs linking survey responses to genetic data
            - Health outcomes or traits that can be associated with genetic variants

            This is different from pure genetic datasets - we're looking for survey data that has been augmented with genetic information.

            Task 2: If this IS GWAS data, determine whether the target variable of interest is:
            1. EXPOSURE: A phenotype/trait that we want to find genetic variants for (e.g., "find SNPs associated with insomnia")
            2. OUTCOME: A phenotype/trait that we want to predict using genetic risk scores (e.g., "predict heart disease using genetic variants")

            Instructions:
            1. First determine if this is questionnaire data with genetic components suitable for GWAS
            2. If YES, then determine if the question is about finding genetic associations (EXPOSURE) or predicting traits using genetics (OUTCOME)
            3. The target variable will be either the treatment or outcome variable already identified
            4. Return "Yes" or "No" for GWAS suitability, and if Yes, include the target variable type

            Return your response as a valid JSON object in the following format:
            {{ 
            "is_questionnaire_with_genetics": "Yes" or "No",
            "target_variable_type": "EXPOSURE" or "OUTCOME" (only if is_questionnaire_with_genetics is "Yes"),
            "confidence": YOUR_CONFIDENCE_SCORE
            }}

            Where confidence is a number between 0 and 1 indicating how certain you are about this assessment (0 = not sure at all, 1 = completely certain).
            """},
            {"role":"user", "content":"Is this questionnaire/survey data that includes genetic information suitable for GWAS analysis? If yes, is the target variable being used as an EXPOSURE (to find genetic associations) or OUTCOME (to be predicted by genetics)?"}
        ], temperature = 0.1, thinking = False
    )
    
    try:
        parsed = json.loads(gwas_response)
        gwas_data = parsed.get("is_questionnaire_with_genetics", "No")
        target_type = parsed.get("target_variable_type", None)
        gwas_confidence = parsed.get("confidence", 0.5)
        
        results['is_questionnaire_with_genetics'] = {'value': gwas_data, 'confidence': gwas_confidence}
        if gwas_data == "Yes" and target_type:
            results['gwas_target_type'] = {'value': target_type, 'confidence': gwas_confidence}
        
    except (json.JSONDecodeError, AttributeError):
        # Fallback to old parsing method
        gwas_data, gwas_confidence = parse_json_response(gwas_response, "is_questionnaire_with_genetics")
        results['is_questionnaire_with_genetics'] = {'value': gwas_data, 'confidence': gwas_confidence}

    # Override algorithm selection for GWAS data
    if results['is_questionnaire_with_genetics']['value'] == "Yes":
        # For GWAS data, use Mendelian Randomization (MR) as the appropriate causal inference method
        results['inference_algorithm'] = {
            'value': 'MR', 
            'confidence': 0.9,
            'rationale': 'GWAS data detected - using Mendelian Randomization for causal inference with genetic instruments'
        }
        print("GWAS data detected - algorithm overridden to MR (Mendelian Randomization)")

    # If this is GWAS data, query for variable mapping and OpenGWAS IDs
    if results['is_questionnaire_with_genetics']['value'] == "Yes":
        target_type = results.get('gwas_target_type', {}).get('value', 'OUTCOME')
        results.update(_query_gwas_variable_mapping(question, enhanced_context, 
                                                  results['treatment']['value'], 
                                                  results['outcome']['value'], 
                                                  target_type))
        
        # Search for OpenGWAS ID for the complementary variable only
        # If dataset has exposure data, search for outcome OpenGWAS ID (and vice versa)
        treatment_var = results['treatment']['value']
        outcome_var = results['outcome']['value']
        
        if target_type == "EXPOSURE":
            # Dataset has exposure data, search for outcome in OpenGWAS
            print(f"User data represents EXPOSURE - searching OpenGWAS for outcome variable: {outcome_var}")
            opengwas_search = _query_opengwas_id(
                variable_name=outcome_var,
                variable_description=f"Outcome variable from research question: {question}",
                variable_type="OUTCOME",
                question_context=question
            )
            results['opengwas_outcome_id'] = opengwas_search
            
        elif target_type == "OUTCOME":
            # Dataset has outcome data, search for exposure in OpenGWAS  
            print(f"User data represents OUTCOME - searching OpenGWAS for exposure variable: {treatment_var}")
            opengwas_search = _query_opengwas_id(
                variable_name=treatment_var,
                variable_description=f"Treatment/exposure variable from research question: {question}",
                variable_type="EXPOSURE", 
                question_context=question
            )
            results['opengwas_exposure_id'] = opengwas_search

    # Now query for algorithm-specific variables
    algorithm = results['inference_algorithm']['value']
    
    # Initialize empty lists and default values
    instruments = []
    mediator = []
    adjustment_set = []
    
    # DiD-specific variables
    if algorithm == "DiD":
        results.update(_query_did_variables(question, enhanced_context))
    
    # RDD-specific variables
    elif algorithm == "RDD":
        results.update(_query_rdd_variables(question, enhanced_context))
    
    # IV-specific variables
    elif algorithm == "IV":
        results.update(_query_iv_variables(question, enhanced_context, results['treatment']['value'], results['outcome']['value'], simple_context))
        # Update the instruments list for compatibility
        if results['instruments']['value'] != "None":
            for instrument in results['instruments']['value'].split(","):
                instruments.append(instrument.strip())
    
    # MR-specific variables (similar to IV but with genetic instruments)
    elif algorithm == "MR":
        # For MR, we rely on genetic instruments which are typically identified from GWAS
        # Query for potential genetic instruments
        results.update(_query_iv_variables(question, enhanced_context, results['treatment']['value'], results['outcome']['value'], simple_context))
        # Update the instruments list for compatibility
        if results['instruments']['value'] != "None":
            for instrument in results['instruments']['value'].split(","):
                instruments.append(instrument.strip())
        
        # Add specific information about MR being used
        results['mr_analysis_type'] = {
            'value': 'two_sample',
            'confidence': 0.9,
            'rationale': 'Using two-sample MR with genetic instruments from GWAS data'
        }
    
    # Frontdoor Adjustment-specific variables
    elif algorithm == "Frontdoor Adjustment":
        results.update(_query_mediator_variables(question, enhanced_context, results['treatment']['value'], results['outcome']['value'], simple_context))
        # Update the mediators list for compatibility
        if results['mediators']['value'] != "None":
            for med in results['mediators']['value'].split(","):
                mediator.append(med.strip())
    
    # Variables needed for most algorithms (OLS, Propensity Score, G Computation, DML)
    elif algorithm in ["OLS", "Propensity Score", "G Computation", "DML"]:
        results.update(_query_adjustment_variables(question, enhanced_context, results['treatment']['value'], results['outcome']['value'], simple_context))
        # Update the adjustment_set list for compatibility
        if results['adjustment_set']['value'] != "None":
            for adj in results['adjustment_set']['value'].split(","):
                adjustment_set.append(adj.strip())
    
    # If no algorithm specified or unknown, query for common variables
    if algorithm == "None" or algorithm not in ["DiD", "RDD", "IV", "MR", "Frontdoor Adjustment", "OLS", "Propensity Score", "G Computation", "DML"]:
        # Query for potential instruments to check if IV should be used
        results.update(_query_iv_variables(question, enhanced_context, results['treatment']['value'], results['outcome']['value'], simple_context))
        if results['instruments']['value'] != "None":
            for instrument in results['instruments']['value'].split(","):
                instruments.append(instrument.strip())
            results['inference_algorithm']['value'] = "IV"
        else:
            # Query for adjustment set as fallback
            results.update(_query_adjustment_variables(question, enhanced_context, results['treatment']['value'], results['outcome']['value'], simple_context))
            if results['adjustment_set']['value'] != "None":
                for adj in results['adjustment_set']['value'].split(","):
                    adjustment_set.append(adj.strip())

    # Generate causal graph (always useful)
    results.update(_query_causal_graph(question, enhanced_context, results['treatment']['value'], results['outcome']['value']))
    
    # Debug print to check algorithm selection
    if 'inference_algorithm' in results:
        print(f"DEBUG: Selected inference algorithm: {results['inference_algorithm']['value']}")
        print(f"DEBUG: Algorithm confidence: {results['inference_algorithm']['confidence']}")
        print(f"DEBUG: Algorithm rationale: {results['inference_algorithm']['rationale']}")
    
    return results

def parse_intent_legacy(question, context):
    """
    Legacy parse_intent function for backward compatibility.
    Creates basic statistical analysis and calls enhanced version.
    """
    import pandas as pd
    from .variable_analysis import analyze_variable_patterns, find_likely_relationships, detect_causal_patterns
    
    # Extract data from context (assuming it's in the format we expect)
    # Try to reconstruct the data from the context string
    lines = context.strip().split('\n')
    if len(lines) > 1:
        try:
            # Try to parse as CSV-like format
            import io
            data = pd.read_csv(io.StringIO(context))
        except:
            # Fallback: create dummy analysis
            print("Warning: Could not parse context for statistical analysis, using basic mode")
            return parse_intent_basic(question, context)
    else:
        print("Warning: No data available for statistical analysis, using basic mode")
        return parse_intent_basic(question, context)
    
    # Perform statistical analysis
    variable_analysis = analyze_variable_patterns(data)
    relationships = find_likely_relationships(data)
    causal_patterns = detect_causal_patterns(data, variable_analysis, relationships)
    
    # Call enhanced version
    return parse_intent(question, context, data, variable_analysis, relationships, causal_patterns)

def parse_intent_basic(question, context):
    """Fallback version without statistical analysis for backward compatibility"""
    # Dictionary to store all value-confidence pairs
    results = {}

    # Core variables needed for all analyses
    treatment_response = create_chat_completion(
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
            3. Return the exact column name from the table that is the treatment variable.
            4. Do NOT return any instrument variables.
            
            Return your response as a valid JSON object in the following format:
            {{ 
            "treatment_variable": "COLUMN_NAME",
            "confidence": YOUR_CONFIDENCE_SCORE
            }}
            
            Where confidence is a number between 0 and 1 indicating how certain you are about this identification (0 = not sure at all, 1 = completely certain)."""},
            {"role": "user", "content": question}
        ], temperature = 0.1, thinking = False
    )
    treatment, treatment_confidence = parse_json_response(treatment_response, "treatment_variable")
    results['treatment'] = {'value': treatment, 'confidence': treatment_confidence}

    outcome_response = create_chat_completion(
        messages = [
            {"role": "system", "content": f"""
            You are given a CSV table containing variable names and sample data:

            {context}

            Your task is to identify the **outcome variable**: the variable that is influenced, affected, or predicted in the user's question. This is the result of a causal or correlational relationship — it's the effect, not the cause.

            Key concepts:
            - The outcome is the variable whose value changes *as a result of* the treatment (or predictor).
            - It may also be described as the variable we are trying to explain or model in the question.

            Important Rules:
            - Only identify the exact column name from the data that best matches the outcome variable.
            - Do not include multiple column names, explanations, comments, or instrument variables.
            - Ignore how the treatment is estimated (e.g. via instruments); focus only on what the treatment is affecting.

            Example:
            If the question is: "What is the effect of exercise frequency on blood pressure, using distance to gym as an instrument?"
            → The outcome is the column related to blood pressure

            Return your response as a valid JSON object in the following format:
            {{ 
            "outcome_variable": "COLUMN_NAME",
            "confidence": YOUR_CONFIDENCE_SCORE
            }}
            
            Where confidence is a number between 0 and 1 indicating how certain you are about this identification (0 = not sure at all, 1 = completely certain).
            """},
            {"role": "user", "content": question}
        ], temperature = 0.1, thinking = False
    )
    outcome, outcome_confidence = parse_json_response(outcome_response, "outcome_variable")
    results['outcome'] = {'value': outcome, 'confidence': outcome_confidence}
    
    # Simplified algorithm detection
    results['inference_algorithm'] = {'value': 'OLS', 'confidence': 0.5, 'rationale': 'Default algorithm selection'}
    
    return results

def search_opengwas_for_trait(trait_name, trait_description="", trait_type="OUTCOME", context=""):
    """
    Standalone function to search for OpenGWAS IDs for any trait
    
    Args:
        trait_name (str): Name of the trait/phenotype to search for
        trait_description (str): Optional description of the trait
        trait_type (str): "EXPOSURE" or "OUTCOME" 
        context (str): Optional research context
        
    Returns:
        dict: OpenGWAS search results with ID, confidence, and metadata
    """
    
    full_description = trait_description if trait_description else f"Searching for GWAS data for trait: {trait_name}"
    search_context = context if context else f"General search for {trait_type.lower()} trait: {trait_name}"
    
    return _query_opengwas_id(
        variable_name=trait_name,
        variable_description=full_description, 
        variable_type=trait_type,
        question_context=search_context
    )

def get_complementary_opengwas_ids(results_dict):
    """
    Extract OpenGWAS IDs from parse_intent results for use in MR analysis
    
    Args:
        results_dict (dict): Results from parse_intent function
        
    Returns:
        dict: Structured OpenGWAS IDs for exposure and outcome
    """
    
    opengwas_ids = {
        'exposure_id': None,
        'outcome_id': None,
        'search_performed': False,
        'gwas_data_detected': False,
        'user_data_type': None  # Track what type of data the user has
    }
    
    # Check if GWAS data was detected
    if results_dict.get('is_questionnaire_with_genetics', {}).get('value') == "Yes":
        opengwas_ids['gwas_data_detected'] = True
        opengwas_ids['search_performed'] = True
        opengwas_ids['user_data_type'] = results_dict.get('gwas_target_type', {}).get('value', 'UNKNOWN')
        
        # Extract found OpenGWAS IDs (only one will be present - the complementary one)
        if 'opengwas_exposure_id' in results_dict:
            opengwas_ids['exposure_id'] = results_dict['opengwas_exposure_id']['opengwas_id']
            
        if 'opengwas_outcome_id' in results_dict:
            opengwas_ids['outcome_id'] = results_dict['opengwas_outcome_id']['opengwas_id']
    
    return opengwas_ids

def format_opengwas_search_summary(results_dict):
    """
    Create a human-readable summary of OpenGWAS search results
    
    Args:
        results_dict (dict): Results from parse_intent function
        
    Returns:
        str: Formatted summary of OpenGWAS findings
    """
    
    if results_dict.get('is_questionnaire_with_genetics', {}).get('value') != "Yes":
        return "No GWAS data detected - OpenGWAS search not performed."
    
    summary_lines = ["=== OpenGWAS Search Results ==="]
    
    # Check for exposure ID
    if 'opengwas_exposure_id' in results_dict:
        exp_data = results_dict['opengwas_exposure_id']
        summary_lines.extend([
            f"Exposure OpenGWAS ID: {exp_data['opengwas_id']}",
            f"  Study: {exp_data['study_description']}",
            f"  Population: {exp_data['population']}",
            f"  Sample Size: {exp_data['sample_size_estimate']}",
            f"  Confidence: {exp_data['confidence']:.2f}",
            f"  Reasoning: {exp_data['reasoning']}"
        ])
        
        if exp_data['alternative_ids']:
            summary_lines.append(f"  Alternatives: {', '.join(exp_data['alternative_ids'])}")
    
    # Check for outcome ID  
    if 'opengwas_outcome_id' in results_dict:
        out_data = results_dict['opengwas_outcome_id']
        summary_lines.extend([
            f"Outcome OpenGWAS ID: {out_data['opengwas_id']}",
            f"  Study: {out_data['study_description']}",
            f"  Population: {out_data['population']}",
            f"  Sample Size: {out_data['sample_size_estimate']}",
            f"  Confidence: {out_data['confidence']:.2f}",
            f"  Reasoning: {out_data['reasoning']}"
        ])
        
        if out_data['alternative_ids']:
            summary_lines.append(f"  Alternatives: {', '.join(out_data['alternative_ids'])}")
    
    # Add information about what type of data the user has
    user_data_type = results_dict.get('gwas_target_type', {}).get('value', 'UNKNOWN')
    if user_data_type != 'UNKNOWN':
        summary_lines.append(f"\nUser dataset represents: {user_data_type}")
        complementary_type = "OUTCOME" if user_data_type == "EXPOSURE" else "EXPOSURE"
        summary_lines.append(f"Searched OpenGWAS for: {complementary_type}")
    
    summary_lines.append("\nThis ID can be used for two-sample Mendelian Randomization analysis.")
    
    return "\n".join(summary_lines)
