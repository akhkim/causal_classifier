import pandas as pd
from causal_classifier.llm_query import parse_intent
from causal_classifier.inference_classifier import run_inference_algorithm, match_algorithm
from causal_classifier.discovery_classifier import run_discovery_algorithm
from causal_classifier.preprocessing import full_preprocess
from causal_classifier.variable_analysis import analyze_variable_patterns, find_likely_relationships, detect_causal_patterns
from causal_classifier.interactive_selection import interactive_variable_selection, display_analysis_summary

# Variables the user has to provide
data = pd.read_csv('C:\\Users\\Andrew\\Downloads\\Causal-Copilot-main\\simulated_data\\nodes10_samples500\\data_500.csv')
question = "Find all the causal relationships in the data"     # "How does the education affect earnings?"
assignment_style = 'observational' # 'observational', 'randomized'
latent_confounders = False  # Set to True if latent confounders are present

# Enhanced preprocessing and statistical analysis
numeric_data = full_preprocess(data)

# Create simple tabular context for LLM (traditional approach)
context = numeric_data.head(3).to_string(index=False)

# Perform statistical analysis to understand variable patterns
print("Performing statistical analysis of variables...")
variable_analysis = analyze_variable_patterns(numeric_data)
relationships = find_likely_relationships(numeric_data)
causal_patterns = detect_causal_patterns(numeric_data, variable_analysis, relationships)

# Display analysis summary
display_analysis_summary(variable_analysis, relationships, causal_patterns)

# Enhanced LLM parsing with statistical context
print("\nAnalyzing causal structure with enhanced context...")
results = parse_intent(question, context, numeric_data, variable_analysis, relationships, causal_patterns)

# Check confidence levels - if low, offer interactive selection
min_confidence = min(
    results['treatment']['confidence'],
    results['outcome']['confidence'],
    results['inference_algorithm']['confidence']
)

print(f"\nConfidence scores: Treatment={results['treatment']['confidence']:.2f}, "
      f"Outcome={results['outcome']['confidence']:.2f}, "
      f"Algorithm={results['inference_algorithm']['confidence']:.2f}")

if min_confidence < 0.7:  # Threshold for low confidence
    print(f"\nLow confidence detected (min: {min_confidence:.2f})")
    use_interactive = input("Use interactive variable selection? (y/n): ").lower() == 'y'
    
    if use_interactive:
        results = interactive_variable_selection(numeric_data, variable_analysis, relationships, causal_patterns)

# Extract values from the dictionary structure
treatment = results['treatment']['value']
outcome = results['outcome']['value']
inference_algorithm = results['inference_algorithm']['value']

# Extract algorithm-specific variables if they exist
time_variable = results.get('time_variable', {}).get('value', 'None')
group_variable = results.get('group_variable', {}).get('value', 'None')
running_variable = results.get('running_variable', {}).get('value', 'None')
cutoff_value = results.get('cutoff_value', {}).get('value', 'None')

# Handle instruments, mediators, and adjustment_set
instruments_str = results.get('instruments', {}).get('value', 'None')
instruments = []
if instruments_str != 'None':
    instruments = [inst.strip() for inst in instruments_str.split(',')]

mediators_str = results.get('mediators', {}).get('value', 'None')
mediators = []
if mediators_str != 'None':
    mediators = [med.strip() for med in mediators_str.split(',')]

adjustment_set_str = results.get('adjustment_set', {}).get('value', 'None')
adjustment_set = []
if adjustment_set_str != 'None':
    adjustment_set = [adj.strip() for adj in adjustment_set_str.split(',')]

graph = results.get('graph', {}).get('value', 'None')

print("treatment:", treatment, f"(confidence: {results['treatment']['confidence']:.2f})")
print("outcome:", outcome, f"(confidence: {results['outcome']['confidence']:.2f})")
print("time variable:", time_variable, f"(confidence: {results.get('time_variable', {}).get('confidence', 0):.2f})" if 'time_variable' in results else "")
print("group variable:", group_variable, f"(confidence: {results.get('group_variable', {}).get('confidence', 0):.2f})" if 'group_variable' in results else "")
print("inference algorithm:", inference_algorithm, f"(confidence: {results['inference_algorithm']['confidence']:.2f})")

# Print instruments with individual confidence scores
if 'instruments' in results:
    print(f"instruments: {instruments} (overall confidence: {results['instruments']['confidence']:.2f})")
    if 'individual_confidences' in results['instruments']:
        print("  Individual instrument confidences:")
        for inst, conf in results['instruments']['individual_confidences'].items():
            print(f"    {inst}: {conf:.2f}")

# Print mediators with individual confidence scores  
if 'mediators' in results:
    print(f"mediators: {mediators} (overall confidence: {results['mediators']['confidence']:.2f})")
    if 'individual_confidences' in results['mediators']:
        print("  Individual mediator confidences:")
        for med, conf in results['mediators']['individual_confidences'].items():
            print(f"    {med}: {conf:.2f}")

# Print adjustment set with individual confidence scores
if 'adjustment_set' in results:
    print(f"adjustment_set: {adjustment_set} (overall confidence: {results['adjustment_set']['confidence']:.2f})")
    if 'individual_confidences' in results['adjustment_set']:
        print("  Individual confounder confidences:")
        for adj, conf in results['adjustment_set']['individual_confidences'].items():
            print(f"    {adj}: {conf:.2f}")

nx_graph, graph_suggestions = run_discovery_algorithm(numeric_data, latent_confounders, variable_analysis, use_gpu=True)
sample_size = numeric_data.shape[0]

if graph_suggestions:
    print(f"\nGraph-based causal suggestions:")
    for suggestion in graph_suggestions:
        print(f"- {suggestion['treatment']} → {suggestion['outcome']} (confidence: {suggestion['confidence']:.2f})")
        print(f"  Reasoning: {suggestion['reasoning']}")

# Filter out latent confounders (U_ nodes) from covariates since they're not measured variables
all_nodes = set(nx_graph.nodes)
latent_nodes = {node for node in all_nodes if str(node).startswith('U_')}
measured_nodes = all_nodes - latent_nodes - {treatment, outcome}

# Covariates should only include measured variables
covariates = list(measured_nodes)

print(f"All graph nodes: {all_nodes}")
print(f"Latent confounders found: {latent_nodes}")
print(f"Measured covariates: {covariates}")

for edge in nx_graph.edges():
    print(edge)

# Update latent_confounders flag if FCI found any
if latent_nodes:
    latent_confounders = True
    print(f"Updated latent_confounders flag to True based on {len(latent_nodes)} latent confounders found by FCI")

if not inference_algorithm:
    result = run_inference_algorithm(
        numeric_data,
        treatment,
        outcome,
        covariates,
        nx_graph,
        sample_size,
        assignment_style,
        latent_confounders,
        cutoff_value,
        time_variable,
        group_variable,
        running_variable,
        llm_results=results
)
else:
    result = match_algorithm(inference_algorithm, numeric_data, treatment, outcome, covariates, sample_size,
                    cutoff_value, time_variable, group_variable, running_variable, 
                    mediators, instruments, adjustment_set, llm_results=results)

print(result)