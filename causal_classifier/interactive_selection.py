import pandas as pd

def interactive_variable_selection(data, variable_analysis, relationships, causal_patterns):
    """Interactive fallback when LLM confidence is low"""
    print("\n" + "="*60)
    print("INTERACTIVE VARIABLE SELECTION")
    print("="*60)
    print("The automatic detection had low confidence. Please help identify variables.\n")
    
    # Display variable analysis in a nice format
    print("VARIABLE ANALYSIS:")
    print("-" * 40)
    for var, analysis in variable_analysis.items():
        roles = ", ".join(analysis['potential_roles']) if analysis['potential_roles'] else "general variable"
        print(f"• {var:15} | {analysis['type']:12} | {analysis['unique_values']:3} unique | {roles}")
    
    # Display causal suggestions if any
    if causal_patterns['causal_suggestions']:
        print(f"\nSUGGESTED CAUSAL RELATIONSHIPS:")
        print("-" * 40)
        for i, suggestion in enumerate(causal_patterns['causal_suggestions'][:3], 1):
            print(f"{i}. {suggestion['treatment']} → {suggestion['outcome']} (confidence: {suggestion['overall_confidence']:.2f})")
            if suggestion['suggested_confounders']:
                print(f"   Potential confounders: {', '.join(suggestion['suggested_confounders'])}")
    
    # Display top relationships
    if relationships:
        print(f"\nSTRONGEST VARIABLE RELATIONSHIPS:")
        print("-" * 40)
        for i, rel in enumerate(relationships[:5], 1):
            print(f"{i}. {rel['var1']} ↔ {rel['var2']} (strength: {rel['strength']:.3f})")
    
    print(f"\nAvailable variables: {', '.join(data.columns)}")
    print("\n" + "="*60)
    
    # Interactive selection with suggestions
    results = {}
    
    # Treatment selection
    print("\n1. TREATMENT VARIABLE SELECTION:")
    if causal_patterns['likely_treatments']:
        print("   Suggested treatments based on analysis:")
        for i, treatment in enumerate(causal_patterns['likely_treatments'][:3], 1):
            print(f"   {i}. {treatment['variable']} (confidence: {treatment['confidence']:.2f}) - {treatment['reasoning']}")
        
        use_suggestion = input(f"\n   Use suggestion 1 ({causal_patterns['likely_treatments'][0]['variable']})? (y/n): ").lower()
        if use_suggestion == 'y':
            treatment = causal_patterns['likely_treatments'][0]['variable']
        else:
            treatment = input("   Enter treatment variable name: ").strip()
    else:
        treatment = input("   Enter treatment variable name: ").strip()
    
    results['treatment'] = {'value': treatment, 'confidence': 1.0}
    
    # Outcome selection
    print("\n2. OUTCOME VARIABLE SELECTION:")
    if causal_patterns['likely_outcomes']:
        print("   Suggested outcomes based on analysis:")
        for i, outcome in enumerate(causal_patterns['likely_outcomes'][:3], 1):
            print(f"   {i}. {outcome['variable']} (confidence: {outcome['confidence']:.2f}) - {outcome['reasoning']}")
        
        # Filter out the selected treatment
        available_outcomes = [o for o in causal_patterns['likely_outcomes'] if o['variable'] != treatment]
        if available_outcomes:
            use_suggestion = input(f"\n   Use suggestion 1 ({available_outcomes[0]['variable']})? (y/n): ").lower()
            if use_suggestion == 'y':
                outcome = available_outcomes[0]['variable']
            else:
                outcome = input("   Enter outcome variable name: ").strip()
        else:
            outcome = input("   Enter outcome variable name: ").strip()
    else:
        outcome = input("   Enter outcome variable name: ").strip()
    
    results['outcome'] = {'value': outcome, 'confidence': 1.0}
    
    # Optional variables
    print("\n3. OPTIONAL VARIABLES:")
    
    # Time variable
    time_candidates = [var for var, analysis in variable_analysis.items() 
                      if 'time_variable' in analysis['potential_roles'] or 'year_variable' in analysis['potential_roles']]
    if time_candidates:
        print(f"   Suggested time variables: {', '.join(time_candidates)}")
        time_var = input("   Enter time variable (or press Enter to skip): ").strip()
    else:
        time_var = input("   Enter time variable (or press Enter to skip): ").strip()
    
    if time_var:
        results['time_variable'] = {'value': time_var, 'confidence': 1.0}
    
    # Group variable
    group_candidates = [var for var, analysis in variable_analysis.items() 
                       if 'group_variable' in analysis['potential_roles']]
    if group_candidates:
        print(f"   Suggested group variables: {', '.join(group_candidates)}")
        group_var = input("   Enter group variable (or press Enter to skip): ").strip()
    else:
        group_var = input("   Enter group variable (or press Enter to skip): ").strip()
    
    if group_var:
        results['group_variable'] = {'value': group_var, 'confidence': 1.0}
    
    # Running variable for RDD
    running_var = input("   Enter running variable for RDD (or press Enter to skip): ").strip()
    if running_var:
        results['running_variable'] = {'value': running_var, 'confidence': 1.0}
        cutoff = input("   Enter cutoff value for running variable: ").strip()
        if cutoff:
            results['cutoff_value'] = {'value': cutoff, 'confidence': 1.0}
    
    # Confounders
    print("\n4. CONFOUNDERS/ADJUSTMENT SET:")
    if causal_patterns['likely_confounders']:
        print("   Suggested confounders based on analysis:")
        for conf in causal_patterns['likely_confounders']:
            print(f"   • {conf['variable']} (confidence: {conf['confidence']:.2f}) - {conf['reasoning']}")
        
        use_suggested = input("   Use suggested confounders? (y/n): ").lower()
        if use_suggested == 'y':
            confounders = [conf['variable'] for conf in causal_patterns['likely_confounders']]
        else:
            print("   Enter potential confounders (comma-separated, or press Enter to skip):")
            confounders_input = input("   ").strip()
            confounders = [c.strip() for c in confounders_input.split(',')] if confounders_input else []
    else:
        print("   Enter potential confounders (comma-separated, or press Enter to skip):")
        confounders_input = input("   ").strip()
        confounders = [c.strip() for c in confounders_input.split(',')] if confounders_input else []
    
    if confounders:
        results['adjustment_set'] = {'value': ','.join(confounders), 'confidence': 1.0}
    
    # Instruments
    instruments_input = input("\n   Enter instrumental variables (comma-separated, or press Enter to skip): ").strip()
    if instruments_input:
        instruments = [i.strip() for i in instruments_input.split(',')]
        results['instruments'] = {'value': ','.join(instruments), 'confidence': 1.0}
    
    # Mediators
    mediators_input = input("   Enter mediator variables (comma-separated, or press Enter to skip): ").strip()
    if mediators_input:
        mediators = [m.strip() for m in mediators_input.split(',')]
        results['mediators'] = {'value': ','.join(mediators), 'confidence': 1.0}
    
    # Inference algorithm suggestion
    print("\n5. INFERENCE ALGORITHM:")
    print("   Leave empty to let the system choose automatically")
    algorithm = input("   Enter preferred algorithm (or press Enter for auto): ").strip()
    if algorithm:
        results['inference_algorithm'] = {'value': algorithm, 'confidence': 1.0}
    else:
        results['inference_algorithm'] = {'value': None, 'confidence': 0.0}
    
    print("\n" + "="*60)
    print("Selection complete!")
    print("="*60)
    
    return results

def display_analysis_summary(variable_analysis, relationships, causal_patterns):
    """Display a summary of the statistical analysis for user review"""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("="*60)
    
    # Variable types summary
    type_counts = {}
    role_counts = {}
    
    for var, analysis in variable_analysis.items():
        var_type = analysis['type']
        type_counts[var_type] = type_counts.get(var_type, 0) + 1
        
        for role in analysis['potential_roles']:
            role_counts[role] = role_counts.get(role, 0) + 1
    
    print("VARIABLE TYPES:")
    for var_type, count in type_counts.items():
        print(f"  {var_type}: {count}")
    
    if role_counts:
        print("\nPOTENTIAL ROLES DETECTED:")
        for role, count in sorted(role_counts.items()):
            print(f"  {role}: {count}")
    
    # Top causal suggestions
    if causal_patterns['causal_suggestions']:
        print(f"\nTOP CAUSAL RELATIONSHIP SUGGESTIONS:")
        for i, suggestion in enumerate(causal_patterns['causal_suggestions'][:3], 1):
            print(f"  {i}. {suggestion['treatment']} → {suggestion['outcome']} (confidence: {suggestion['overall_confidence']:.2f})")
    
    print("="*60)
