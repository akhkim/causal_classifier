import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def analyze_variable_patterns(data):
    """Analyze statistical patterns to infer variable types and relationships"""
    analysis = {}
    
    for col in data.columns:
        col_analysis = {
            'type': 'continuous' if data[col].dtype in ['int64', 'float64'] else 'categorical',
            'unique_values': data[col].nunique(),
            'unique_ratio': data[col].nunique() / len(data),
            'missing_ratio': data[col].isnull().sum() / len(data),
            'distribution_type': None,
            'potential_roles': [],
            'variance': None,
            'skewness': None,
            'kurtosis': None
        }
        
        # Binary variable detection
        if col_analysis['unique_values'] == 2:
            col_analysis['potential_roles'].append('binary_treatment')
            unique_vals = data[col].dropna().unique()
            if set(unique_vals).issubset({0, 1}):
                col_analysis['potential_roles'].append('binary_indicator')
                
        # Time variable detection (increasing integers, dates, etc.)
        if data[col].dtype in ['int64', 'float64']:
            if data[col].is_monotonic_increasing or data[col].is_monotonic_decreasing:
                col_analysis['potential_roles'].append('time_variable')
            
            # Check for year-like patterns
            if data[col].min() > 1900 and data[col].max() < 2100:
                col_analysis['potential_roles'].append('year_variable')
                
            # Calculate statistical measures
            col_analysis['variance'] = data[col].var()
            col_analysis['skewness'] = data[col].skew()
            col_analysis['kurtosis'] = data[col].kurtosis()
                
        # ID variable detection (high uniqueness)
        if col_analysis['unique_ratio'] > 0.9:
            col_analysis['potential_roles'].append('identifier')
            
        # Categorical with few levels (potential grouping variable)
        if col_analysis['type'] == 'categorical' or (col_analysis['unique_values'] <= 10 and col_analysis['unique_values'] > 2):
            col_analysis['potential_roles'].append('group_variable')
            
        # Age-like variable detection
        if data[col].dtype in ['int64', 'float64']:
            if 0 <= data[col].min() <= 100 and 0 <= data[col].max() <= 120:
                col_analysis['potential_roles'].append('age_variable')
        
        # Income/wage-like variable detection (high positive skew, wide range)
        if data[col].dtype in ['int64', 'float64'] and col_analysis['skewness'] and col_analysis['skewness'] > 2:
            if data[col].min() >= 0 and data[col].max() / data[col].mean() > 10:
                col_analysis['potential_roles'].append('income_variable')
        
        # Education-like variable detection (ordered, small range)
        if data[col].dtype in ['int64', 'float64']:
            if 5 <= col_analysis['unique_values'] <= 25 and data[col].min() >= 0:
                col_analysis['potential_roles'].append('education_variable')
        
        # Distribution analysis for continuous variables
        if col_analysis['type'] == 'continuous' and data[col].notna().sum() > 10:
            try:
                _, p_value = stats.normaltest(data[col].dropna())
                col_analysis['distribution_type'] = 'normal' if p_value > 0.05 else 'non_normal'
            except:
                col_analysis['distribution_type'] = 'unknown'
                
        analysis[col] = col_analysis
    
    return analysis

def find_likely_relationships(data, max_pairs=20):
    """Find variable pairs with strong statistical relationships"""
    relationships = []
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    # Calculate mutual information for all pairs
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            if len(relationships) >= max_pairs:
                break
                
            try:
                # Calculate correlation and mutual information
                corr = data[col1].corr(data[col2])
                
                # Mutual information (handle missing values)
                clean_data = data[[col1, col2]].dropna()
                if len(clean_data) > 10:
                    mi = mutual_info_regression(clean_data[[col1]], clean_data[col2])[0]
                    relationships.append({
                        'var1': col1,
                        'var2': col2,
                        'correlation': abs(corr) if not pd.isna(corr) else 0,
                        'mutual_info': mi,
                        'strength': (abs(corr) if not pd.isna(corr) else 0) + mi
                    })
            except:
                continue
    
    return sorted(relationships, key=lambda x: x['strength'], reverse=True)

def detect_causal_patterns(data, variable_analysis, relationships):
    """Detect likely causal patterns based on statistical analysis"""
    patterns = {
        'likely_treatments': [],
        'likely_outcomes': [],
        'likely_confounders': [],
        'likely_instruments': [],
        'causal_suggestions': []
    }
    
    # Identify likely treatments (binary variables, experimental indicators)
    for var, analysis in variable_analysis.items():
        if 'binary_treatment' in analysis['potential_roles'] or 'binary_indicator' in analysis['potential_roles']:
            patterns['likely_treatments'].append({
                'variable': var,
                'confidence': 0.7 if 'binary_treatment' in analysis['potential_roles'] else 0.5,
                'reasoning': 'Binary variable, potential treatment indicator'
            })
    
    # Identify likely outcomes (variables with many relationships, income-like variables)
    for var, analysis in variable_analysis.items():
        confidence = 0.0
        reasons = []
        
        if 'income_variable' in analysis['potential_roles']:
            confidence += 0.6
            reasons.append('Income-like distribution')
            
        # Count how many strong relationships this variable has
        strong_relationships = sum(1 for rel in relationships[:10] if var in [rel['var1'], rel['var2']])
        if strong_relationships >= 3:
            confidence += 0.4
            reasons.append(f'Connected to {strong_relationships} other variables')
            
        if confidence > 0.5:
            patterns['likely_outcomes'].append({
                'variable': var,
                'confidence': min(confidence, 0.9),
                'reasoning': '; '.join(reasons)
            })
    
    # Identify likely confounders (age, education-like variables)
    for var, analysis in variable_analysis.items():
        confidence = 0.0
        reasons = []
        
        if 'age_variable' in analysis['potential_roles']:
            confidence += 0.8
            reasons.append('Age-like variable')
            
        if 'education_variable' in analysis['potential_roles']:
            confidence += 0.7
            reasons.append('Education-like variable')
            
        if confidence > 0.6:
            patterns['likely_confounders'].append({
                'variable': var,
                'confidence': confidence,
                'reasoning': '; '.join(reasons)
            })
    
    # Generate causal suggestions based on patterns
    for treatment in patterns['likely_treatments'][:3]:
        for outcome in patterns['likely_outcomes'][:3]:
            if treatment['variable'] != outcome['variable']:
                # Find potential confounders for this pair
                pair_confounders = []
                for confounder in patterns['likely_confounders']:
                    # Check if confounder is related to both treatment and outcome
                    related_to_treatment = any(
                        (rel['var1'] == treatment['variable'] and rel['var2'] == confounder['variable']) or
                        (rel['var2'] == treatment['variable'] and rel['var1'] == confounder['variable'])
                        for rel in relationships[:15]
                    )
                    related_to_outcome = any(
                        (rel['var1'] == outcome['variable'] and rel['var2'] == confounder['variable']) or
                        (rel['var2'] == outcome['variable'] and rel['var1'] == confounder['variable'])
                        for rel in relationships[:15]
                    )
                    
                    if related_to_treatment and related_to_outcome:
                        pair_confounders.append(confounder['variable'])
                
                patterns['causal_suggestions'].append({
                    'treatment': treatment['variable'],
                    'outcome': outcome['variable'],
                    'treatment_confidence': treatment['confidence'],
                    'outcome_confidence': outcome['confidence'],
                    'suggested_confounders': pair_confounders,
                    'overall_confidence': (treatment['confidence'] + outcome['confidence']) / 2
                })
    
    # Sort suggestions by confidence
    patterns['causal_suggestions'] = sorted(
        patterns['causal_suggestions'], 
        key=lambda x: x['overall_confidence'], 
        reverse=True
    )
    
    return patterns

def create_enhanced_context(data, variable_analysis, relationships, causal_patterns):
    """Create rich context including statistical patterns for LLM"""
    # Start with simple tabular context (traditional approach)
    simple_context = data.head(3).to_string(index=False)
    
    context = f"Dataset shape: {data.shape[0]} rows, {data.shape[1]} columns\n\n"
    
    # Add simple tabular view first
    context += "=== SAMPLE DATA ===\n"
    context += f"{simple_context}\n\n"
    
    # Add variable analysis
    context += "=== VARIABLE ANALYSIS ===\n"
    for var, analysis in variable_analysis.items():
        roles = ", ".join(analysis['potential_roles']) if analysis['potential_roles'] else "general variable"
        context += f"• {var}: {analysis['type']}, {analysis['unique_values']} unique values"
        if analysis['unique_ratio'] < 0.1:
            context += f" ({analysis['unique_ratio']:.1%} unique)"
        context += f", likely roles: {roles}\n"
        
        # Add statistical details for continuous variables
        if analysis['type'] == 'continuous' and analysis['variance'] is not None:
            context += f"  Stats: variance={analysis['variance']:.2f}"
            if analysis['skewness'] is not None:
                context += f", skew={analysis['skewness']:.2f}"
            context += "\n"
    
    # Add causal pattern suggestions
    if causal_patterns['causal_suggestions']:
        context += f"\n=== SUGGESTED CAUSAL RELATIONSHIPS ===\n"
        for i, suggestion in enumerate(causal_patterns['causal_suggestions'][:3], 1):
            context += f"{i}. {suggestion['treatment']} → {suggestion['outcome']} "
            context += f"(confidence: {suggestion['overall_confidence']:.2f})\n"
            if suggestion['suggested_confounders']:
                context += f"   Potential confounders: {', '.join(suggestion['suggested_confounders'])}\n"
    
    # Add top relationships
    context += f"\n=== STRONGEST VARIABLE RELATIONSHIPS ===\n"
    for i, rel in enumerate(relationships[:5], 1):
        context += f"{i}. {rel['var1']} ↔ {rel['var2']}: "
        context += f"correlation={rel['correlation']:.3f}, mutual_info={rel['mutual_info']:.3f}\n"
    
    return context

def suggest_causal_pairs_from_graph(graph, variable_analysis):
    """Suggest likely treatment-outcome pairs based on graph structure"""
    suggestions = []
    
    if graph is None:
        return suggestions
    
    # Look for binary variables with high out-degree (potential treatments)
    potential_treatments = []
    for node in graph.nodes():
        if str(node) in variable_analysis:
            analysis = variable_analysis[str(node)]
            if 'binary_treatment' in analysis['potential_roles'] or 'binary_indicator' in analysis['potential_roles']:
                out_degree = graph.out_degree(node)
                potential_treatments.append((str(node), out_degree))
    
    # Look for variables with high in-degree (potential outcomes)
    potential_outcomes = []
    for node in graph.nodes():
        in_degree = graph.in_degree(node)
        if in_degree > 1:  # Influenced by multiple variables
            potential_outcomes.append((str(node), in_degree))
    
    # Create suggestions
    for treatment, out_deg in sorted(potential_treatments, key=lambda x: x[1], reverse=True)[:3]:
        for outcome, in_deg in sorted(potential_outcomes, key=lambda x: x[1], reverse=True)[:3]:
            if treatment != outcome:
                # Calculate confidence based on graph structure and variable analysis
                confidence = min(0.9, (out_deg + in_deg) / 10)
                if treatment in variable_analysis and 'binary_treatment' in variable_analysis[treatment]['potential_roles']:
                    confidence += 0.2
                
                suggestions.append({
                    'treatment': treatment,
                    'outcome': outcome,
                    'confidence': min(confidence, 0.95),
                    'reasoning': f'Graph structure: {treatment} has {out_deg} outgoing edges, {outcome} has {in_deg} incoming edges'
                })
    
    return suggestions[:5]  # Top 5 suggestions
