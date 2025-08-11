import matplotlib.pyplot as plt
import ieugwaspy.query as query
import pandas as pd
import genal
from genal.tools import set_plink
import numpy as np
import re
from scipy import stats
import sys
import os
from pathlib import Path

# Import OpenGWAS helper functions
try:
    from ..llm_query import search_opengwas_for_trait
    OPENGWAS_SEARCH_AVAILABLE = True
except ImportError:
    print("⚠️ Could not import OpenGWAS search functions from llm_query")
    OPENGWAS_SEARCH_AVAILABLE = False

plt.style.use('seaborn-v0_8')

# Try to set PLINK path if it exists
plink_path = r"C:\Personal\Applications\PLINK2\plink2.exe"
try:
    if os.path.exists(plink_path):
        set_plink(plink_path)
        print(f"✓ PLINK configured at {plink_path}")
    else:
        print(f"⚠ PLINK not found at {plink_path}, some features may be limited")
except Exception as e:
    print(f"⚠ Could not configure PLINK: {e}")
    print("Continuing without PLINK - some genal features may be limited")

# Lazy import PyGWAS for integrated GWAS analysis - imported only when needed
PYGWAS_AVAILABLE = None  # None means not yet checked

def _check_pygwas_availability():
    """Check if PyGWAS is available for import"""
    global PYGWAS_AVAILABLE
    
    if PYGWAS_AVAILABLE is not None:
        return PYGWAS_AVAILABLE
    
    try:
        # Add pygwas to path if not already added
        pygwas_path = Path(__file__).parent.parent / "pygwas"
        if str(pygwas_path) not in sys.path:
            sys.path.insert(0, str(pygwas_path))
        
        # Check if PyGWAS directory exists first
        if pygwas_path.exists():
            print(f"Attempting to import PyGWAS from {pygwas_path}")
            from pygwas import GWAS
            PYGWAS_AVAILABLE = True
            print("✓ PyGWAS available for integrated MR analysis")
        else:
            print(f"⚠ PyGWAS directory not found at {pygwas_path}")
            print("MR analysis will use external GWAS databases only")
            PYGWAS_AVAILABLE = False
            
    except ImportError as e:
        print(f"⚠ PyGWAS not available: {e}")
        print("MR analysis will use external GWAS databases only")
        PYGWAS_AVAILABLE = False
    except Exception as e:
        print(f"⚠ Unexpected error importing PyGWAS: {e}")
        print("MR analysis will use external GWAS databases only")
        PYGWAS_AVAILABLE = False
    
    return PYGWAS_AVAILABLE

def construct_phenotype_from_gwas_mapping(df, gwas_mapping, output_prefix=None):
    """
    Construct phenotype and covariate files from GWAS variable mapping information.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset containing questionnaire responses
    gwas_mapping : dict
        GWAS variable mapping from LLM query containing:
        - target_phenotype: name of the target phenotype
        - questionnaire_fields: dict mapping field names to categories and confidence
        - phenotype_construction: method and logic for construction
        - covariates: list of covariate field names
    output_prefix : str, optional
        Prefix for output files. If None, uses target_phenotype name
    
    Returns:
    --------
    dict : Contains constructed phenotype info, file paths, and construction summary
    """
    
    target_phenotype = gwas_mapping['target_phenotype']
    questionnaire_fields = gwas_mapping['questionnaire_fields']
    phenotype_construction = gwas_mapping['phenotype_construction']
    covariates = gwas_mapping['covariates']
    
    if output_prefix is None:
        output_prefix = target_phenotype.lower().replace(' ', '_')
    
    print(f"Constructing GWAS phenotype: {target_phenotype}")
    print(f"Construction method: {phenotype_construction.get('method', 'unknown')}")
    print(f"Number of questionnaire fields: {len(questionnaire_fields)}")
    
    # Prepare working dataframe
    work_df = df.copy()
    
    # Ensure we have individual IDs (assume first column or look for common ID names)
    id_candidates = ['IID', 'eid', 'ID', 'subject_id', 'participant_id']
    id_column = None
    
    for candidate in id_candidates:
        if candidate in work_df.columns:
            id_column = candidate
            break
    
    if id_column is None:
        # Use index as ID if no standard ID column found
        work_df['IID'] = work_df.index
        id_column = 'IID'
        print("Warning: No standard ID column found, using row index as IID")
    elif id_column != 'IID':
        work_df['IID'] = work_df[id_column]
    
    # Group fields by category for phenotype construction
    field_categories = {}
    for field_name, field_info in questionnaire_fields.items():
        if field_name in work_df.columns:
            category = field_info['category']
            if category not in field_categories:
                field_categories[category] = []
            field_categories[category].append(field_name)
        else:
            print(f"Warning: Field '{field_name}' not found in dataset")
    
    # Construct phenotype based on method and available fields
    construction_method = phenotype_construction.get('method', 'binary')
    construction_logic = phenotype_construction.get('logic', '')
    threshold_suggestions = phenotype_construction.get('threshold_suggestions', '')
    
    print(f"\nField categories identified:")
    for category, fields in field_categories.items():
        print(f"  {category}: {fields}")
    
    if construction_method == 'binary':
        # Construct binary phenotype following insomnia-like logic
        phenotype_components = {}
        
        # Extract threshold from suggestions (look for numbers like ≥3, >2, etc.)
        threshold = 3  # default
        threshold_match = re.search(r'[≥>]\s*(\d+)', threshold_suggestions)
        if threshold_match:
            threshold = int(threshold_match.group(1))
        
        print(f"Using threshold: ≥{threshold} for categorical variables")
        
        # Process each category
        for category, fields in field_categories.items():
            if not fields:
                continue
                
            if category == 'symptoms':
                # Any symptom present (OR logic)
                symptom_conditions = []
                for field in fields:
                    if work_df[field].dtype == 'object':
                        # Try to convert to numeric if possible
                        work_df[field] = pd.to_numeric(work_df[field], errors='coerce')
                    symptom_conditions.append(work_df[field].astype('Int64').ge(threshold))
                
                if symptom_conditions:
                    phenotype_components['symptoms'] = pd.concat(symptom_conditions, axis=1).any(axis=1)
            
            elif category == 'frequency':
                # Frequency requirement (AND logic for all frequency measures)
                freq_conditions = []
                for field in fields:
                    if work_df[field].dtype == 'object':
                        work_df[field] = pd.to_numeric(work_df[field], errors='coerce')
                    freq_conditions.append(work_df[field].astype('Int64').ge(threshold))
                
                if freq_conditions:
                    phenotype_components['frequency'] = pd.concat(freq_conditions, axis=1).all(axis=1)
            
            elif category == 'impact':
                # Any impact present (OR logic)
                impact_conditions = []
                for field in fields:
                    if work_df[field].dtype == 'object':
                        work_df[field] = pd.to_numeric(work_df[field], errors='coerce')
                    impact_conditions.append(work_df[field].astype('Int64').ge(threshold))
                
                if impact_conditions:
                    phenotype_components['impact'] = pd.concat(impact_conditions, axis=1).any(axis=1)
            
            elif category in ['duration', 'severity']:
                # Duration/severity requirement (AND logic)
                dur_sev_conditions = []
                for field in fields:
                    if work_df[field].dtype == 'object':
                        work_df[field] = pd.to_numeric(work_df[field], errors='coerce')
                    dur_sev_conditions.append(work_df[field].astype('Int64').ge(threshold))
                
                if dur_sev_conditions:
                    phenotype_components[category] = pd.concat(dur_sev_conditions, axis=1).all(axis=1)
            
            else:  # 'other' category
                # Default to OR logic for other fields
                other_conditions = []
                for field in fields:
                    if work_df[field].dtype == 'object':
                        work_df[field] = pd.to_numeric(work_df[field], errors='coerce')
                    other_conditions.append(work_df[field].astype('Int64').ge(threshold))
                
                if other_conditions:
                    phenotype_components['other'] = pd.concat(other_conditions, axis=1).any(axis=1)
        
        # Combine all components with AND logic (all must be true)
        if phenotype_components:
            final_phenotype = pd.concat(list(phenotype_components.values()), axis=1).all(axis=1)
            work_df[target_phenotype] = final_phenotype.astype(int)
        else:
            print("Warning: No valid phenotype components found")
            work_df[target_phenotype] = 0
    
    elif construction_method == 'continuous':
        # Construct continuous phenotype (e.g., sum or average of relevant fields)
        relevant_fields = [f for f in questionnaire_fields.keys() if f in work_df.columns]
        if relevant_fields:
            # Convert to numeric and sum
            numeric_data = work_df[relevant_fields].apply(pd.to_numeric, errors='coerce')
            work_df[target_phenotype] = numeric_data.sum(axis=1)
        else:
            print("Warning: No valid fields found for continuous phenotype")
            work_df[target_phenotype] = 0
    
    elif construction_method == 'ordinal':
        # Construct ordinal phenotype (e.g., severity levels)
        relevant_fields = [f for f in questionnaire_fields.keys() if f in work_df.columns]
        if relevant_fields:
            # Take maximum value across relevant fields
            numeric_data = work_df[relevant_fields].apply(pd.to_numeric, errors='coerce')
            work_df[target_phenotype] = numeric_data.max(axis=1)
        else:
            print("Warning: No valid fields found for ordinal phenotype")
            work_df[target_phenotype] = 0
    
    # Remove rows with missing phenotype
    work_df = work_df.dropna(subset=[target_phenotype])
    
    # Prepare covariate list
    available_covariates = [cov for cov in covariates if cov in work_df.columns]
    if not available_covariates:
        # Use standard GWAS covariates if available
        standard_covars = ['age', 'sex'] + [f'PC{i}' for i in range(1, 21)]
        available_covariates = [cov for cov in standard_covars if cov in work_df.columns]
    
    print(f"\nAvailable covariates: {available_covariates}")
    
    # Create FID (Family ID) - typically same as IID for unrelated individuals
    work_df.insert(0, 'FID', work_df['IID'])
    
    # Create phenotype file for PLINK
    pheno_file = f"{output_prefix}_pheno.txt"
    pheno_df = work_df[['FID', 'IID', target_phenotype]].copy()
    pheno_df.to_csv(pheno_file, sep='\t', index=False)
    
    # Create covariate file for PLINK
    covar_file = f"{output_prefix}_covars.txt"
    if available_covariates:
        covar_df = work_df[['FID', 'IID'] + available_covariates].copy()
        covar_df.to_csv(covar_file, sep='\t', index=False)
    else:
        covar_df = None
        print("Warning: No covariates available for analysis")
    
    # Create summary statistics
    phenotype_summary = {
        'total_samples': len(work_df),
        'phenotype_name': target_phenotype,
        'phenotype_type': construction_method,
        'case_count': work_df[target_phenotype].sum() if construction_method == 'binary' else None,
        'control_count': (work_df[target_phenotype] == 0).sum() if construction_method == 'binary' else None,
        'mean_value': work_df[target_phenotype].mean(),
        'std_value': work_df[target_phenotype].std(),
        'missing_rate': work_df[target_phenotype].isna().mean(),
        'fields_used': list(questionnaire_fields.keys()),
        'field_categories': field_categories,
        'covariates_used': available_covariates,
        'phenotype_file': pheno_file,
        'covariate_file': covar_file if available_covariates else None,
        'construction_logic': construction_logic,
        'threshold_used': threshold if construction_method == 'binary' else None
    }
    
    # Print summary
    print(f"\n=== GWAS Phenotype Construction Summary ===")
    print(f"Phenotype: {target_phenotype}")
    print(f"Total samples: {phenotype_summary['total_samples']:,}")
    if construction_method == 'binary':
        print(f"Cases: {phenotype_summary['case_count']:,}")
        print(f"Controls: {phenotype_summary['control_count']:,}")
        print(f"Case rate: {phenotype_summary['case_count']/phenotype_summary['total_samples']:.1%}")
    else:
        print(f"Mean value: {phenotype_summary['mean_value']:.3f}")
        print(f"Std deviation: {phenotype_summary['std_value']:.3f}")
    
    print(f"Files created:")
    print(f"  - Phenotype: {pheno_file}")
    if covar_file:
        print(f"  - Covariates: {covar_file}")
    
    return {
        'phenotype_data': work_df,
        'summary': phenotype_summary,
        'files_created': {
            'phenotype': pheno_file,
            'covariates': covar_file
        }
    }

def create_gwas_field_mapping(gwas_mapping):
    """
    Create field mapping tuples in the format expected by UK Biobank analysis.
    
    Parameters:
    -----------
    gwas_mapping : dict
        GWAS variable mapping from LLM query
        
    Returns:
    --------
    tuple : (PHENOTYPE_FIELDS, COVARS) similar to insomnia example
    """
    
    questionnaire_fields = gwas_mapping['questionnaire_fields']
    covariates = gwas_mapping['covariates']
    
    # Create phenotype fields list (assuming categorical for now)
    # Format: (field_name, field_code, data_type)
    PHENOTYPE_FIELDS = []
    for field_name, field_info in questionnaire_fields.items():
        # Extract field code if available in description or use placeholder
        field_code = hash(field_name) % 100000  # Generate pseudo-code
        data_type = 'categorical'  # Default assumption
        
        PHENOTYPE_FIELDS.append((field_name, field_code, data_type))
    
    # Create covariate fields list
    COVARS = []
    standard_covar_codes = {
        'age': 21003,
        'sex': 31
    }
    
    for covar in covariates:
        if covar in standard_covar_codes:
            code = standard_covar_codes[covar]
            data_type = 'continuous' if covar == 'age' else 'categorical'
        elif covar.startswith('PC') and covar[2:].isdigit():
            # Principal component
            pc_num = int(covar[2:])
            code = 22009 + pc_num - 1
            data_type = 'continuous'
        else:
            # Generate pseudo-code for other covariates
            code = hash(covar) % 100000
            data_type = 'continuous'
        
        COVARS.append((covar, code, data_type))
    
    return PHENOTYPE_FIELDS, COVARS

def run_pygwas_analysis(df, gwas_mapping, genotype_data=None):
    """
    Run GWAS analysis using PyGWAS on questionnaire data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with questionnaire responses and sample IDs
    gwas_mapping : dict
        GWAS variable mapping from LLM query
    genotype_data : dict, optional
        Dictionary containing genotype data with keys:
        - 'genotypes': numpy array of genotype matrix
        - 'variant_info': DataFrame with variant information
        - 'sample_ids': list of sample IDs
        
    Returns:
    --------
    dict : PyGWAS results for MR analysis
    """
    
    if not _check_pygwas_availability():
        raise ImportError("PyGWAS not available. Cannot run integrated GWAS analysis.")
    
    print("=== Running PyGWAS Analysis for MR ===")
    
    # Import GWAS only when needed
    try:
        from pygwas import GWAS
    except ImportError as e:
        raise ImportError(f"Could not import PyGWAS: {e}")
    
    # Step 1: Construct phenotype using GWAS variable mapping
    phenotype_result = construct_phenotype_from_gwas_mapping(df, gwas_mapping)
    phenotype_data = phenotype_result['phenotype_data']
    summary = phenotype_result['summary']
    
    target_phenotype = summary['phenotype_name']
    print(f"Target phenotype for GWAS: {target_phenotype}")
    
    # Step 2: Initialize PyGWAS with LLM enhancement
    gwas = GWAS(enable_llm=True, verbose=True, output_dir="mr_gwas_output")
    
    # Step 3: Apply GWAS variable mapping to PyGWAS
    if gwas_mapping:
        gwas.set_gwas_variable_mapping(gwas_mapping)
        print("✓ Applied GWAS variable mapping to PyGWAS")
    
    print("Using provided genotype data")
    gwas.load_data(
        genotypes=genotype_data['genotypes'],
        phenotypes=phenotype_data,
        variant_info=genotype_data['variant_info'],
        sample_ids=genotype_data['sample_ids']
    )
    
    print(f"Loaded {gwas.genotypes.shape[0]} samples, {gwas.genotypes.shape[1]} variants")
    
    # Step 5: Run quality control
    print("Running quality control...")
    gwas.run_qc(
        sample_call_rate=0.95,
        snp_call_rate=0.95,
        min_maf=0.01,
        hwe_threshold=1e-6
    )
    
    # Step 6: Calculate population structure
    print("Calculating population structure...")
    gwas.calculate_population_structure(n_components=10)
    
    # Step 7: Run association test
    print("Running association test...")
    association_results = gwas.run_association_test(
        trait_type='auto',
        test_method='auto',
        n_pcs=3,
        use_kinship=False
    )
    
    # Step 8: Process results for MR analysis
    mr_ready_results = prepare_gwas_results_for_mr(association_results, target_phenotype)
    
    # Step 9: Generate summary
    analysis_summary = gwas.get_llm_analysis_summary()
    
    print(f"✓ PyGWAS analysis complete!")
    print(f"  - Significant associations (P < 5e-8): {np.sum(association_results['P'] < 5e-8)}")
    print(f"  - Suggestive associations (P < 1e-5): {np.sum(association_results['P'] < 1e-5)}")
    
    return {
        'gwas_results': association_results,
        'mr_instruments': mr_ready_results,
        'phenotype_summary': summary,
        'analysis_summary': analysis_summary,
        'gwas_object': gwas,
        'target_phenotype': target_phenotype
    }

def prepare_gwas_results_for_mr(association_results, target_phenotype, p_threshold=5e-8):
    """
    Prepare PyGWAS results for MR analysis by extracting significant SNPs as instruments
    
    Parameters:
    -----------
    association_results : pandas.DataFrame
        GWAS association results from PyGWAS
    target_phenotype : str
        Name of the target phenotype
    p_threshold : float
        P-value threshold for selecting instruments
        
    Returns:
    --------
    pandas.DataFrame : MR-ready instrument data
    """
    
    print(f"Preparing GWAS results for MR analysis (p < {p_threshold})")
    
    # Filter significant associations
    significant_snps = association_results[association_results['P'] < p_threshold].copy()
    
    if len(significant_snps) == 0:
        print(f"⚠ No genome-wide significant SNPs found at p < {p_threshold}")
        print(f"  Using suggestive threshold p < 1e-5 instead")
        significant_snps = association_results[association_results['P'] < 1e-5].copy()
    
    if len(significant_snps) == 0:
        print(f"⚠ No significant SNPs found even at p < 1e-5")
        print(f"  Using top 100 SNPs for demonstration")
        significant_snps = association_results.nsmallest(100, 'P').copy()
    
    # Prepare MR format
    mr_instruments = significant_snps.copy()
    mr_instruments['beta'] = mr_instruments['BETA']
    mr_instruments['se'] = mr_instruments['SE']
    mr_instruments['pval'] = mr_instruments['P']
    mr_instruments['rsid'] = mr_instruments['SNP']
    mr_instruments['chr'] = mr_instruments['CHR']
    mr_instruments['pos'] = mr_instruments['POS']
    mr_instruments['phenotype'] = target_phenotype
    
    # Add effect allele and other allele info
    mr_instruments['effect_allele'] = mr_instruments['A1']
    mr_instruments['other_allele'] = mr_instruments['A2']
    
    print(f"✓ Prepared {len(mr_instruments)} SNPs as potential instruments")
    
    return mr_instruments

def run_gwas_analysis(df, gwas_mapping, exposure_id=None, outcome_id=None, genotype_data=None):
    """
    Run complete GWAS analysis using the LLM-identified variable mapping.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with questionnaire responses
    gwas_mapping : dict
        GWAS variable mapping from LLM query
    exposure_id : str, optional
        OpenGWAS ID for exposure data (if using external exposure)
    outcome_id : str, optional  
        OpenGWAS ID for outcome data (if using external outcome)
        
    Returns:
    --------
    dict : Results of MR analysis
    """
    
    print("=== Running GWAS Analysis Pipeline ===")
    
    # Step 1: Check if we should use PyGWAS for integrated analysis
    if _check_pygwas_availability() and not (exposure_id and outcome_id):
        print("Using PyGWAS for integrated GWAS -> MR analysis")
        return run_integrated_pygwas_mr(df, gwas_mapping, exposure_id, outcome_id, genotype_data)
    
    # Step 2: Fallback to external GWAS databases
    print("Using external GWAS databases for MR analysis")
    
    # Construct phenotype from questionnaire data
    phenotype_result = construct_phenotype_from_gwas_mapping(df, gwas_mapping)
    phenotype_data = phenotype_result['phenotype_data']
    summary = phenotype_result['summary']
    
    target_phenotype = summary['phenotype_name']
    pheno_file = summary['phenotype_file']
    covar_file = summary['covariate_file']
    
    # Determine analysis type based on available data
    if exposure_id and outcome_id:
        print(f"Running external exposure -> external outcome MR")
        return run_external_mr(exposure_id, outcome_id)
    
    elif exposure_id:
        print(f"Running external exposure -> constructed phenotype MR")
        # Run GWAS on constructed phenotype first, then MR
        return run_exposure_to_phenotype_mr(exposure_id, phenotype_data, target_phenotype, covar_file)
    
    elif outcome_id:
        print(f"Running constructed phenotype -> external outcome MR")
        # Run GWAS on constructed phenotype as exposure, then MR to external outcome
        return run_phenotype_to_outcome_mr(phenotype_data, target_phenotype, covar_file, outcome_id)
    
    else:
        print(f"Running GWAS discovery on constructed phenotype")
        # Just run GWAS discovery on the constructed phenotype
        return run_phenotype_gwas(phenotype_data, target_phenotype, covar_file)

def run_integrated_pygwas_mr(df, gwas_mapping, exposure_id=None, outcome_id=None, genotype_data=None):
    """
    Run integrated PyGWAS -> MR analysis pipeline like simple_gwas_mr_example.py
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with questionnaire responses
    gwas_mapping : dict
        GWAS variable mapping from LLM query
    exposure_id : str, optional
        OpenGWAS ID for external exposure data
    outcome_id : str, optional
        OpenGWAS ID for external outcome data
    genotype_data : dict, optional
        Genotype data for PyGWAS analysis
        
    Returns:
    --------
    dict : Complete MR analysis results
    """
    
    print("=== Integrated PyGWAS -> MR Analysis ===")
    
    # Check if PyGWAS is available
    if not _check_pygwas_availability():
        return {
            'success': False,
            'error': 'PyGWAS not available for integrated analysis',
            'note': 'Install PyGWAS for integrated GWAS-MR pipeline'
        }
    
    try:
        # Import GWAS and related modules
        import sys
        from pathlib import Path
        
        # Add pygwas to the path (similar to simple_gwas_mr_example.py)
        current_dir = Path.cwd()
        pygwas_path = current_dir / "pygwas"
        if str(pygwas_path) not in sys.path:
            sys.path.insert(0, str(pygwas_path))
        
        from pygwas import GWAS
        
        print("✅ PyGWAS imported successfully")
        
        # Check if we have genetic data or need to use OpenGWAS approach
        if genotype_data is None:
            # No genetic data provided - use OpenGWAS + dataset approach (not synthetic)
            print("No genetic data provided - using OpenGWAS integration with dataset...")
            
            # Get GWAS compatibility information from LLM results
            is_gwas_compatible = False
            dataset_variable_type = None  # "EXPOSURE" or "OUTCOME"
            
            # Check if LLM determined this is GWAS-compatible data
            if hasattr(sys.modules.get('__main__', sys.modules[__name__]), 'llm_results'):
                # Access llm_results from main module context if available
                main_llm_results = getattr(sys.modules.get('__main__', sys.modules[__name__]), 'llm_results', None)
                if main_llm_results:
                    is_gwas_compatible = main_llm_results.get('is_questionnaire_with_genetics', {}).get('value') == 'Yes'
                    dataset_variable_type = main_llm_results.get('gwas_target_type', {}).get('value')
            
            # Also check gwas_mapping for GWAS compatibility indicators
            if gwas_mapping and 'target_phenotype' in gwas_mapping:
                target_phenotype = gwas_mapping['target_phenotype']
                is_gwas_compatible = True
                # Try to determine variable type from gwas_mapping or infer from OpenGWAS IDs
                if exposure_id and not outcome_id:
                    dataset_variable_type = "OUTCOME"  # We have external exposure, dataset is outcome
                elif outcome_id and not exposure_id:
                    dataset_variable_type = "EXPOSURE"  # We have external outcome, dataset is exposure
                elif not exposure_id and not outcome_id:
                    # Default to EXPOSURE if no external data provided
                    dataset_variable_type = "EXPOSURE"
            else:
                # Use first numeric column as target phenotype
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                target_phenotype = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]
            
            print(f"🎯 Target phenotype from dataset: {target_phenotype}")
            print(f"📊 GWAS compatible: {is_gwas_compatible}")
            print(f"📈 Dataset variable type: {dataset_variable_type}")
            
            if exposure_id:
                print(f"🔗 External exposure OpenGWAS ID: {exposure_id}")
            if outcome_id:
                print(f"🎯 External outcome OpenGWAS ID: {outcome_id}")
            
            # Use OpenGWAS for exposure/outcome data with dataset phenotypes
            return run_opengwas_with_dataset_integration(
                df=df,
                target_phenotype=target_phenotype,
                gwas_mapping=gwas_mapping,
                exposure_id=exposure_id,
                outcome_id=outcome_id,
                dataset_variable_type=dataset_variable_type
            )
        else:
            # Use provided genetic data (original PyGWAS approach)
            print("Using provided genetic data for full PyGWAS pipeline...")
            genotypes = genotype_data['genotypes']
            variant_info = genotype_data['variant_info']
            sample_ids = genotype_data['sample_ids']
            phenotypes = df.copy()
            if 'sample_id' not in phenotypes.columns:
                phenotypes.insert(0, 'sample_id', sample_ids)
        
            # Determine target phenotype for analysis
            if gwas_mapping and 'target_phenotype' in gwas_mapping:
                target_phenotype = gwas_mapping['target_phenotype']
            else:
                # Use first numeric column as target phenotype
                numeric_cols = phenotypes.select_dtypes(include=[np.number]).columns
                target_phenotype = numeric_cols[0] if len(numeric_cols) > 0 else phenotypes.columns[1]
            
            print(f"📊 Data prepared:")
            print(f"   Samples: {len(sample_ids)}")
            print(f"   Variants: {len(variant_info)}")
            print(f"   Phenotypes: {list(phenotypes.columns)}")
            print(f"🎯 Target phenotype: {target_phenotype}")
            
            # Initialize GWAS (similar to simple_gwas_mr_example.py)
            gwas = GWAS(output_dir="mr_integrated_output", verbose=False)
            
            # Create simplified llm_results for the pipeline
            pipeline_llm_results = {}
            if gwas_mapping:
                pipeline_llm_results = {
                    'outcome': {'value': target_phenotype},
                    'treatment': {'value': gwas_mapping.get('treatment_variable', target_phenotype)}
                }
            
            # Run GWAS Pipeline (ONE FUNCTION CALL like simple_gwas_mr_example.py)
            print("🚀 Running GWAS pipeline...")
            gwas_results = gwas.run_full_gwas_pipeline(
                genotypes=genotypes,
                phenotypes=phenotypes,
                variant_info=variant_info,
                sample_ids=sample_ids,
                llm_results=pipeline_llm_results,
                target_phenotype=target_phenotype,
                save_results=False  # Don't save files in integrated mode
            )
            
            print("✅ GWAS pipeline complete!")
            print(f"   Variants tested: {gwas_results['summary']['n_variants_tested']:,}")
            print(f"   Significant: {gwas_results['summary']['n_significant']}")
            
            # Run MR Pipeline (ONE FUNCTION CALL like simple_gwas_mr_example.py)
            print("🚀 Running MR pipeline...")
            mr_results = gwas.run_full_mr_pipeline(
                gwas_results=gwas_results,
                llm_results=pipeline_llm_results,
                mr_type='llm_guided'
            )
            
            print("✅ MR pipeline complete!")
            print(f"   MR Type: {mr_results.get('mr_type', 'unknown')}")
            print(f"   Success: {mr_results.get('success', False)}")
            
            # Return standardized results
            return {
                'success': mr_results.get('success', True),
                'causal_effect': mr_results.get('causal_effect'),
                'standard_error': mr_results.get('standard_error'),
                'confidence_interval': mr_results.get('confidence_interval', [None, None]),
                'p_value': mr_results.get('p_value'),
                'mr_method': mr_results.get('mr_method', 'PyGWAS Integrated'),
                'n_instruments': mr_results.get('n_instruments', 0),
                'f_statistics': mr_results.get('f_statistics', []),
                'heterogeneity': mr_results.get('heterogeneity', {}),
                'pleiotropy_test': mr_results.get('pleiotropy_test', {}),
                'gwas_summary': gwas_results.get('summary', {}),
                'target_phenotype': target_phenotype,
                'note': 'Integrated PyGWAS-MR analysis using complete pipeline'
            }
        
    except Exception as e:
        print(f"❌ Integrated PyGWAS-MR analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': f'PyGWAS integration error: {str(e)}',
            'note': 'Check PyGWAS installation and data compatibility'
        }

def run_opengwas_with_dataset_integration(df, target_phenotype, gwas_mapping=None, exposure_id=None, outcome_id=None, dataset_variable_type=None):
    """
    Run MR analysis using the original dataset combined with OpenGWAS data
    (This is the approach that was used before synthetic data)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The actual dataset provided by the user
    target_phenotype : str
        Target phenotype from the dataset
    gwas_mapping : dict, optional
        GWAS variable mapping from LLM query
    exposure_id : str, optional
        OpenGWAS ID for external exposure data
    outcome_id : str, optional
        OpenGWAS ID for external outcome data
    dataset_variable_type : str, optional
        Whether the dataset represents "EXPOSURE" or "OUTCOME" (from LLM analysis)
        
    Returns:
    --------
    dict : MR analysis results
    """
    
    print("=== OpenGWAS + Dataset Integration ===")
    print(f"Using dataset phenotype: {target_phenotype}")
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset variable type: {dataset_variable_type}")
    
    # Determine MR approach based on available OpenGWAS IDs and dataset type
    if exposure_id and outcome_id:
        print(f"📊 Two-sample MR: {exposure_id} -> {outcome_id}")
        print("Using both exposure and outcome from OpenGWAS")
        
        # Run standard two-sample MR (original approach)
        try:
            mr_results = run_external_mr(exposure_id, outcome_id)
            if mr_results and mr_results.get('success'):
                return {
                    'success': True,
                    'causal_effect': mr_results.get('causal_effect'),
                    'confidence_interval': mr_results.get('confidence_interval'),
                    'p_value': mr_results.get('p_value'),
                    'method': 'Mendelian Randomization (Two-Sample OpenGWAS)',
                    'mr_method': 'External GWAS Integration',
                    'n_instruments': mr_results.get('n_instruments', 0),
                    'exposure_gwas_id': exposure_id,
                    'outcome_gwas_id': outcome_id,
                    'dataset_phenotype': target_phenotype,
                    'note': f'Two-sample MR using OpenGWAS: {exposure_id} -> {outcome_id}, with dataset context ({dataset_variable_type})'
                }
            else:
                return {
                    'success': False,
                    'error': f'Two-sample MR failed between {exposure_id} and {outcome_id}',
                    'note': 'Both OpenGWAS IDs provided but MR analysis failed'
                }
        except Exception as e:
            return {
                'success': False,
                'error': f'Two-sample MR error: {str(e)}',
                'note': f'Failed to run MR between {exposure_id} and {outcome_id}'
            }
            
    elif exposure_id and dataset_variable_type == "OUTCOME":
        print(f"📊 External exposure -> Dataset outcome: {exposure_id} -> {target_phenotype}")
        print("Using exposure from OpenGWAS, outcome from dataset")
        
        # Use external exposure data with dataset outcome
        return run_external_exposure_to_dataset_outcome(
            exposure_id=exposure_id,
            df=df,
            target_phenotype=target_phenotype,
            gwas_mapping=gwas_mapping
        )
        
    elif outcome_id and dataset_variable_type == "EXPOSURE":
        print(f"📊 Dataset exposure -> External outcome: {target_phenotype} -> {outcome_id}")
        print("Using exposure from dataset, outcome from OpenGWAS")
        
        # Use dataset exposure data with external outcome
        return run_dataset_exposure_to_external_outcome(
            df=df,
            target_phenotype=target_phenotype,
            outcome_id=outcome_id,
            gwas_mapping=gwas_mapping
        )
        
    elif exposure_id and not dataset_variable_type:
        # Backward compatibility: if dataset_variable_type not specified, infer from context
        print(f"📊 External exposure -> Dataset outcome (inferred): {exposure_id} -> {target_phenotype}")
        print("Using exposure from OpenGWAS, outcome from dataset (backward compatibility)")
        
        return run_external_exposure_to_dataset_outcome(
            exposure_id=exposure_id,
            df=df,
            target_phenotype=target_phenotype,
            gwas_mapping=gwas_mapping
        )
        
    elif outcome_id and not dataset_variable_type:
        # Backward compatibility: if dataset_variable_type not specified, infer from context  
        print(f"📊 Dataset exposure -> External outcome (inferred): {target_phenotype} -> {outcome_id}")
        print("Using exposure from dataset, outcome from OpenGWAS (backward compatibility)")
        
        # This would require running GWAS on the dataset phenotype first
        return {
            'success': False,
            'analysis_type': 'dataset_exposure_to_external_outcome',
            'exposure_phenotype': target_phenotype,
            'outcome_gwas_id': outcome_id,
            'dataset_shape': df.shape,
            'status': 'requires_dataset_gwas',
            'error': 'This analysis requires running GWAS on the dataset phenotype first',
            'note': f'To analyze {target_phenotype} -> {outcome_id}, need genetic data for the dataset',
            'dataset_variable_type': dataset_variable_type,
            'next_steps': [
                f"1. Obtain genetic data for the {len(df)} samples in the dataset",
                f"2. Run GWAS on {target_phenotype} using the dataset",
                f"3. Extract genome-wide significant SNPs (p < 5e-8)",
                "4. Clump SNPs to remove linkage disequilibrium",
                f"5. Query {outcome_id} for associations with these SNPs",
                "6. Perform MR analysis using harmonized data"
            ]
        }
        
    else:
        print("📊 No OpenGWAS IDs provided - Dataset-only analysis")
        print(f"Dataset contains phenotype: {target_phenotype}")
        
        # Look for potential genetic instruments in the dataset itself
        genetic_cols = [col for col in df.columns if any(pattern in col.lower() 
                       for pattern in ['snp', 'rs', 'genetic', 'variant', 'allele'])]
        
        if len(genetic_cols) > 0:
            print(f"Found {len(genetic_cols)} potential genetic columns in dataset: {genetic_cols[:5]}...")
            
            # Return informative message instead of running simplified MR
            return {
                'success': False,
                'analysis_type': 'dataset_only_genetics_found',
                'target_phenotype': target_phenotype,
                'genetic_columns_found': genetic_cols,
                'dataset_shape': df.shape,
                'status': 'requires_external_gwas_data',
                'error': 'Genetic columns found in dataset but no OpenGWAS IDs provided',
                'note': f'Dataset contains {len(genetic_cols)} potential genetic variants but MR analysis requires external GWAS data for proper causal inference',
                'suggestion': 'Provide exposure_id and/or outcome_id for OpenGWAS integration to perform robust MR analysis'
            }
        else:
            return {
                'success': False,
                'analysis_type': 'dataset_only_no_genetics',
                'target_phenotype': target_phenotype,
                'dataset_shape': df.shape,
                'status': 'no_genetic_data',
                'error': 'No OpenGWAS IDs provided and no genetic data found in dataset',
                'note': 'MR analysis requires either OpenGWAS IDs or genetic variants in the dataset',
                'suggestion': 'Provide exposure_id and/or outcome_id for OpenGWAS integration'
            }

def run_pygwas_to_external_mr(instruments, exposure_phenotype, outcome_id):
    """
    Run MR from PyGWAS instruments (exposure) to external outcome
    
    Parameters:
    -----------
    instruments : pandas.DataFrame
        Genetic instruments from PyGWAS analysis
    exposure_phenotype : str
        Name of the exposure phenotype from PyGWAS
    outcome_id : str
        OpenGWAS ID for outcome data
        
    Returns:
    --------
    dict : MR analysis results
    """
    
    try:
        # Get outcome data from OpenGWAS
        print(f"Fetching outcome data for {outcome_id}...")
        outcome_data = query.associations(variants=instruments['rsid'].tolist(), id=outcome_id)
        outcome_df = pd.DataFrame(outcome_data)
        
        if len(outcome_df) == 0:
            return {
                'success': False,
                'error': f'No outcome data found for {outcome_id} with the provided instruments'
            }
        
        # Harmonize data
        harmonized_data = harmonize_mr_data(instruments, outcome_df, exposure_phenotype, outcome_id)
        
        if len(harmonized_data) < 3:
            return {
                'success': False,
                'error': f'Insufficient instruments after harmonization ({len(harmonized_data)} < 3)'
            }
        
        # Run MR analysis
        mr_results = perform_mr_analysis(harmonized_data, exposure_phenotype, outcome_id)
        
        return {
            'success': True,
            'mr_results': mr_results,
            'n_instruments_used': len(harmonized_data),
            'exposure': exposure_phenotype,
            'outcome': outcome_id
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Error in PyGWAS to external MR: {str(e)}'
        }

def run_external_to_pygwas_mr(exposure_id, outcome_instruments, outcome_phenotype):
    """
    Run MR from external exposure to PyGWAS outcome
    
    Parameters:
    -----------
    exposure_id : str
        OpenGWAS ID for exposure data
    outcome_instruments : pandas.DataFrame
        Genetic instruments for the outcome from PyGWAS
    outcome_phenotype : str
        Name of the outcome phenotype from PyGWAS
        
    Returns:
    --------
    dict : MR analysis results
    """
    
    try:
        # Get exposure data for the outcome SNPs
        print(f"Fetching exposure data for {exposure_id}...")
        exposure_data = query.associations(variants=outcome_instruments['rsid'].tolist(), id=exposure_id)
        exposure_df = pd.DataFrame(exposure_data)
        
        if len(exposure_df) == 0:
            return {
                'success': False,
                'error': f'No exposure data found for {exposure_id} with the provided instruments'
            }
        
        # Harmonize data (reverse direction)
        harmonized_data = harmonize_mr_data(exposure_df, outcome_instruments, exposure_id, outcome_phenotype)
        
        if len(harmonized_data) < 3:
            return {
                'success': False,
                'error': f'Insufficient instruments after harmonization ({len(harmonized_data)} < 3)'
            }
        
        # Run MR analysis
        mr_results = perform_mr_analysis(harmonized_data, exposure_id, outcome_phenotype)
        
        return {
            'success': True,
            'mr_results': mr_results,
            'n_instruments_used': len(harmonized_data),
            'exposure': exposure_id,
            'outcome': outcome_phenotype
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Error in external to PyGWAS MR: {str(e)}'
        }

def harmonize_mr_data(exposure_data, outcome_data, exposure_name, outcome_name):
    """
    Harmonize exposure and outcome data for MR analysis
    
    Parameters:
    -----------
    exposure_data : pandas.DataFrame
        Genetic associations for exposure
    outcome_data : pandas.DataFrame
        Genetic associations for outcome
    exposure_name : str
        Name of exposure trait
    outcome_name : str
        Name of outcome trait
        
    Returns:
    --------
    pandas.DataFrame : Harmonized data for MR analysis
    """
    
    print(f"Harmonizing MR data: {exposure_name} -> {outcome_name}")
    
    # Ensure consistent column names
    exposure_df = exposure_data.copy()
    outcome_df = outcome_data.copy()
    
    # Standardize column names for exposure
    if 'rsid' in exposure_df.columns:
        exposure_df['SNP'] = exposure_df['rsid']
    elif 'SNP' not in exposure_df.columns and 'variant' in exposure_df.columns:
        exposure_df['SNP'] = exposure_df['variant']
    
    # Standardize column names for outcome
    if 'rsid' in outcome_df.columns:
        outcome_df['SNP'] = outcome_df['rsid']
    elif 'SNP' not in outcome_df.columns and 'variant' in outcome_df.columns:
        outcome_df['SNP'] = outcome_df['variant']
    
    # Merge on SNP ID
    harmonized = pd.merge(
        exposure_df[['SNP', 'beta', 'se', 'pval', 'effect_allele', 'other_allele']],
        outcome_df[['SNP', 'beta', 'se', 'pval', 'effect_allele', 'other_allele']],
        on='SNP',
        suffixes=('_exp', '_out')
    )
    
    # Basic allele harmonization (simplified)
    harmonized = harmonized[
        (harmonized['effect_allele_exp'] == harmonized['effect_allele_out']) &
        (harmonized['other_allele_exp'] == harmonized['other_allele_out'])
    ].copy()
    
    # Filter for reasonable effect sizes and p-values
    harmonized = harmonized[
        (harmonized['pval_exp'] < 0.05) &
        (harmonized['se_exp'] > 0) &
        (harmonized['se_out'] > 0) &
        (np.abs(harmonized['beta_exp']) < 10) &
        (np.abs(harmonized['beta_out']) < 10)
    ].copy()
    
    print(f"✓ Harmonized {len(harmonized)} instruments")
    return harmonized

def perform_mr_analysis(harmonized_data, exposure_name, outcome_name):
    """
    Perform Mendelian Randomization analysis
    
    Parameters:
    -----------
    harmonized_data : pandas.DataFrame
        Harmonized exposure and outcome data
    exposure_name : str
        Name of exposure trait
    outcome_name : str
        Name of outcome trait
        
    Returns:
    --------
    dict : MR analysis results
    """
    
    print(f"Performing MR analysis: {exposure_name} -> {outcome_name}")
    
    # Extract data
    beta_exp = harmonized_data['beta_exp'].values
    se_exp = harmonized_data['se_exp'].values
    beta_out = harmonized_data['beta_out'].values
    se_out = harmonized_data['se_out'].values
    
    # Inverse variance weighted (IVW) method
    ivw_weights = 1 / (se_out**2)
    ivw_beta = np.sum(beta_out * beta_exp * ivw_weights) / np.sum(beta_exp**2 * ivw_weights)
    ivw_se = np.sqrt(1 / np.sum(beta_exp**2 * ivw_weights))
    ivw_pval = 2 * (1 - stats.norm.cdf(np.abs(ivw_beta / ivw_se)))
    
    # MR-Egger method (if sufficient instruments)
    egger_results = None
    if len(harmonized_data) >= 10:
        try:
            # Simple MR-Egger implementation
            X = np.column_stack([np.ones(len(beta_exp)), beta_exp])
            W = np.diag(1 / se_out**2)
            y = beta_out
            
            # Weighted least squares
            XtWX_inv = np.linalg.inv(X.T @ W @ X)
            egger_coef = XtWX_inv @ X.T @ W @ y
            egger_var = np.diag(XtWX_inv)
            
            egger_results = {
                'beta': egger_coef[1],
                'se': np.sqrt(egger_var[1]),
                'pval': 2 * (1 - stats.norm.cdf(np.abs(egger_coef[1] / np.sqrt(egger_var[1])))),
                'intercept': egger_coef[0],
                'intercept_pval': 2 * (1 - stats.norm.cdf(np.abs(egger_coef[0] / np.sqrt(egger_var[0]))))
            }
        except:
            egger_results = None
    
    # Weighted median method
    median_weights = ivw_weights / np.sum(ivw_weights)
    ratio_estimates = beta_out / beta_exp
    sorted_indices = np.argsort(ratio_estimates)
    cumulative_weights = np.cumsum(median_weights[sorted_indices])
    median_index = np.where(cumulative_weights >= 0.5)[0][0]
    weighted_median_beta = ratio_estimates[sorted_indices[median_index]]
    
    results = {
        'method': 'Mendelian Randomization',
        'exposure': exposure_name,
        'outcome': outcome_name,
        'n_instruments': len(harmonized_data),
        'ivw': {
            'beta': ivw_beta,
            'se': ivw_se,
            'pval': ivw_pval,
            'ci_lower': ivw_beta - 1.96 * ivw_se,
            'ci_upper': ivw_beta + 1.96 * ivw_se
        },
        'weighted_median': {
            'beta': weighted_median_beta
        },
        'egger': egger_results
    }
    
    print(f"✓ MR analysis complete:")
    print(f"  IVW estimate: {ivw_beta:.4f} (SE: {ivw_se:.4f}, P: {ivw_pval:.2e})")
    if egger_results:
        print(f"  MR-Egger estimate: {egger_results['beta']:.4f} (P: {egger_results['pval']:.2e})")
    
    return results

def run_external_mr(exposure_id, outcome_id):
    """Run MR between two external OpenGWAS datasets"""
    
    try:
        print(f"Running two-sample MR: {exposure_id} -> {outcome_id}")
        
        # This would use genal or similar package to run MR
        # For now, return a placeholder structure
        return {
            'success': False,
            'error': 'External MR function not fully implemented',
            'note': 'This requires genal package integration for OpenGWAS queries',
            'exposure_id': exposure_id,
            'outcome_id': outcome_id
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'External MR error: {str(e)}',
            'note': 'Error in external GWAS MR analysis'
        }

def run_external_exposure_to_dataset_outcome(exposure_id, df, target_phenotype, gwas_mapping=None):
    """Run MR from external exposure to dataset outcome"""
    
    return {
        'success': False,
        'error': 'External exposure to dataset outcome MR not fully implemented',
        'note': 'This requires running GWAS on the dataset phenotype first',
        'exposure_id': exposure_id,
        'outcome_phenotype': target_phenotype,
        'dataset_shape': df.shape
    }

def run_dataset_exposure_to_external_outcome(df, target_phenotype, outcome_id, gwas_mapping=None):
    """Run MR from dataset exposure to external outcome"""
    
    return {
        'success': False,
        'error': 'Dataset exposure to external outcome MR not fully implemented', 
        'note': 'This requires running GWAS on the dataset phenotype first',
        'exposure_phenotype': target_phenotype,
        'outcome_id': outcome_id,
        'dataset_shape': df.shape
    }

def run_exposure_to_phenotype_mr(exposure_id, phenotype_data, target_phenotype, covar_file):
    """Run MR from external exposure to constructed phenotype"""
    
    return {
        'success': False,
        'error': 'Exposure to phenotype MR not fully implemented',
        'note': 'This requires GWAS analysis on the constructed phenotype',
        'exposure_id': exposure_id,
        'target_phenotype': target_phenotype
    }

def run_phenotype_to_outcome_mr(phenotype_data, target_phenotype, covar_file, outcome_id):
    """Run MR from constructed phenotype to external outcome"""
    
    return {
        'success': False,
        'error': 'Phenotype to outcome MR not fully implemented',
        'note': 'This requires GWAS analysis on the constructed phenotype',
        'target_phenotype': target_phenotype,
        'outcome_id': outcome_id
    }

def run_phenotype_gwas(phenotype_data, target_phenotype, covar_file):
    """Run GWAS discovery on constructed phenotype"""
    
    return {
        'success': False,
        'error': 'Phenotype GWAS not fully implemented',
        'note': 'This requires genetic data for GWAS analysis',
        'target_phenotype': target_phenotype
    }

def run_external_mr(exposure_id, outcome_id):
    """Run MR between two external OpenGWAS datasets"""
    
    try:
        print(f"Running two-sample MR: {exposure_id} -> {outcome_id}")
        
        # Use genal for two-sample MR if available
        try:
            # Initialize genal MR object
            mr = genal.MR()
            
            # Run two-sample MR
            results = mr.two_sample(exposure_id, outcome_id)
            
            if results and isinstance(results, dict):
                # Extract IVW results (most common method)
                if 'IVW' in results:
                    ivw = results['IVW']
                    return {
                        'success': True,
                        'causal_effect': ivw.get('beta', ivw.get('b')),
                        'standard_error': ivw.get('se'),
                        'confidence_interval': [
                            ivw.get('beta', 0) - 1.96 * ivw.get('se', 0),
                            ivw.get('beta', 0) + 1.96 * ivw.get('se', 0)
                        ],
                        'p_value': ivw.get('pval', ivw.get('p')),
                        'mr_method': 'IVW',
                        'n_instruments': results.get('n_instruments', 0),
                        'mr_results': results,
                        'exposure_id': exposure_id,
                        'outcome_id': outcome_id
                    }
                else:
                    return {
                        'success': False,
                        'error': 'No IVW results found in MR analysis',
                        'mr_results': results,
                        'exposure_id': exposure_id,
                        'outcome_id': outcome_id
                    }
            else:
                return {
                    'success': False,
                    'error': 'No valid results returned from two-sample MR',
                    'exposure_id': exposure_id,
                    'outcome_id': outcome_id
                }
                
        except Exception as genal_error:
            print(f"genal MR failed: {genal_error}")
            return {
                'success': False,
                'error': f'genal MR analysis failed: {str(genal_error)}',
                'note': 'Error in genal two-sample MR analysis',
                'exposure_id': exposure_id,
                'outcome_id': outcome_id
            }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'External MR error: {str(e)}',
            'note': 'Error in external GWAS MR analysis',
            'exposure_id': exposure_id,
            'outcome_id': outcome_id
        }

# Additional MR analysis functions and example code (commented out for reference)
#
# The following code shows how to use genal for detailed MR analysis:
#
        
        # Check if exposure_data is empty or None
        if not exposure_data:
            print("⚠️ No exposure data returned from OpenGWAS")
            return {
                'success': False,
                'error': f'No significant SNPs found for exposure {exposure_id}'
            }
        
        # Convert to DataFrame with error handling
        if isinstance(exposure_data, list) and len(exposure_data) == 0:
            print("⚠️ Empty exposure data list")
            return {
                'success': False,
                'error': f'Empty exposure data for {exposure_id}'
            }
        
        exposure_df = pd.DataFrame(exposure_data)
        print(f"✓ Found {len(exposure_df)} exposure SNPs")
        
        if len(exposure_df) == 0:
            return {
                'success': False,
                'error': f'No exposure SNPs found for {exposure_id}'
            }
        
        exposure_geno = genal.Geno(
            exposure_df,
            CHR="chr",
            POS="position", 
            SNP="rsid",
            EA="ea",
            NEA="nea", 
            BETA="beta",
            SE="se",
            P="p",
            EAF="eaf"
        )
        
        snp_list = exposure_geno.data["SNP"].tolist()
        print(f"Using {len(snp_list)} SNPs to query outcome...")
        
        # Get outcome data
        outcome_data = query.associations(
            variant=snp_list,
            id=[outcome_id],
            proxies=1,
            r2=0.8
        )
        
        # Check if outcome_data is empty or None
        if not outcome_data:
            print("⚠️ No outcome data returned from OpenGWAS")
            return {
                'success': False,
                'error': f'No outcome associations found for {outcome_id} with provided SNPs'
            }
        
        # Convert to DataFrame with error handling
        if isinstance(outcome_data, list) and len(outcome_data) == 0:
            print("⚠️ Empty outcome data list")
            return {
                'success': False,
                'error': f'Empty outcome data for {outcome_id}'
            }
        
        outcome_df = pd.DataFrame(outcome_data)
        print(f"✓ Found {len(outcome_df)} outcome associations")
        
        if len(outcome_df) == 0:
            return {
                'success': False,
                'error': f'No outcome associations found for {outcome_id}'
            }
        
        outcome_geno = genal.Geno(
            df=outcome_df,
            CHR="chr",
            POS="position",
            SNP="rsid", 
            EA="ea",
            NEA="nea",
            BETA="beta",
            SE="se",
            P="p",
            EAF="eaf"
        )
        
        # Harmonize and run MR
        exposure_geno.query_outcome(
            outcome=outcome_geno,
            proxy=0,
            r2=0.8,
            kb=5000
        )
        
        # Check if we have harmonized data
        if len(exposure_geno.data) == 0:
            return {
                'success': False,
                'error': 'No instruments available after harmonization'
            }
        
        # Filter F-stat
        dg = exposure_geno.data
        exposure_geno.data["F"] = (dg["beta"] / dg["se"]) ** 2
        exposure_geno.data = exposure_geno.data[exposure_geno.data["F"] >= 10]
        
        if len(exposure_geno.data) == 0:
            return {
                'success': False,
                'error': 'No strong instruments available (F-stat < 10)'
            }
        
        print(f"✓ Using {len(exposure_geno.data)} instruments with F-stat ≥ 10")
        print(f"F-stat summary after filtering:")
        print(exposure_geno.data["F"].describe())
        
        # Run MR-PRESSO and MR analysis
        try:
            exposure_geno.MRpresso(action=2, n_iterations=30000)
        except Exception as e:
            print(f"⚠️ MR-PRESSO failed: {e}, continuing with standard MR...")
        
        mr_results = exposure_geno.MR(
            exposure_name=exposure_id, 
            outcome_name=outcome_id, 
            methods=["IVW", "IVW_RE", "UWR", "Egger", "WM", "Weighted-mode"], 
            heterogeneity=True, 
            use_mrpresso_data=True
        )
        
        try:
            exposure_geno.MR_plot(filename=f"MR_plot_{exposure_id}_to_{outcome_id}")
        except Exception as e:
            print(f"⚠️ MR plot generation failed: {e}")
        
        return {
            'success': True,
            'mr_results': mr_results,
            'exposure_data': exposure_geno.data,
            'analysis_type': 'external_to_external',
            'n_instruments': len(exposure_geno.data),
            'exposure_id': exposure_id,
            'outcome_id': outcome_id
        }
        
    except Exception as e:
        print(f"⚠️ MR analysis failed with error: {str(e)}")
        return {
            'success': False,
            'error': f'MR analysis failed: {str(e)}',
            'exposure_id': exposure_id,
            'outcome_id': outcome_id
        }

def run_exposure_to_phenotype_mr(exposure_id, phenotype_data, target_phenotype, covar_file):
    """Run MR from external exposure to constructed phenotype (requires GWAS on phenotype first)"""
    
    print(f"Running MR: {exposure_id} -> {target_phenotype}")
    print("Note: This requires running GWAS on the phenotype first to get outcome data")
    
    # Step 1: Would run GWAS on the phenotype to get summary statistics
    # Step 2: Use those summary statistics as outcome in MR
    
    return {
        'analysis_type': 'external_exposure_to_phenotype',
        'exposure': exposure_id,
        'outcome': target_phenotype,
        'status': 'requires_gwas_on_phenotype',
        'next_steps': [
            f"1. Run GWAS on {target_phenotype} using constructed phenotype data",
            f"2. Extract summary statistics for SNPs from {exposure_id}",
            "3. Perform MR analysis using genal.Geno methods"
        ]
    }

def run_phenotype_to_outcome_mr(phenotype_data, target_phenotype, covar_file, outcome_id):
    """Run MR from constructed phenotype to external outcome (requires GWAS on phenotype first)"""
    
    print(f"Running MR: {target_phenotype} -> {outcome_id}")
    print("Note: This requires running GWAS on the phenotype first to get exposure data")
    
    # Step 1: Would run GWAS on the phenotype to get summary statistics
    # Step 2: Use those summary statistics as exposure in MR
    
    return {
        'analysis_type': 'phenotype_to_external_outcome', 
        'exposure': target_phenotype,
        'outcome': outcome_id,
        'status': 'requires_gwas_on_phenotype',
        'next_steps': [
            f"1. Run GWAS on {target_phenotype} using constructed phenotype data",
            "2. Identify genome-wide significant SNPs (p < 5e-8)",
            "3. Clump SNPs to remove linkage disequilibrium",
            f"4. Query {outcome_id} for associations with these SNPs",
            "5. Perform MR analysis using genal.Geno methods"
        ]
    }

def run_phenotype_gwas(phenotype_data, target_phenotype, covar_file):
    """Run GWAS discovery on constructed phenotype"""
    
    print(f"Running GWAS discovery on {target_phenotype}")
    
    # This would interface with PLINK2 or other GWAS software
    # For now, return summary of what would be done
    
    gwas_summary = {
        'phenotype': target_phenotype,
        'sample_size': len(phenotype_data),
        'covariate_file': covar_file,
        'analysis_type': 'gwas_discovery',
        'next_steps': [
            f"Run: plink2 --bfile genotype_data --pheno {target_phenotype}_pheno.txt --pheno-name {target_phenotype}",
            f"Add: --covar {covar_file.split('/')[-1]} if covariates available",
            "Add: --glm to perform association testing",
            "Add: --out results_prefix for output files"
        ]
    }
    
    print("GWAS Discovery Analysis Summary:")
    print(f"  Phenotype: {target_phenotype}")
    print(f"  Sample size: {gwas_summary['sample_size']:,}")
    print(f"  Covariate file: {covar_file}")
    print("\nNext steps for GWAS:")
    for step in gwas_summary['next_steps']:
        print(f"  {step}")
    
    return gwas_summary

def search_and_run_two_sample_mr(exposure_trait, outcome_trait, exposure_description="", outcome_description=""):
    """
    Search for OpenGWAS IDs for two traits and run two-sample MR analysis
    
    Args:
        exposure_trait (str): Name of the exposure trait
        outcome_trait (str): Name of the outcome trait  
        exposure_description (str): Optional description of exposure
        outcome_description (str): Optional description of outcome
        
    Returns:
        dict: Complete MR analysis results including search results and causal estimates
    """
    
    if not OPENGWAS_SEARCH_AVAILABLE:
        return {
            'success': False,
            'error': 'OpenGWAS search functionality not available',
            'note': 'Please ensure llm_query module is properly imported'
        }
    
    print(f"=== Searching OpenGWAS for Two-Sample MR ===")
    print(f"Exposure: {exposure_trait}")
    print(f"Outcome: {outcome_trait}")
    
    # Search for exposure OpenGWAS ID
    print("\n1. Searching for exposure GWAS data...")
    exposure_search = search_opengwas_for_trait(
        trait_name=exposure_trait,
        trait_description=exposure_description,
        trait_type="EXPOSURE",
        context=f"Two-sample MR: {exposure_trait} -> {outcome_trait}"
    )
    
    # Search for outcome OpenGWAS ID  
    print("\n2. Searching for outcome GWAS data...")
    outcome_search = search_opengwas_for_trait(
        trait_name=outcome_trait,
        trait_description=outcome_description,
        trait_type="OUTCOME", 
        context=f"Two-sample MR: {exposure_trait} -> {outcome_trait}"
    )
    
    print(f"\nExposure search result: {exposure_search['opengwas_id']} (confidence: {exposure_search['confidence']:.2f})")
    print(f"Outcome search result: {outcome_search['opengwas_id']} (confidence: {outcome_search['confidence']:.2f})")
    
    # Check if both searches were successful
    if (exposure_search['opengwas_id'] == 'unknown' or 
        outcome_search['opengwas_id'] == 'unknown'):
        
        return {
            'success': False,
            'exposure_search': exposure_search,
            'outcome_search': outcome_search,
            'error': 'Could not find suitable OpenGWAS studies for one or both traits',
            'note': 'Consider using alternative trait names or checking OpenGWAS directly'
        }
    
    # Run two-sample MR analysis
    print(f"\n3. Running two-sample MR: {exposure_search['opengwas_id']} -> {outcome_search['opengwas_id']}")
    
    try:
        mr_results = run_external_mr(
            exposure_id=exposure_search['opengwas_id'],
            outcome_id=outcome_search['opengwas_id']
        )
        
        if mr_results and mr_results.get('success'):
            # Extract MR results from genal analysis
            mr_data = mr_results.get('mr_results', {})
            
            # Try to extract causal effect from different MR methods
            causal_effect = None
            confidence_interval = None
            p_value = None
            
            # Check for IVW results (most common method)
            if isinstance(mr_data, dict) and 'IVW' in mr_data:
                ivw_results = mr_data['IVW']
                causal_effect = ivw_results.get('beta', ivw_results.get('b'))
                se = ivw_results.get('se')
                p_value = ivw_results.get('pval', ivw_results.get('p'))
                
                if causal_effect is not None and se is not None:
                    confidence_interval = [
                        causal_effect - 1.96 * se,
                        causal_effect + 1.96 * se
                    ]
            
            return {
                'success': True,
                'exposure_search': exposure_search,
                'outcome_search': outcome_search,
                'mr_results': mr_results,
                'causal_effect': causal_effect,
                'confidence_interval': confidence_interval,
                'p_value': p_value,
                'method': 'Two-Sample Mendelian Randomization',
                'n_instruments': mr_results.get('n_instruments'),
                'exposure_gwas_id': exposure_search['opengwas_id'],
                'outcome_gwas_id': outcome_search['opengwas_id'],
                'note': f'Automated OpenGWAS search and two-sample MR: {exposure_trait} -> {outcome_trait}'
            }
        else:
            error_msg = mr_results.get('error', 'Unknown MR analysis error') if mr_results else 'MR analysis returned None'
            return {
                'success': False,
                'exposure_search': exposure_search,
                'outcome_search': outcome_search,
                'mr_results': mr_results,
                'error': f'MR analysis failed: {error_msg}',
                'note': 'OpenGWAS studies found but MR analysis encountered issues'
            }
            
    except Exception as e:
        return {
            'success': False,
            'exposure_search': exposure_search,
            'outcome_search': outcome_search,
            'error': f'MR analysis failed with error: {str(e)}',
            'note': 'Error occurred during MR computation'
        }

def estimate(data, treatment, outcome, covariates, llm_results=None, latent_confounders=None, **kwargs):
    """
    Main estimate function that interfaces with the causal inference pipeline.
    
    This function should be called by the inference classifier when MR is selected.
    It integrates PyGWAS analysis with Mendelian Randomization when appropriate.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataset for analysis
    treatment : str
        Treatment/exposure variable name
    outcome : str
        Outcome variable name
    covariates : list
        List of covariate names
    llm_results : dict, optional
        Results from LLM query containing GWAS variable mapping and OpenGWAS IDs
    latent_confounders : list, optional
        List of latent confounder nodes (U_ format) identified by FCI
    **kwargs : dict
        Additional parameters for MR analysis
    
    Returns:
    --------
    dict : MR analysis results with causal effect estimates
    """
    
    print("Mendelian Randomization Analysis")
    
    # Handle latent confounders
    mr_advantage_note = ""
    if latent_confounders:
        print(f"MR: Handling {len(latent_confounders)} latent confounders: {latent_confounders}")
        print("Note: MR estimation can handle unobserved confounding via genetic instruments")
        mr_advantage_note = f"MR estimation with {len(latent_confounders)} latent confounders detected. " \
                           "Genetic instruments provide robust identification despite unobserved confounding."
    
    # Extract OpenGWAS IDs from LLM results if available
    exposure_id = kwargs.get('exposure_id')
    outcome_id = kwargs.get('outcome_id')
    
    if llm_results:
        # Check for OpenGWAS IDs found by LLM search
        if 'opengwas_exposure_id' in llm_results:
            found_exposure_id = llm_results['opengwas_exposure_id']['opengwas_id']
            if found_exposure_id != 'unknown' and not exposure_id:
                exposure_id = found_exposure_id
                print(f"Using LLM-found exposure OpenGWAS ID: {exposure_id}")
                print(f"  Study: {llm_results['opengwas_exposure_id']['study_description']}")
                print(f"  Confidence: {llm_results['opengwas_exposure_id']['confidence']:.2f}")
                
        if 'opengwas_outcome_id' in llm_results:
            found_outcome_id = llm_results['opengwas_outcome_id']['opengwas_id']
            if found_outcome_id != 'unknown' and not outcome_id:
                outcome_id = found_outcome_id
                print(f"Using LLM-found outcome OpenGWAS ID: {outcome_id}")
                print(f"  Study: {llm_results['opengwas_outcome_id']['study_description']}")
                print(f"  Confidence: {llm_results['opengwas_outcome_id']['confidence']:.2f}")
    
    # Update kwargs with found IDs
    kwargs['exposure_id'] = exposure_id
    kwargs['outcome_id'] = outcome_id
    
    # Check if we can do two-sample MR with external GWAS data
    if exposure_id and outcome_id:
        print(f"Attempting two-sample MR: {exposure_id} -> {outcome_id}")
        try:
            mr_results = run_external_mr(exposure_id, outcome_id)
            if mr_results and mr_results.get('success'):
                print("✅ Two-sample MR analysis completed successfully")
                
                # Extract MR results from genal analysis
                mr_data = mr_results.get('mr_results', {})
                
                # Try to extract causal effect from different MR methods
                causal_effect = None
                confidence_interval = None
                p_value = None
                mr_method = 'IVW'  # Default method
                
                # Check for IVW results (most common method)
                if isinstance(mr_data, dict) and 'IVW' in mr_data:
                    ivw_results = mr_data['IVW']
                    causal_effect = ivw_results.get('beta', ivw_results.get('b'))
                    se = ivw_results.get('se')
                    p_value = ivw_results.get('pval', ivw_results.get('p'))
                    
                    if causal_effect is not None and se is not None:
                        confidence_interval = [
                            causal_effect - 1.96 * se,
                            causal_effect + 1.96 * se
                        ]
                
                return {
                    'causal_effect': causal_effect,
                    'confidence_interval': confidence_interval,
                    'p_value': p_value,
                    'method': 'Mendelian Randomization (Two-Sample External)',
                    'mr_method': mr_method,
                    'n_instruments': mr_results.get('n_instruments'),
                    'mr_results_full': mr_data,  # Include full results for inspection
                    'exposure_gwas_id': exposure_id,
                    'outcome_gwas_id': outcome_id,
                    'note': f'Two-sample MR using OpenGWAS studies: {exposure_id} -> {outcome_id}'
                }
            else:
                error_msg = mr_results.get('error', 'Unknown error') if mr_results else 'MR analysis returned None'
                print(f"⚠️ Two-sample MR failed: {error_msg}")
                print("Falling back to integrated analysis...")
        except Exception as e:
            print(f"⚠️ Two-sample MR failed: {e}")
            print("Falling back to integrated analysis...")
    
    # Check if this is questionnaire data that would benefit from PyGWAS
    has_questionnaire_genetics = (llm_results and 
                                llm_results.get('is_questionnaire_with_genetics', {}).get('value') == 'Yes')
    
    has_gwas_mapping = (llm_results and 
                       llm_results.get('gwas_variable_mapping') is not None)
    
    # Get GWAS compatibility information from LLM results
    dataset_variable_type = None
    if llm_results and 'gwas_target_type' in llm_results:
        dataset_variable_type = llm_results.get('gwas_target_type', {}).get('value')
        print(f"LLM determined dataset variable type: {dataset_variable_type}")
    
    # Determine which OpenGWAS ID to use based on dataset variable type
    if dataset_variable_type == "EXPOSURE" and outcome_id and not exposure_id:
        print(f"Dataset is EXPOSURE, using provided outcome OpenGWAS ID: {outcome_id}")
    elif dataset_variable_type == "OUTCOME" and exposure_id and not outcome_id:
        print(f"Dataset is OUTCOME, using provided exposure OpenGWAS ID: {exposure_id}")
    elif dataset_variable_type:
        print(f"Dataset is {dataset_variable_type} - will use appropriate analysis approach")
    
    # Try PyGWAS integration first if we have questionnaire data with genetics
    if has_questionnaire_genetics or has_gwas_mapping:
        try:
            print("Attempting PyGWAS -> MR integrated analysis...")
            
            # Extract GWAS mapping if available
            gwas_mapping = None
            if has_gwas_mapping:
                gwas_mapping = llm_results.get('gwas_variable_mapping')
            
            # Run integrated PyGWAS -> MR analysis
            mr_results = run_integrated_pygwas_mr(
                df=data,  # Use 'df' parameter name as expected
                gwas_mapping=gwas_mapping,
                exposure_id=kwargs.get('exposure_id'),
                outcome_id=kwargs.get('outcome_id'),
                genotype_data=kwargs.get('genotype_data')
            )
            
            if mr_results and mr_results.get('success'):
                print("✅ PyGWAS -> MR analysis completed successfully")
                return {
                    'causal_effect': mr_results.get('causal_effect'),
                    'confidence_interval': mr_results.get('confidence_interval'),
                    'p_value': mr_results.get('p_value'),
                    'method': 'Mendelian Randomization (PyGWAS Integrated)',
                    'mr_method': mr_results.get('mr_method'),
                    'n_instruments': mr_results.get('n_instruments'),
                    'f_statistics': mr_results.get('f_statistics'),
                    'heterogeneity': mr_results.get('heterogeneity'),
                    'pleiotropy_test': mr_results.get('pleiotropy_test'),
                    'note': 'Analysis using PyGWAS for instrument discovery and MR for causal inference'
                }
            else:
                print("⚠️ PyGWAS -> MR analysis failed, falling back to external GWAS...")
                
        except Exception as e:
            print(f"⚠️ PyGWAS integration failed: {e}")
            print("Falling back to external GWAS sources...")
    
    # Fallback to external GWAS database approach
    try:
        print("Using external GWAS database for MR analysis...")
        
        # Look for instruments in the LLM results or use treatment variable
        instruments = []
        if llm_results and 'instruments' in llm_results:
            instruments_str = llm_results['instruments']['value']
            if instruments_str != 'None':
                instruments = [inst.strip() for inst in instruments_str.split(',')]
        
        # Use external GWAS -> MR pipeline  
        # For this fallback, we'll use a simplified approach
        # since we don't have PyGWAS outcome instruments
        exposure_id = kwargs.get('exposure_id', treatment)
        outcome_instruments = pd.DataFrame()  # Empty - would need PyGWAS to populate
        outcome_phenotype = outcome
        
        if not outcome_instruments.empty:
            mr_results = run_external_to_pygwas_mr(
                exposure_id=exposure_id,
                outcome_instruments=outcome_instruments,
                outcome_phenotype=outcome_phenotype
            )
        else:
            # Skip this approach if we don't have outcome instruments
            mr_results = None
        
        if mr_results and mr_results.get('success'):
            print("✅ External GWAS -> MR analysis completed")
            return {
                'causal_effect': mr_results.get('causal_effect'),
                'confidence_interval': mr_results.get('confidence_interval'),
                'p_value': mr_results.get('p_value'),
                'method': 'Mendelian Randomization (External GWAS)',
                'mr_method': mr_results.get('mr_method'),
                'n_instruments': mr_results.get('n_instruments'),
                'f_statistics': mr_results.get('f_statistics'),
                'heterogeneity': mr_results.get('heterogeneity'),
                'pleiotropy_test': mr_results.get('pleiotropy_test'),
                'note': 'Analysis using external GWAS databases for instrument discovery'
            }
        
    except Exception as e:
        print(f"⚠️ External GWAS MR analysis failed: {e}")
    
    # Final fallback - basic diagnostic information
    print("⚠️ All MR approaches failed. Providing diagnostic information...")
    
    return {
        'causal_effect': None,
        'confidence_interval': None,
        'p_value': None,
        'method': 'Mendelian Randomization (Diagnostic Only)',
        'note': 'MR analysis requires either genetic data in questionnaire or external GWAS IDs. Please ensure data contains genetic variants or provide OpenGWAS IDs for exposure and outcome.',
        'diagnostic_info': {
            'has_questionnaire_genetics': has_questionnaire_genetics,
            'has_gwas_mapping': has_gwas_mapping,
            'pygwas_available': _check_pygwas_availability(),
            'genal_available': True  # genal is imported at top of file
        }
    }

def diagnose(data, treatment, outcome, covariates, alpha=0.05):
    """
    Comprehensive Mendelian Randomization diagnostic tests.
    
    Returns:
    --------
    dict : Dictionary with diagnostic test results
    """
    
    diagnostics = {
        'genetic_data_available': False,
        'sufficient_instruments': False,
        'relevance_assumption': False,
        'independence_assumption': True,
        'exclusion_restriction': True,
        'population_stratification': True,
        'overall_valid': False
    }
    
    try:
        # Check if genetic data is available in the dataset
        genetic_columns = [col for col in data.columns if col.startswith(('rs', 'chr', 'snp', 'SNP', 'variant'))]
        
        # Check for standard genetic data columns
        gwas_columns = ['CHR', 'POS', 'SNP', 'A1', 'A2', 'BETA', 'SE', 'P', 'EAF']
        has_gwas_format = any(col in data.columns for col in gwas_columns)
        
        # Check for genotype data (columns with genetic variant patterns)
        genotype_pattern_cols = [col for col in data.columns if 
                               col.startswith(('rs', 'chr')) or 
                               '_' in col and any(x in col.lower() for x in ['snp', 'variant', 'allele'])]
        
        diagnostics['genetic_data_available'] = (
            len(genetic_columns) > 0 or 
            has_gwas_format or 
            len(genotype_pattern_cols) > 0
        )
        
        # Check for sufficient number of potential instruments
        # MR typically requires at least 3-10 strong instruments
        potential_instruments = genetic_columns + genotype_pattern_cols
        diagnostics['sufficient_instruments'] = len(potential_instruments) >= 3
        
        # Test relevance assumption (instruments should be associated with exposure)
        if potential_instruments and treatment in data.columns:
            # Check if any genetic variants are correlated with treatment
            treatment_data = pd.to_numeric(data[treatment], errors='coerce')
            correlations = []
            
            for instrument in potential_instruments[:10]:  # Check first 10 instruments
                try:
                    instrument_data = pd.to_numeric(data[instrument], errors='coerce')
                    if not instrument_data.isna().all():
                        corr, p_val = stats.pearsonr(
                            treatment_data.dropna(), 
                            instrument_data.dropna()
                        )
                        if not np.isnan(corr) and abs(corr) > 0.01:  # Minimum effect size
                            correlations.append(abs(corr))
                except:
                    continue
            
            # At least some instruments should show association with treatment
            diagnostics['relevance_assumption'] = len(correlations) > 0 and max(correlations) > 0.05
        
        # Independence assumption (check for population structure)
        # Look for principal components or population indicators
        pc_columns = [col for col in data.columns if col.upper().startswith('PC') and col[2:].isdigit()]
        ancestry_columns = [col for col in data.columns if any(x in col.lower() for x in ['ancestry', 'population', 'ethnicity'])]
        
        # Having population controls is good for MR
        diagnostics['population_stratification'] = len(pc_columns) >= 5 or len(ancestry_columns) > 0
        
        # Exclusion restriction is largely untestable but assume valid if other checks pass
        # In practice, this requires biological knowledge
        
        # Overall validity
        diagnostics['overall_valid'] = (
            diagnostics['genetic_data_available'] and
            diagnostics['sufficient_instruments'] and
            diagnostics['relevance_assumption'] and
            diagnostics['population_stratification']
        )
        
    except Exception as e:
        print(f"Mendelian Randomization diagnostic failed: {e}")
        diagnostics['overall_valid'] = False
    
    return diagnostics

# exposure_data = query.tophits(
#     id = ["ukb-b-9942"],
#     pval = 5e-8,
#     clump = 1,
#     r2 = 0.01,
#     kb = 10000
# )
# exposure_df = pd.DataFrame(exposure_data)

# exposure_geno = genal.Geno(
#     exposure_df,
#     CHR="chr",
#     POS="position", 
#     SNP="rsid",
#     EA="ea",
#     NEA="nea", 
#     BETA="beta",
#     SE="se",
#     P="p",
#     EAF="eaf"
# ) 
# snp_list = exposure_geno.data["SNP"].tolist()

# # Perform Cochran's Q heterogeneity test (p < 0.05 indicates heterogeneity, I2 > 25% indivates moderate to high heterogeneity)
# beta_x = exposure_geno.data["beta"].to_numpy()
# se_x = exposure_geno.data["se"].to_numpy()
# beta_y = exposure_geno.data["beta_2"].to_numpy()
# se_y = exposure_geno.data["se_2"].to_numpy()

# # If using OpenGWAS data as outcome
# outcome_data = query.associations(
#     variant = snp_list,
#     id = ["ukb-a-13"],
#     proxies=1,
#     r2=0.8
# )
# outcome_df = pd.DataFrame(outcome_data)

# # If using PLINK2 custom GWAS data
# # plink_glm   = r"path_to_PLINK2_glm_file"
# # use_cols    = ["CHR","POS","ID","REF","ALT","TEST","BETA","SE","P","A1_FREQ"]

# # glm_raw = pd.read_csv(plink_glm, sep=r"\s+", usecols=use_cols)
# # glm_raw = glm_raw[glm_raw["TEST"] == "ADD"]              # additive model only
# # glm_raw = glm_raw[glm_raw["ID"].isin(snp_list)]          # keep overlapping SNPs

# # outcome_df = glm_raw.rename(columns={
# #     "CHR":"chr","POS":"position","ID":"rsid",
# #     "ALT":"ea","REF":"nea",
# #     "BETA":"beta","SE":"se","P":"p","A1_FREQ":"eaf"
# # })

# outcome_geno = genal.Geno(
#     df=outcome_df,
#     CHR="chr",
#     POS="position",
#     SNP="rsid", 
#     EA="ea",
#     NEA="nea",
#     BETA="beta",
#     SE="se",
#     P="p",
#     EAF="eaf"
# )

# exposure_geno.query_outcome(   # Haromize exposure and outcome SNPs
#     outcome=outcome_geno,
#     proxy=0,
#     r2=0.8,
#     kb=5000
# )

# # Filter low F-stat SNPs
# dg = exposure_geno.data
# exposure_geno.data["F"] = (dg["beta"] / dg["se"]) ** 2
# exposure_geno.data = exposure_geno.data[exposure_geno.data["F"] >= 10]
# print("\nF-stat summary after filtering:")
# print(exposure_geno.data["F"].describe())

# exposure_geno.MRpresso(action=2, n_iterations=30000)
# print(exposure_geno.data)

# exposure_geno.MR(exposure_name="Air Quality", outcome_name="Sleeplessness/Insomnia", methods=["IVW", "IVW_RE", "UWR", "Egger", "WM", "Weighted-mode"], heterogeneity=True, use_mrpresso_data=True)
# exposure_geno.MR_plot(filename="MR_plot")