import optuna
import pandas as pd
import re
import glob

# ==========================================
# 1. Configuration
# ==========================================
base_dir = "result_steering_fa_p"
optuna_journal_file = f"{base_dir}/bias_steering_journal.log"
slurm_file_pattern = f"{base_dir}/result*.out"  
study_name = "LLaVA_Bias_Reduction" 
threshold = 0.3  

# ==========================================
# 2. Parse Active Configs from MULTIPLE Slurm Logs
# ==========================================
def get_active_configs_from_slurm(file_pattern):
    print(f"Scanning for Slurm logs matching: '{file_pattern}'...")
    filepaths = glob.glob(file_pattern)
    
    if not filepaths:
        print(f"Warning: No files found matching pattern '{file_pattern}'.")
        return {}

    active_configs = {}
    
    for filepath in filepaths:
        print(f"  -> Parsing {filepath}...")
        current_trial = None

        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    trial_match = re.match(r'\[Trial (\d+)\] Testing Configuration:', line)
                    if trial_match:
                        current_trial = int(trial_match.group(1))
                        active_configs[current_trial] = []
                        continue

                    if current_trial is not None:
                        if line.startswith("- ("):
                            # NEW REGEX: Extracts the SAE name, neuron ID, and Alpha value separately
                            # Example line: "- ('batch_topk_sae_22', 14423) | Neuron (...) | Alpha: 50.00"
                            match = re.search(r"-\s*\('([^']+)',\s*(\d+)\).*?Alpha:\s*([-\d.]+)", line)
                            if match:
                                sae_name = match.group(1)            # e.g., 'batch_topk_sae_22'
                                neuron_idx = int(match.group(2))     # e.g., 14423
                                alpha_val = float(match.group(3))    # e.g., 50.00
                                
                                # Append as a pure Python tuple
                                active_configs[current_trial].append((sae_name, neuron_idx, alpha_val))
                        elif line == "":
                            continue 
                        else:
                            current_trial = None
                            
        except Exception as e:
            print(f"  [!] Error reading {filepath}: {e}")
            
    print(f"Successfully extracted configurations for {len(active_configs)} total trials.")
    return active_configs

slurm_configs = get_active_configs_from_slurm(slurm_file_pattern)

# ==========================================
# 3. Load the Optuna Study
# ==========================================
print(f"\nLoading Optuna study '{study_name}'...")
storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(optuna_journal_file))

try:
    study = optuna.load_study(study_name=study_name, storage=storage)
except KeyError:
    print(f"Error: Study '{study_name}' not found. Check the exact name.")
    exit()

# ==========================================
# 4. Filter, Map, and De-Duplicate Data
# ==========================================
df = study.trials_dataframe()

good_trials = df[(df['state'] == 'COMPLETE') & (df['value'] < threshold)].copy()

if good_trials.empty:
    print(f"\nNo trials found with a J-Score below {threshold}.")
else:
    # 1. Sort them by best score first
    good_trials = good_trials.sort_values(by='value')

    # 2. Map the active configurations as a pure list of tuples
    good_trials['Active_Interventions'] = good_trials['number'].map(
        lambda x: slurm_configs.get(x, [])
    )

    # 3. Drop duplicates
    # We must convert lists to tuples temporarily because Pandas cannot hash a list for drop_duplicates
    good_trials['Hashable_Interventions'] = good_trials['Active_Interventions'].apply(tuple)
    
    initial_count = len(good_trials)
    good_trials = good_trials.drop_duplicates(subset=['Hashable_Interventions'], keep='first')
    removed_count = initial_count - len(good_trials)
    
    # Clean up the temporary column
    good_trials = good_trials.drop(columns=['Hashable_Interventions'])
    
    if removed_count > 0:
        print(f"Cleaned up {removed_count} functionally identical configurations.")

    display_cols = ['number', 'value', 'Active_Interventions']
    attr_cols = [col for col in good_trials.columns if col.startswith('user_attrs_')]
    
    final_output = good_trials[display_cols + attr_cols]

    # ==========================================
    # 5. Display and Save
    # ==========================================
    print("\n" + "="*100)
    print(f" 🏆 FOUND {len(final_output)} UNIQUE CONFIGS BELOW {threshold} J-SCORE 🏆")
    print("="*100 + "\n")
    
    print(final_output.to_string(index=False))
    
    csv_filename = f"{base_dir}/best_trials.csv"
    final_output.to_csv(csv_filename, index=False)
    print(f"\nSaved filtered results and active configurations to '{csv_filename}'")