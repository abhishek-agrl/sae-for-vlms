import optuna
import pandas as pd
import re
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. Configuration
# ==========================================
# Map your custom names to their respective base directories
profession_dirs = {
    "ComputerProg_Receptionist": "result_steering_cp_r",
    "Doctor_Nurse": "result_steering_d_n",
    "Pilot_FlightAttendant": "result_steering_fa_p"
}

study_name = "LLaVA_Bias_Reduction" 
threshold = 0.2  # Increased threshold to cast a wider net for commonalities

# ==========================================
# 2. Parsing Function
# ==========================================
def get_active_configs_from_slurm(file_pattern):
    filepaths = glob.glob(file_pattern)
    active_configs = {}
    
    for filepath in filepaths:
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
                            match = re.search(r"-\s*\('([^']+)',\s*(\d+)\).*?Alpha:\s*([-\d.]+)", line)
                            if match:
                                sae_name = match.group(1)
                                neuron_idx = int(match.group(2))
                                alpha_val = float(match.group(3))
                                active_configs[current_trial].append((sae_name, neuron_idx, alpha_val))
                        elif line == "":
                            continue 
                        else:
                            current_trial = None
        except Exception as e:
            pass
    return active_configs

# ==========================================
# 3. Process All Directories
# ==========================================
all_good_trials = []

for task_name, base_dir in profession_dirs.items():
    print(f"\n[{task_name}] Processing directory: {base_dir}...")
    
    optuna_file = f"{base_dir}/bias_steering_journal.log"
    slurm_pattern = f"{base_dir}/result*.out"
    
    if not os.path.exists(optuna_file):
        print(f"  [!] Optuna log not found: {optuna_file}. Skipping.")
        continue

    # 1. Parse Slurm
    slurm_configs = get_active_configs_from_slurm(slurm_pattern)
    
    # 2. Load Optuna Study
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(optuna_file))
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        print(f"  [!] Study '{study_name}' not found in {optuna_file}. Skipping.")
        continue
        
    # 3. Filter and Map
    df = study.trials_dataframe()
    good = df[(df['state'] == 'COMPLETE') & (df['value'] < threshold)].copy()
    
    if good.empty:
        print(f"  [-] No trials below {threshold} J-Score.")
        continue
        
    good['Active_Interventions'] = good['number'].map(lambda x: slurm_configs.get(x, []))
    
    # 4. De-duplicate locally
    good['Hashable'] = good['Active_Interventions'].apply(tuple)
    good = good.sort_values(by='value').drop_duplicates(subset=['Hashable'], keep='first')
    
    # 5. Add task identifier and append
    good['Task'] = task_name
    all_good_trials.append(good)
    print(f"  [+] Found {len(good)} unique good configurations.")

if not all_good_trials:
    print("\nNo data found across any directories to plot.")
    exit()

# Combine everything into one master dataframe
master_df = pd.concat(all_good_trials, ignore_index=True)

# Save Master CSV
master_csv = "global_best_trials.csv"
display_cols = ['Task', 'number', 'value', 'Active_Interventions']
master_df[display_cols].to_csv(master_csv, index=False)
print(f"\nSaved global aggregated data to '{master_csv}'")

# ==========================================
# 4. Analyze & Plot Feature Importance
# ==========================================
print("\nGenerating Cross-Task Analysis...")

# "Explode" the list of tuples so each neuron gets its own row
exploded_data = []
for _, row in master_df.iterrows():
    task = row['Task']
    j_score = row['value']
    for intervention in row['Active_Interventions']:
        sae_name, neuron_idx, alpha = intervention
        exploded_data.append({
            'Task': task,
            'J_Score': j_score,
            'Neuron_ID': f"{sae_name}_{neuron_idx}",
            'Alpha': alpha
        })

if not exploded_data:
    print("No active interventions found to plot.")
    exit()

feature_df = pd.DataFrame(exploded_data)

# Calculate how often each neuron is used per task
frequency_pivot = pd.crosstab(feature_df['Neuron_ID'], feature_df['Task'])

# Sort by total occurrences across all tasks to show the most important ones at the top
frequency_pivot['Total'] = frequency_pivot.sum(axis=1)
frequency_pivot = frequency_pivot.sort_values(by='Total', ascending=False).drop(columns=['Total'])

# Plot 1: Heatmap of Neuron Frequency
plt.figure(figsize=(10, 8))
sns.heatmap(frequency_pivot, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5)
plt.title(f"Frequency of Neuron Usage in Successful Trials (J-Score < {threshold})")
plt.ylabel("SAE Layer & Neuron Index")
plt.xlabel("Profession Task")
plt.tight_layout()
plt.savefig("neuron_frequency_heatmap.png", dpi=300)
print("Saved 'neuron_frequency_heatmap.png'")

# Plot 2: Alpha Distributions for Top Neurons
# Take the top 5 most frequently used neurons across all tasks
top_neurons = frequency_pivot.sum(axis=1).sort_values(ascending=False).head(10).index
top_features_df = feature_df[feature_df['Neuron_ID'].isin(top_neurons)]

plt.figure(figsize=(12, 6))
sns.boxplot(data=top_features_df, x='Neuron_ID', y='Alpha', hue='Task')
# sns.stripplot(data=top_features_df, x='Neuron_ID', y='Alpha', hue='Task', dodge=True, color='black', alpha=0.5)
plt.title(f"Distribution of Alpha Values for Top {len(top_neurons)} Neurons")
plt.xticks(rotation=45)
plt.axhline(0, color='red', linestyle='--', alpha=0.5) # Add a line at zero
plt.tight_layout()
plt.savefig("top_neurons_alpha_distribution.png", dpi=300)
print("Saved 'top_neurons_alpha_distribution.png'")

# --- Plot 3: Correlation between Alpha and J-Score ---
# 1. Safely grab strictly the FIRST string from the top_neurons index
number_one_neuron = top_neurons[0]

# 2. Filter the dataframe (standard equality works fine now that it's just one string)
best_neuron_data = feature_df[feature_df['Neuron_ID'] == number_one_neuron]

# 3. Clean the name for the title
clean_name = number_one_neuron.replace('batch_topk_sae_', 'L').replace('topk_sae_', 'L').replace('_', ' Neuron ')

# 4. Generate the plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=best_neuron_data, x='Alpha', y='J_Score', hue='Task', 
                s=100, alpha=0.7, edgecolor='black')

plt.title(f"Performance Correlation for {clean_name}")
plt.xlabel("Alpha (Steering Magnitude)")
plt.ylabel("J-Score (Lower is Better Parity)")

# Draw a line showing your success threshold
plt.axhline(threshold, color='red', linestyle=':', alpha=0.6, label=f'Threshold ({threshold})')
plt.legend()

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("alpha_vs_jscore_scatter.png", dpi=300)
print("Saved 'alpha_vs_jscore_scatter.png'")