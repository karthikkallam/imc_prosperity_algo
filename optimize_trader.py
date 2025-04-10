# optimize_trader.py
# Script to automatically tune parameters for IMC Prosperity 3 Trader using Optuna
# Version with corrected PnL parsing

import optuna
import subprocess
import json
import os
import shutil # Import the shutil module for file copying
import re
import tempfile
import traceback

# --- Configuration ---
TRADER_FILE_ORIGINAL = "trader.py" # Your main trader script file name
DATAMODEL_FILE_ORIGINAL = "datamodel.py" # Your datamodel file name
STUDY_NAME = "prosperity3-tutorial-optimizer" # Name for the Optuna study
STORAGE_DB = "sqlite:///prosperity3_optimization.db" # SQLite database file to store results
N_TRIALS = 10  # << SET THE NUMBER OF TRIALS TO RUN >> (e.g., 50, 100, 200+)
ROUND_TO_TEST = "0" # Tutorial round

# --- Parameter Search Space ---
# Define the ranges for Optuna to explore.
# Adjust these ranges based on your testing and intuition!
def define_search_space(trial: optuna.Trial) -> dict:
    """Defines the hyperparameter search space for Optuna."""
    params = {
        "shared": {
            "take_profit_threshold": trial.suggest_float("shared_take_profit", 0.35, 1.0, step=0.01),
            "max_history_length": trial.suggest_int("shared_max_hist", 80, 100, step=1)
        },
        "RAINFOREST_RESIN": {
            "anchor_blend_alpha": trial.suggest_float("r_anchor_alpha", 0.05, 0.15, step=0.002),
            "min_spread": trial.suggest_int("r_min_spread", 6, 8),
            "volatility_spread_factor": trial.suggest_float("r_vol_spread", 0.1, 0.4, step=0.01),
            "inventory_skew_factor": trial.suggest_float("r_skew", 0.0, 0.02, step=0.001),
            "base_order_qty": trial.suggest_int("r_qty", 30, 48, step=1),
            "reversion_threshold": trial.suggest_int("r_revert", 1, 7)
        },
        "KELP": {
            "ema_alpha": trial.suggest_float("k_ema_alpha", 0.0, 1.2, step=0.05),
            "min_spread": trial.suggest_int("k_min_spread", 2, 3),
            "volatility_spread_factor": trial.suggest_float("k_vol_spread", 0.2, 2.5, step=0.1),
            "inventory_skew_factor": trial.suggest_float("k_skew", 0, 0.2, step=0.002),
            "base_order_qty": trial.suggest_int("k_qty", 15, 40, step=1),
            "min_volatility_qty_factor": trial.suggest_float("k_min_qty_f", 1.5, 1.8, step=0.05),
            "max_volatility_for_qty_reduction": trial.suggest_float("k_max_vol", 2.0, 7.0, step=0.5),
            "imbalance_depth": trial.suggest_int("k_imb_depth", 4, 5),
            "imbalance_fv_adjustment_factor": trial.suggest_float("k_imb_adj", 0.0, 0.7, step=0.02)
        }
    }
    params["RAINFOREST_RESIN"]["fair_value_anchor"] = 10000.0
    return params

# --- Objective Function ---
def objective(trial: optuna.Trial) -> float:
    """Runs a backtest trial with parameters suggested by Optuna."""
    current_params = define_search_space(trial)
    temp_dir = None

    try:
        temp_dir = tempfile.mkdtemp()
        temp_trader_file = os.path.join(temp_dir, os.path.basename(TRADER_FILE_ORIGINAL))
        temp_datamodel_file = os.path.join(temp_dir, os.path.basename(DATAMODEL_FILE_ORIGINAL))

        if not os.path.exists(TRADER_FILE_ORIGINAL):
            print(f"ERROR: Original trader file '{TRADER_FILE_ORIGINAL}' not found.")
            return -float('inf')
        with open(TRADER_FILE_ORIGINAL, 'r') as f:
            original_code = f.read()

        params_start_index = original_code.find("PARAMS = {")
        if params_start_index == -1: raise ValueError("Could not find 'PARAMS = {'")
        brace_level = 0; params_end_index = -1; in_string_literal = False; string_char_literal = ''; escaped = False
        search_start = params_start_index + len("PARAMS = {") -1
        for i in range(search_start, len(original_code)):
            char = original_code[i]
            if not escaped:
                if char in ('"', "'"):
                    if not in_string_literal: in_string_literal = True; string_char_literal = char
                    elif char == string_char_literal: in_string_literal = False
                elif char == '\\': escaped = True; continue
            if not in_string_literal:
                if char == '{':
                    if i == search_start : brace_level = 1
                    else: brace_level += 1
                elif char == '}':
                    brace_level -= 1
                    if brace_level == 0: params_end_index = i + 1; break
            escaped = False
        if params_end_index == -1: raise ValueError("Could not find matching '}' for PARAMS")

        params_string = f"PARAMS = {json.dumps(current_params, indent=4)}"
        modified_code = original_code[:params_start_index] + params_string + original_code[params_end_index:]

        with open(temp_trader_file, 'w') as tmp_f:
            tmp_f.write(modified_code)

        if os.path.exists(DATAMODEL_FILE_ORIGINAL):
            shutil.copyfile(DATAMODEL_FILE_ORIGINAL, temp_datamodel_file)
        else:
            print(f"ERROR: Original datamodel file '{DATAMODEL_FILE_ORIGINAL}' not found.")
            return -float('inf')

        cmd = ["prosperity3bt", temp_trader_file, ROUND_TO_TEST, "--merge-pnl", "--no-out"]
        print(f"Running Trial {trial.number}...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=300)

        pnl = -float('inf')
        if result.returncode == 0:
             # **MODIFIED REGEX and Parsing**
             # Look for "Total profit:" followed by optional whitespace, then numbers and commas
             match = re.search(r"Total profit:\s*([+-]?[\d,]+(?:\.\d+)?)", result.stdout)
             if match:
                 try:
                     pnl_str = match.group(1).replace(',', '') # Remove commas before converting
                     pnl = float(pnl_str)
                     # print(f"Trial {trial.number} completed. PnL: {pnl}") # Optuna logs this
                 except ValueError:
                     print(f"Warning: Could not parse PnL float from string '{match.group(1)}' for trial {trial.number}.")
             else:
                  print(f"Warning: Could not find 'Total profit:' pattern in stdout for trial {trial.number}.")
                  # print(f"STDOUT Snippet (Trial {trial.number}):\n{result.stdout[-500:]}\n---") # Optional debug: print last part of output
        else:
             print(f"Error Trial {trial.number}: Backtester failed (Code {result.returncode}).")
             stderr_snippet = result.stderr[:1000] + ('...' if len(result.stderr) > 1000 else '')
             print(f"STDERR Snippet: {stderr_snippet}")
             raise optuna.exceptions.TrialPruned(f"Backtester failed with code {result.returncode}")

        return pnl

    except optuna.exceptions.TrialPruned as e:
        print(f"Trial {trial.number} pruned: {e}")
        raise
    except Exception as e:
        print(f"Error in objective function for trial {trial.number}: {e}")
        traceback.print_exc()
        return -float('inf')
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except OSError as e:
                print(f"Error removing temporary directory {temp_dir}: {e}")

# --- Main Optimization Execution ---
if __name__ == "__main__":
    # (Main execution block remains the same)
    print(f"\n--- Starting Optuna Optimization ---")
    print(f"Study Name: {STUDY_NAME}")
    print(f"Storage:    {STORAGE_DB}")
    print(f"Trials:     {N_TRIALS}")
    print(f"Trader:     {TRADER_FILE_ORIGINAL}")
    print(f"Datamodel:  {DATAMODEL_FILE_ORIGINAL}")
    print("-" * 35)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_DB,
        load_if_exists=True,
        direction="maximize"
    )

    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=None, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\nOptimization stopped manually.")
    except Exception as e:
        print(f"\nAn optimization error occurred: {e}")
        traceback.print_exc()

    print("\n--- Optimization Results ---")
    print(f"Number of finished trials: {len(study.trials)}")

    try:
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
             print("No trials completed successfully. Cannot determine best parameters.")
        else:
            best_trial = study.best_trial
            print(f"Best trial number: {best_trial.number}")
            print(f"Best trial value (Max PnL): {best_trial.value:.2f}")
            print("\nBest parameters found:")
            print(json.dumps(best_trial.params, indent=4))

            if optuna.visualization.is_available():
                print("\n--- Visualizations ---")
                print("(Plots might open in your browser)")
                try:
                    fig_history = optuna.visualization.plot_optimization_history(study)
                    fig_importance = optuna.visualization.plot_param_importances(study)
                    fig_slice = optuna.visualization.plot_slice(study)
                    fig_history.show()
                    fig_importance.show()
                    fig_slice.show()
                    print("\nYou can also explore interactively:")
                    print(f"  optuna-dashboard {STORAGE_DB}")
                except Exception as ve:
                    print(f"\nError generating visualization: {ve}")
                    print("You can still explore results using:")
                    print(f"  optuna-dashboard {STORAGE_DB}")
            else:
                print("\nInstall plotly and matplotlib for visualizations:")
                print("  pip install plotly matplotlib")
                print("\nExplore results using:")
                print(f"  optuna-dashboard {STORAGE_DB}")

    except ValueError:
         print("No trials completed successfully yet (or study is empty).")
    except Exception as e:
        print(f"An error occurred during final reporting: {e}")