import sys
import os
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from src.utils.traci_interface import start_simulation, stop_simulation, collect_metrics, step
from src.controller import run_framework

os.makedirs('results/runs', exist_ok=True)

scenarios = ['low', 'medium', 'peak']
methods   = ['WOA', 'HHO', 'GA']
N_RUNS    = 5

for SCENARIO in scenarios:
    CONFIG = f'sumo_network/blida_{SCENARIO}.sumocfg'
    print(f"\n{'='*60}")
    print(f"5x RUNS — {SCENARIO.upper()}")
    print(f"{'='*60}")

    for METHOD in methods:
        print(f"\n>>> {METHOD} — {SCENARIO} — 5 runs")
        wait_list  = []
        stops_list = []
        trip_list  = []
        arr_list   = []

        for run in range(1, N_RUNS + 1):
            print(f"  Run {run}/{N_RUNS}...")
            save_path = f'results/runs/{METHOD.lower()}_{SCENARIO}_run{run}.csv'
            avg_wait, max_q, arrived = run_framework(
                CONFIG, algorithm=METHOD,
                save_path=save_path)

            df = pd.read_csv(save_path)
            wait_list.append(df['total_waiting_time'].mean())
            stops_list.append(df['total_stops'].mean())
            trip_list.append(df['total_trip_time'].mean())
            arr_list.append(df['arrived'].sum())

        # Save summary
        summary = {
            'method':   METHOD,
            'scenario': SCENARIO,
            'wait_mean':  np.mean(wait_list),
            'wait_std':   np.std(wait_list),
            'stops_mean': np.mean(stops_list),
            'stops_std':  np.std(stops_list),
            'trip_mean':  np.mean(trip_list),
            'trip_std':   np.std(trip_list),
            'arr_mean':   np.mean(arr_list),
            'arr_std':    np.std(arr_list),
        }

        print(f"  {METHOD} {SCENARIO}: "
              f"wait={summary['wait_mean']:.1f} ± {summary['wait_std']:.1f}s")

        # Append to master summary
        summary_path = 'results/runs/summary_5x.csv'
        df_sum = pd.DataFrame([summary])
        if os.path.exists(summary_path):
            df_sum.to_csv(summary_path, mode='a', header=False, index=False)
        else:
            df_sum.to_csv(summary_path, index=False)

print("\n ALL 5x RUNS DONE!")
print("Results saved to results/runs/summary_5x.csv")