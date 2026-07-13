import sys
import os
sys.path.insert(0, '.')

import pandas as pd
from src.utils.traci_interface import start_simulation, stop_simulation, collect_metrics, step
from src.controller import run_framework

os.makedirs('results', exist_ok=True)

scenarios = ['low', 'medium', 'peak']

for SCENARIO in scenarios:
    CONFIG = f'sumo_network/blida_{SCENARIO}.sumocfg'
    print(f"\n{'='*60}")
    print(f"SCENARIO: {SCENARIO.upper()}")
    print(f"{'='*60}")

    # ── Baseline
    baseline_path = f'results/baseline_{SCENARIO}.csv'
    if os.path.exists(baseline_path):
        print(f"\n>>> Loading BASELINE {SCENARIO}...")
        df_base = pd.read_csv(baseline_path)
    else:
        print(f"\n>>> Running BASELINE {SCENARIO}...")
        start_simulation(CONFIG, gui=False)
        rows = []
        for _ in range(720):
            step(5)
            rows.append(collect_metrics())
        stop_simulation()
        df_base = pd.DataFrame(rows)
        df_base.to_csv(baseline_path, index=False)

    base_wait = df_base['total_waiting_time'].mean()
    base_arr  = int(df_base['arrived'].sum())
    print(f"  Baseline avg wait : {base_wait:.1f}s")
    print(f"  Baseline arrived  : {base_arr}")

    # ── WOA
    print(f"\n>>> Running WOA {SCENARIO}...")
    woa_wait, woa_q, woa_arr = run_framework(
        CONFIG, algorithm='WOA',
        save_path=f'results/woa_{SCENARIO}.csv')

    # ── HHO
    print(f"\n>>> Running HHO {SCENARIO}...")
    hho_wait, hho_q, hho_arr = run_framework(
        CONFIG, algorithm='HHO',
        save_path=f'results/hho_{SCENARIO}.csv')

    # ── GA
    print(f"\n>>> Running GA {SCENARIO}...")
    ga_wait, ga_q, ga_arr = run_framework(
        CONFIG, algorithm='GA',
        save_path=f'results/ga_{SCENARIO}.csv')

    # ── Comparison
    woa_imp = (base_wait - woa_wait) / base_wait * 100
    hho_imp = (base_wait - hho_wait) / base_wait * 100
    ga_imp  = (base_wait - ga_wait)  / base_wait * 100

    print(f"\n{'='*60}")
    print(f"FINAL COMPARISON — {SCENARIO.capitalize()} traffic")
    print(f"{'='*60}")
    print(f"  {'Method':<12} {'Avg Wait':>10}   {'vs Baseline':>12}    {'Arrived':>8}")
    print(f"  {'-'*50}")
    print(f"  {'Baseline':<12} {base_wait:>9.1f}s   {'—':>12}    {base_arr:>8}")
    print(f"  {'GA':<12} {ga_wait:>9.1f}s   {ga_imp:>+11.1f}%    {int(ga_arr):>8}")
    print(f"  {'WOA':<12} {woa_wait:>9.1f}s   {woa_imp:>+11.1f}%    {int(woa_arr):>8}")
    print(f"  {'HHO':<12} {hho_wait:>9.1f}s   {hho_imp:>+11.1f}%    {int(hho_arr):>8}")
    print(f"{'='*60}")

print("\n ALL SCENARIOS DONE!")