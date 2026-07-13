import sys
import os
sys.path.insert(0, os.getcwd())
from src.woa.woa import WOA
from src.hho.hho import HHO
import numpy as np
import traci

from src.utils.tls_ids import TLS_IDS, NUM_INTERSECTIONS
from src.utils.traci_interface import (start_simulation, stop_simulation,
                                        collect_metrics, step,
                                        get_network_state, MAIN_LABEL)
from src.lstm.predictor import TrafficPredictor
from src.utils.fitness import evaluate_signal_plan

SIM_DURATION  = 3600
HORIZON_STEPS = 12
STEP_SIZE     = 5
POP_SIZE      = 6
MAX_ITER      = 10
MIN_GREEN     = 15
MAX_GREEN     = 50
N_VARS        = NUM_INTERSECTIONS

def apply_signal_plan(signal_plan):
    signal_plan = np.clip(signal_plan, MIN_GREEN, MAX_GREEN)
    traci.switch(MAIN_LABEL)
    for i, tls_id in enumerate(TLS_IDS):
        green = int(signal_plan[i])
        try:
            phases = []
            logic  = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            for ph in logic.phases:
                if 'y' in ph.state.lower():
                    phases.append(traci.trafficlight.Phase(4, ph.state))
                else:
                    phases.append(traci.trafficlight.Phase(green, ph.state))
            traci.trafficlight.setProgramLogic(tls_id,
                traci.trafficlight.Logic(logic.programID, logic.type,
                                         logic.currentPhaseIndex, phases))
        except Exception:
            pass

def run_framework(config_file, algorithm='WOA', save_path=None):
    from src.woa.woa import WOA
    from src.hho.hho import HHO

    print("\n" + "=" * 60)
    print(f"PREDICTIVE FRAMEWORK — {algorithm}")
    print(f"Config     : {config_file}")
    print(f"Pop size   : {POP_SIZE} | Max iter: {MAX_ITER}")
    print(f"Horizon    : {HORIZON_STEPS * STEP_SIZE}s | Step: {STEP_SIZE}s")
    print("=" * 60)

    predictor     = TrafficPredictor()
    start_simulation(config_file, gui=False)

    rows          = []
    total_steps   = SIM_DURATION // STEP_SIZE
    window_number = 0

    for t_idx in range(total_steps):
        step(STEP_SIZE)
        metrics = collect_metrics()
        rows.append(metrics)
        predictor.update_buffer(get_network_state())

        if (t_idx + 1) % HORIZON_STEPS == 0:
            window_number += 1
            sim_time   = (t_idx + 1) * STEP_SIZE
            avg_wait   = metrics.get('total_waiting_time', 0)
            lstm_ready = predictor.is_ready()

            print(f"\n[Window {window_number:3d}] t={sim_time:4d}s | "
                  f"avg_wait={avg_wait:.1f}s | lstm_ready={lstm_ready}")

            if not lstm_ready:
                print(f"  Warming up LSTM ({len(predictor.buffer)}/{predictor.seq_len})")
                continue

            pred = predictor.predict()
            print(f"  LSTM prediction : {pred:.1f}s ahead")

            os.makedirs('temp', exist_ok=True)
            try:
                traci.switch(MAIN_LABEL)
                traci.simulation.saveState('temp/sim_state.xml')
                print(f"  State saved     : temp/sim_state.xml")
            except Exception:
                pass

            def fitness_fn(plan):
                return evaluate_signal_plan(plan)

            lb = np.full(N_VARS, MIN_GREEN, dtype=float)
            ub = np.full(N_VARS, MAX_GREEN, dtype=float)

            if algorithm == 'WOA':
             opt = WOA(fitness_fn, N_VARS, lb, ub, POP_SIZE, MAX_ITER)
            elif algorithm == 'HHO':
             opt = HHO(fitness_fn, N_VARS, lb, ub, POP_SIZE, MAX_ITER)
            else:
             from src.ga.ga import GA
             opt = GA(fitness_fn, N_VARS, lb, ub, POP_SIZE, MAX_ITER)

            best_plan, best_fit = opt.run()

            initial_fit = fitness_fn(np.full(N_VARS, 30.0))
            improvement = (initial_fit - best_fit) / max(initial_fit, 1e-6) * 100
            print(f"  Best fitness    : {best_fit:.1f} (improved {improvement:.1f}%)")
            print(f"  Avg green       : {np.mean(best_plan):.1f}s "
                  f"[{np.min(best_plan):.0f}-{np.max(best_plan):.0f}]s")

            apply_signal_plan(best_plan)

    stop_simulation()

    import pandas as pd
    df = pd.DataFrame(rows)
    if save_path:
        os.makedirs('results', exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"\n[Controller] Saved to {save_path}")

    avg_wait = df['total_waiting_time'].mean()
    max_q    = df['total_queue'].max()
    arrived  = df['arrived'].sum()

    print("\n" + "=" * 60)
    print(f"RESULTS — {algorithm}")
    print("=" * 60)
    print(f"  Avg waiting time : {avg_wait:.1f}s")
    print(f"  Max queue        : {max_q:.0f} vehicles")
    print(f"  Total arrived    : {arrived:.0f} vehicles")
    print("=" * 60)

    return avg_wait, max_q, arrived