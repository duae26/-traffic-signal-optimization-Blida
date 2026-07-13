import numpy as np
import traci
from src.utils.tls_ids import TLS_IDS
from src.utils.traci_interface import MAIN_LABEL

def evaluate_signal_plan(signal_plan, config_file=None, port=None):
    signal_plan = np.clip(signal_plan, 15, 50)
    try:
        traci.switch(MAIN_LABEL)
        total_fitness = 0.0
        for i, tls_id in enumerate(TLS_IDS):
            green  = float(signal_plan[i])
            yellow = 4.0
            cycle  = 2 * green + 2 * yellow
            red    = cycle - green - yellow
            try:
                lanes   = list(set(
                    traci.trafficlight.getControlledLanes(tls_id)))
                queue   = sum(traci.lane.getLastStepHaltingNumber(l)
                              for l in lanes)
                wait    = sum(traci.lane.getWaitingTime(l)
                              for l in lanes)
                n_lanes = max(len(lanes), 1)
                cost    = (queue * red) + (wait * red / cycle)
            except Exception:
                cost = red * 10.0
            total_fitness += cost
        return float(total_fitness)
    except Exception:
        return 999999.0