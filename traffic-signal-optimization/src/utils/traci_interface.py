import traci
import numpy as np
from src.utils.tls_ids import TLS_IDS, NET_FILE

MIN_GREEN  = 15
MAX_GREEN  = 50
YELLOW     = 4
NUM_TLS    = len(TLS_IDS)
MAIN_LABEL = 'main_sim'


def start_simulation(config_file, gui=False, port=8813):
    binary = 'sumo-gui' if gui else 'sumo'
    traci.start([
        binary,
        '-c', config_file,
        '--no-warnings',
        '--no-step-log',
    ], port=port, label=MAIN_LABEL)
    traci.switch(MAIN_LABEL)
    print(f"[TraCI] Simulation started — {config_file}")


def stop_simulation():
    try:
        traci.switch(MAIN_LABEL)
        traci.close()
    except Exception:
        pass
    print("[TraCI] Simulation stopped.")


def get_waiting_time(tls_id):
    total = 0.0
    for lane in traci.trafficlight.getControlledLanes(tls_id):
        total += traci.lane.getWaitingTime(lane)
    return total


def get_queue_length(tls_id):
    total = 0
    for lane in traci.trafficlight.getControlledLanes(tls_id):
        total += traci.lane.getLastStepHaltingNumber(lane)
    return total


def get_vehicle_count(tls_id):
    total = 0
    for lane in traci.trafficlight.getControlledLanes(tls_id):
        total += traci.lane.getLastStepVehicleNumber(lane)
    return total


def get_network_state():
    traci.switch(MAIN_LABEL)
    state = {}
    for tls_id in TLS_IDS:
        try:
            state[tls_id] = {
                'waiting_time': get_waiting_time(tls_id),
                'queue_length': get_queue_length(tls_id),
                'vehicle_count': get_vehicle_count(tls_id),
            }
        except Exception:
            state[tls_id] = {
                'waiting_time': 0.0,
                'queue_length': 0,
                'vehicle_count': 0,
            }
    return state


def get_state_vector():
    state = get_network_state()
    vector = []
    for tls_id in TLS_IDS:
        s = state[tls_id]
        vector.extend([
            s['waiting_time'],
            s['queue_length'],
            s['vehicle_count'],
        ])
    return np.array(vector, dtype=np.float32)


def apply_signal_plan(signal_plan):
    traci.switch(MAIN_LABEL)
    for i, tls_id in enumerate(TLS_IDS):
        green_duration = int(np.clip(signal_plan[i], MIN_GREEN, MAX_GREEN))
        try:
            logic  = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            phases = list(logic.phases)
            if len(phases) >= 4:
                phases[0].duration = green_duration
                phases[2].duration = green_duration
            new_logic = traci.trafficlight.Logic(
                logic.programID,
                logic.type,
                logic.currentPhaseIndex,
                phases
            )
            traci.trafficlight.setProgramLogic(tls_id, new_logic)
        except Exception:
            pass


def step(n=1):
    traci.switch(MAIN_LABEL)
    for _ in range(n):
        traci.simulationStep()


def get_simulation_time():
    traci.switch(MAIN_LABEL)
    return traci.simulation.getTime()


def get_arrived_vehicles():
    traci.switch(MAIN_LABEL)
    return traci.simulation.getArrivedNumber()


def get_departed_vehicles():
    traci.switch(MAIN_LABEL)
    return traci.simulation.getDepartedNumber()


def collect_metrics():
    traci.switch(MAIN_LABEL)
    state          = get_network_state()
    total_waiting  = sum(s['waiting_time'] for s in state.values())
    total_queue    = sum(s['queue_length']  for s in state.values())
    total_vehicles = sum(s['vehicle_count'] for s in state.values())

    # ── Number of stops: vehicles currently at speed < 0.1 m/s
    total_stops = 0
    try:
        for vid in traci.vehicle.getIDList():
            if traci.vehicle.getSpeed(vid) < 0.1:
                total_stops += 1
    except Exception:
        total_stops = 0

    # ── Total trip time: sum of time each active vehicle spent in network
    total_trip_time = 0.0
    try:
        current_time = traci.simulation.getTime()
        for vid in traci.vehicle.getIDList():
            depart_time      = traci.vehicle.getDeparture(vid)
            total_trip_time += (current_time - depart_time)
    except Exception:
        total_trip_time = 0.0

    return {
        'total_waiting_time': total_waiting,
        'total_queue':        total_queue,
        'total_vehicles':     total_vehicles,
        'total_stops':        total_stops,
        'total_trip_time':    total_trip_time,
        'arrived':            get_arrived_vehicles(),
        'departed':           get_departed_vehicles(),
        'time':               get_simulation_time(),
    }


def save_simulation_state(state_file='temp/sim_state.xml'):
    import os
    os.makedirs('temp', exist_ok=True)
    traci.switch(MAIN_LABEL)
    traci.simulation.saveState(state_file)
    return state_file