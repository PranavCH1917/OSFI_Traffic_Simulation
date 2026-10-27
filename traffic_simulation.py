"""
OSFI-inspired traffic signal simulation prototype
-------------------------------------------------
Run:
    python traffic_simulation.py

This educational prototype compares:
    1. Fixed-time
    2. Mamdani-style fuzzy controller
    3. Sugeno-style fuzzy controller
    4. OSFI-inspired similarity-tuned fuzzy controller

Important:
This is NOT an exact reproduction of the cited paper. The presentation does not
provide all membership-function parameters, rules, or the original truth table.
The OSFI-inspired controller here tunes a small rule table against a transparent
target table. For a research-grade reproduction, replace these definitions with
the paper's exact specifications.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

DIRECTIONS = ["North", "East", "South", "West"]
N = len(DIRECTIONS)

# Simulation settings (seconds)
SIM_TIME = 3600
WARMUP = 300
MIN_GREEN = 8
MAX_GREEN = 35
BASE_GREEN = 10
CLEARANCE = 2                 # simplified all-red clearance between phases
VEHICLES_PER_SECOND = 0.90    # discharge capacity while green
SEED = 42

# Arrival rates (vehicles/second) for moderate and heavy demand.
SCENARIOS = {
    "Moderate": np.array([0.10, 0.12, 0.09, 0.11]),
    "Heavy": np.array([0.18, 0.20, 0.16, 0.19]),
}

OUTPUT_DIR = Path("graphs")
OUTPUT_DIR.mkdir(exist_ok=True)


def clamp(value, low, high):
    return max(low, min(high, value))


def memberships(queue_count, arrival_rate):
    """Simple, transparent membership degrees for Low/Medium/High demand."""
    q = float(queue_count)
    a = float(arrival_rate)

    low_q = clamp((18 - q) / 18, 0, 1)
    high_q = clamp((q - 8) / 22, 0, 1)
    med_q = max(0.0, 1.0 - abs(q - 15) / 15)

    low_a = clamp((0.30 - a) / 0.22, 0, 1)
    high_a = clamp((a - 0.25) / 0.25, 0, 1)
    med_a = max(0.0, 1.0 - abs(a - 0.27) / 0.22)

    return {
        "queue_low": low_q, "queue_medium": med_q, "queue_high": high_q,
        "arrival_low": low_a, "arrival_medium": med_a, "arrival_high": high_a,
    }


def fuzzy_rule_outputs(queue_count, arrival_rate):
    """
    Return weighted rule strengths and output values.
    Outputs are extension seconds: zero, short, medium, long.
    """
    m = memberships(queue_count, arrival_rate)
    # Rule strengths: AND is minimum. These rules are illustrative.
    strengths = np.array([
        min(m["queue_low"], m["arrival_low"]),
        min(m["queue_low"], m["arrival_medium"]),
        min(m["queue_medium"], m["arrival_low"]),
        min(m["queue_medium"], m["arrival_medium"]),
        min(m["queue_high"], max(m["arrival_medium"], m["arrival_high"])),
        min(m["queue_medium"], m["arrival_high"]),
    ], dtype=float)
    outputs = np.array([0.0, 3.0, 3.0, 7.0, 15.0, 11.0])
    return strengths, outputs


def mamdani_extension(queue_count, arrival_rate):
    """Mamdani-style centroid approximation using sampled output membership shapes."""
    strengths, centers = fuzzy_rule_outputs(queue_count, arrival_rate)
    x = np.linspace(0, 18, 181)
    aggregated = np.zeros_like(x)
    # Triangular output membership functions, clipped by each rule strength.
    widths = [3.0, 4.0, 4.0, 5.0, 5.0, 5.0]
    for strength, center, width in zip(strengths, centers, widths):
        if strength <= 0:
            continue
        triangle = np.maximum(0.0, 1.0 - np.abs(x - center) / width)
        aggregated = np.maximum(aggregated, np.minimum(strength, triangle))
    if aggregated.sum() == 0:
        return 0.0
    return float(np.sum(x * aggregated) / np.sum(aggregated))


def sugeno_extension(queue_count, arrival_rate):
    """Sugeno-style weighted average with crisp rule consequents."""
    strengths, outputs = fuzzy_rule_outputs(queue_count, arrival_rate)
    total = strengths.sum()
    if total <= 1e-9:
        return 0.0
    return float(np.dot(strengths, outputs) / total)


def target_extension(queue_count, arrival_rate):
    """
    Transparent target table used to demonstrate similarity-based tuning.
    It is a heuristic target, NOT the paper's unavailable original truth table.
    """
    demand_score = 0.75 * queue_count + 24.0 * arrival_rate
    if demand_score < 6:
        return 0.0
    if demand_score < 14:
        return 4.0
    if demand_score < 24:
        return 9.0
    return 15.0


def build_similarity_tuned_table():
    """
    Build a compact lookup table. Each cell uses the target output for a
    representative queue/arrival pair. This demonstrates the workflow of
    truth-table construction and rule-output tuning.
    """
    queue_bins = [0, 5, 12, 22, 35, 50]
    arrival_bins = [0.10, 0.20, 0.30, 0.40, 0.50]
    table = {}
    for qi, q in enumerate(queue_bins):
        for ai, a in enumerate(arrival_bins):
            table[(qi, ai)] = target_extension(q, a)
    return queue_bins, arrival_bins, table


OSFI_Q_BINS, OSFI_A_BINS, OSFI_TABLE = build_similarity_tuned_table()


def osfi_extension(queue_count, arrival_rate):
    """OSFI-inspired inference by nearest truth-table cell."""
    qi = int(np.argmin(np.abs(np.asarray(OSFI_Q_BINS) - queue_count)))
    ai = int(np.argmin(np.abs(np.asarray(OSFI_A_BINS) - arrival_rate)))
    return float(OSFI_TABLE[(qi, ai)])


def choose_phase(queues, current_phase, controller):
    """Adaptive controllers choose the longest queue; fixed-time cycles."""
    if controller == "Fixed-time":
        return (current_phase + 1) % N
    # Tie-breaking is deterministic, so results are reproducible.
    return int(np.argmax([len(q) for q in queues]))


def green_duration(controller, phase, queues, arrival_rates):
    q_len = len(queues[phase])
    rate = arrival_rates[phase]
    if controller == "Fixed-time":
        return 15
    if controller == "Mamdani":
        ext = mamdani_extension(q_len, rate)
    elif controller == "Sugeno":
        ext = sugeno_extension(q_len, rate)
    elif controller == "OSFI-inspired":
        ext = osfi_extension(q_len, rate)
    else:
        raise ValueError(f"Unknown controller: {controller}")
    return int(round(clamp(BASE_GREEN + ext, MIN_GREEN, MAX_GREEN)))


def run_simulation(scenario_name, arrival_rates, controller, seed):
    rng = np.random.default_rng(seed)
    queues = [[] for _ in range(N)]  # each item is vehicle arrival time
    phase = N - 1
    remaining_green = 0
    clearance_left = 0

    waiting_times = []
    queue_sum = np.zeros(N, dtype=float)
    queue_samples = 0
    exits = np.zeros(N, dtype=int)
    total_arrivals = np.zeros(N, dtype=int)
    timeline = []
    queue_history = []

    for t in range(SIM_TIME):
        # Poisson arrivals, independently generated for each approach.
        for i, rate in enumerate(arrival_rates):
            count = int(rng.poisson(rate))
            total_arrivals[i] += count
            for _ in range(count):
                queues[i].append(t)

        # Start a new phase when current green and clearance are finished.
        if remaining_green <= 0 and clearance_left <= 0:
            phase = choose_phase(queues, phase, controller)
            remaining_green = green_duration(controller, phase, queues, arrival_rates)

        # Vehicles leave only from the active green approach.
        if clearance_left <= 0 and remaining_green > 0:
            # Stochastic discharge with a maximum average service capacity.
            capacity = int(rng.poisson(VEHICLES_PER_SECOND))
            for _ in range(min(capacity, len(queues[phase]))):
                arrival_time = queues[phase].pop(0)
                if t >= WARMUP:
                    waiting_times.append(max(0, t - arrival_time))
                if t >= WARMUP:
                    exits[phase] += 1
            remaining_green -= 1

            if remaining_green <= 0:
                clearance_left = CLEARANCE
        else:
            clearance_left -= 1

        if t >= WARMUP:
            queue_sum += np.array([len(q) for q in queues], dtype=float)
            queue_samples += 1
            timeline.append(t)
            queue_history.append([len(q) for q in queues])

    completed = int(exits.sum())
    mean_delay = float(np.mean(waiting_times)) if waiting_times else 0.0
    mean_queue = float(queue_sum.sum() / max(queue_samples * N, 1))
    unfinished = sum(len(q) for q in queues)

    return {
        "scenario": scenario_name,
        "controller": controller,
        "average_delay_s": mean_delay,
        "throughput_vehicles": completed,
        "average_queue_vehicles_per_approach": mean_queue,
        "arrivals": int(total_arrivals.sum()),
        "unfinished_vehicles": int(unfinished),
        "exits_by_direction": exits.tolist(),
        "timeline": np.asarray(timeline),
        "queue_history": np.asarray(queue_history),
    }


def save_results(results):
    csv_path = Path("results.csv")
    fields = [
        "scenario", "controller", "average_delay_s",
        "throughput_vehicles", "average_queue_vehicles_per_approach",
        "arrivals", "unfinished_vehicles",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fields})
    print(f"\nSaved summary to: {csv_path.resolve()}")


def create_graphs(results):
    controllers = ["Fixed-time", "Mamdani", "Sugeno", "OSFI-inspired"]
    scenarios = list(SCENARIOS.keys())

    for metric, ylabel, filename in [
        ("average_delay_s", "Average waiting delay (seconds)", "average_delay.png"),
        ("throughput_vehicles", "Vehicles discharged after warm-up", "throughput.png"),
        ("average_queue_vehicles_per_approach", "Average queue per approach (vehicles)", "queue_length.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(controllers))
        width = 0.36
        for j, scenario in enumerate(scenarios):
            values = [
                next(r[metric] for r in results
                     if r["scenario"] == scenario and r["controller"] == c)
                for c in controllers
            ]
            ax.bar(x + (j - 0.5) * width, values, width, label=scenario)
        ax.set_xticks(x)
        ax.set_xticklabels(controllers, rotation=12)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by controller")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = OUTPUT_DIR / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Saved graph: {path.resolve()}")

    # Example queue time series for OSFI-inspired controller under heavy traffic.
    example = next(r for r in results
                   if r["scenario"] == "Heavy" and r["controller"] == "OSFI-inspired")
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, direction in enumerate(DIRECTIONS):
        ax.plot(example["timeline"], example["queue_history"][:, i], label=direction)
    ax.set_xlabel("Simulation time (seconds)")
    ax.set_ylabel("Queue length (vehicles)")
    ax.set_title("Queue lengths: OSFI-inspired controller, heavy traffic")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = OUTPUT_DIR / "queue_over_time.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved graph: {path.resolve()}")


def main():
    controllers = ["Fixed-time", "Mamdani", "Sugeno", "OSFI-inspired"]
    all_results = []

    print("OSFI-inspired Four-Way Traffic Signal Simulation")
    print("=" * 58)
    print(f"Simulation duration: {SIM_TIME} s; warm-up excluded from delay/queue metrics: {WARMUP} s")
    print("This is a prototype; do not present its output as the paper's reported results.\n")

    for scenario_index, (scenario, rates) in enumerate(SCENARIOS.items()):
        print(f"\nScenario: {scenario} | arrival rates (vehicles/s): {rates.tolist()}")
        for controller_index, controller in enumerate(controllers):
            # Same seed per scenario for repeatable, comparable arrival streams.
            result = run_simulation(
                scenario, rates, controller,
                seed=SEED + scenario_index * 100 + controller_index
            )
            all_results.append(result)
            print(
                f"{controller:15s} | delay={result['average_delay_s']:7.2f} s"
                f" | throughput={result['throughput_vehicles']:5d}"
                f" | avg queue={result['average_queue_vehicles_per_approach']:6.2f}"
                f" | unfinished={result['unfinished_vehicles']:5d}"
            )

    save_results(all_results)
    create_graphs(all_results)
    print("\nDone. Open results.csv and the PNG files in the graphs folder.")
    print("For a stronger experiment, repeat each run with 10-30 seeds and report mean ± standard deviation.")


if __name__ == "__main__":
    main()
