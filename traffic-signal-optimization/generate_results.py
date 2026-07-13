import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('results/figures', exist_ok=True)

# ── Colors
COLORS = {
    'Baseline': '#95a5a6',
    'GA':       '#e67e22',
    'WOA':      '#2980b9',
    'HHO':      '#27ae60',
}

SCENARIOS = ['low', 'medium', 'peak']
METHODS   = ['GA', 'WOA', 'HHO']

# ── Load single-run results
data = {}
for s in SCENARIOS:
    data[s] = {
        'baseline': pd.read_csv(f'results/baseline_{s}.csv'),
        'ga':       pd.read_csv(f'results/ga_{s}.csv'),
        'woa':      pd.read_csv(f'results/woa_{s}.csv'),
        'hho':      pd.read_csv(f'results/hho_{s}.csv'),
    }

# ── Load 5x summary
df5 = pd.read_csv('results/runs/summary_5x.csv')

# ── Baseline stats
baseline_stats = {
    'low':    {'wait': 446.9,  'stops': data['low']['baseline']['total_stops'].mean(),
               'trip': data['low']['baseline']['total_trip_time'].mean(),
               'arr':  data['low']['baseline']['arrived'].sum()},
    'medium': {'wait': 1161.4, 'stops': data['medium']['baseline']['total_stops'].mean(),
               'trip': data['medium']['baseline']['total_trip_time'].mean(),
               'arr':  data['medium']['baseline']['arrived'].sum()},
    'peak':   {'wait': 1719.0, 'stops': data['peak']['baseline']['total_stops'].mean(),
               'trip': data['peak']['baseline']['total_trip_time'].mean(),
               'arr':  data['peak']['baseline']['arrived'].sum()},
}


# ══════════════════════════════════════════════════════════
# EXCEL SUMMARY TABLE WITH MEAN +- STD
# ══════════════════════════════════════════════════════════
rows = []
for s in SCENARIOS:
    base = baseline_stats[s]
    rows.append({
        'Scenario':    s.capitalize(),
        'Method':      'Baseline',
        'Avg_Wait_s':  round(base['wait'], 1),
        'Std_Wait':    '-',
        'Improvement': '-',
        'Avg_Stops':   round(base['stops'], 1),
        'Std_Stops':   '-',
        'Avg_Trip_s':  round(base['trip'], 1),
        'Std_Trip':    '-',
        'Arrived':     int(base['arr']),
    })
    for m in METHODS:
        row5 = df5[(df5['method'] == m) & (df5['scenario'] == s)].iloc[0]
        imp  = (base['wait'] - row5['wait_mean']) / base['wait'] * 100
        rows.append({
            'Scenario':    s.capitalize(),
            'Method':      m,
            'Avg_Wait_s':  round(row5['wait_mean'], 1),
            'Std_Wait':    f"+- {row5['wait_std']:.1f}",
            'Improvement': f"{imp:+.1f}%",
            'Avg_Stops':   round(row5['stops_mean'], 1),
            'Std_Stops':   f"+- {row5['stops_std']:.2f}",
            'Avg_Trip_s':  round(row5['trip_mean'], 1),
            'Std_Trip':    f"+- {row5['trip_std']:.1f}",
            'Arrived':     round(row5['arr_mean'], 1),
        })

df_table = pd.DataFrame(rows)
df_table.to_excel('results/figures/summary_table.xlsx', index=False)
print("Summary table saved")

# Print to console
print("\n" + "="*95)
print(f"{'Scenario':<10} {'Method':<10} {'Avg Wait':>10} {'Std':>8} {'Improve':>10} "
      f"{'Stops':>7} {'Trip(s)':>10} {'Arrived':>8}")
print("="*95)
for _, r in df_table.iterrows():
    print(f"{r['Scenario']:<10} {r['Method']:<10} {r['Avg_Wait_s']:>10} "
          f"{str(r['Std_Wait']):>8} {str(r['Improvement']):>10} "
          f"{r['Avg_Stops']:>7} {r['Avg_Trip_s']:>10} {r['Arrived']:>8}")
print("="*95)


# ══════════════════════════════════════════════════════════
# FIG 1 — Bar chart: Avg waiting time with error bars
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for ax, s in zip(axes, SCENARIOS):
    base = baseline_stats[s]['wait']
    methods_all = ['Baseline', 'GA', 'WOA', 'HHO']
    values = [base]
    errors = [0]
    for m in METHODS:
        row5 = df5[(df5['method'] == m) & (df5['scenario'] == s)].iloc[0]
        values.append(row5['wait_mean'])
        errors.append(row5['wait_std'])

    bars = ax.bar(methods_all, values,
                  color=[COLORS[m] for m in methods_all],
                  edgecolor='white', linewidth=1.5, width=0.55,
                  yerr=errors, capsize=5,
                  error_kw={'linewidth': 1.5, 'color': 'black'})

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(errors) + max(values)*0.02,
                f'{val:.0f}s', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax.set_title(f'{s.capitalize()} traffic', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average waiting time (s)', fontsize=11)
    ax.set_ylim(0, max(values) * 1.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Average Waiting Time Comparison (mean +- std, 5 runs)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/fig1_waiting_time_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved")


# ══════════════════════════════════════════════════════════
# FIG 2 — Temporal evolution: waiting time over time
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, s in zip(axes, SCENARIOS):
    for method, color, label in [
        ('baseline', COLORS['Baseline'], 'Baseline (Fixed)'),
        ('ga',       COLORS['GA'],       'GA (Reactive)'),
        ('woa',      COLORS['WOA'],      'Predictive WOA'),
        ('hho',      COLORS['HHO'],      'Predictive HHO'),
    ]:
        df       = data[s][method]
        smoothed = df['total_waiting_time'].rolling(10, min_periods=1).mean()
        ax.plot(df['time'], smoothed,
                color=color, label=label,
                linewidth=2, alpha=0.9)

    ax.set_title(f'{s.capitalize()} traffic', fontsize=12, fontweight='bold')
    ax.set_xlabel('Simulation time (s)', fontsize=10)
    ax.set_ylabel('Total waiting time (s)', fontsize=10)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)

plt.suptitle('Temporal Evolution of Waiting Time',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/fig2_temporal_evolution.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved")


# ══════════════════════════════════════════════════════════
# FIG 3 — Grouped bar: 4 metrics for ALL scenarios
# ══════════════════════════════════════════════════════════
metrics_cfg = [
    ('wait_mean',  'wait_std',  'Avg Waiting Time (s)'),
    ('stops_mean', 'stops_std', 'Avg Number of Stops'),
    ('trip_mean',  'trip_std',  'Avg Trip Time (s)'),
    ('arr_mean',   'arr_std',   'Total Arrived'),
]

for s in SCENARIOS:
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    base_vals = {
        'wait_mean':  baseline_stats[s]['wait'],
        'stops_mean': baseline_stats[s]['stops'],
        'trip_mean':  baseline_stats[s]['trip'],
        'arr_mean':   baseline_stats[s]['arr'],
        'wait_std':   0,
        'stops_std':  0,
        'trip_std':   0,
        'arr_std':    0,
    }

    for ax, (mean_col, std_col, ylabel) in zip(axes, metrics_cfg):
        methods_all = ['Baseline', 'GA', 'WOA', 'HHO']
        values = [base_vals[mean_col]]
        errors = [0]
        for m in METHODS:
            row5 = df5[(df5['method'] == m) & (df5['scenario'] == s)].iloc[0]
            values.append(row5[mean_col])
            errors.append(row5[std_col])

        bars = ax.bar(methods_all, values,
                      color=[COLORS[m] for m in methods_all],
                      edgecolor='white', linewidth=1.5,
                      yerr=errors, capsize=5,
                      error_kw={'linewidth': 1.5})

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(values)*0.02,
                    f'{val:.1f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=9, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle(f'4 Performance Metrics — {s.capitalize()} Traffic (mean +- std, 5 runs)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'results/figures/fig3_grouped_metrics_{s}.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure 3 saved → fig3_grouped_metrics_{s}.png")


# ══════════════════════════════════════════════════════════
# FIG 4 — Improvement percentage bar chart
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
x     = np.arange(len(SCENARIOS))
width = 0.25

for i, m in enumerate(METHODS):
    values = []
    errors = []
    for s in SCENARIOS:
        base = baseline_stats[s]['wait']
        row5 = df5[(df5['method'] == m) & (df5['scenario'] == s)].iloc[0]
        imp  = (base - row5['wait_mean']) / base * 100
        values.append(imp)
        errors.append(row5['wait_std'] / base * 100)

    bars = ax.bar(x + i*width, values, width,
                  label=m, color=COLORS[m],
                  edgecolor='white', linewidth=1.5,
                  yerr=errors, capsize=5)

    for bar, val in zip(bars, values):
        sign  = '+' if val >= 0 else ''
        y_pos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 2.5
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{sign}{val:.1f}%',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')

ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
ax.set_xticks(x + width)
ax.set_xticklabels([s.capitalize() for s in SCENARIOS], fontsize=12)
ax.set_ylabel('Improvement over baseline (%)', fontsize=12)
ax.set_title('Waiting Time Reduction vs Fixed-Time Baseline',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/fig4_improvement_percentage.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved")


# ══════════════════════════════════════════════════════════
# FIG 5 — Heatmap: waiting time per method per scenario
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

methods_all = ['Baseline', 'GA', 'WOA', 'HHO']
heatmap_data = []

for m in methods_all:
    row_vals = []
    for s in SCENARIOS:
        if m == 'Baseline':
            row_vals.append(baseline_stats[s]['wait'])
        else:
            row5 = df5[(df5['method'] == m) & (df5['scenario'] == s)].iloc[0]
            row_vals.append(row5['wait_mean'])
    heatmap_data.append(row_vals)

heatmap_data = np.array(heatmap_data)

im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r')
ax.set_xticks(range(len(SCENARIOS)))
ax.set_xticklabels([s.capitalize() for s in SCENARIOS], fontsize=13)
ax.set_yticks(range(len(methods_all)))
ax.set_yticklabels(methods_all, fontsize=13)
ax.set_xlabel('Traffic Scenario', fontsize=12)
ax.set_title('Average Waiting Time Heatmap\n(Red=high, Green=low)',
             fontsize=13, fontweight='bold')

for i in range(len(methods_all)):
    for j in range(len(SCENARIOS)):
        val = heatmap_data[i, j]
        ax.text(j, i, f'{val:.0f}s',
                ha='center', va='center',
                fontsize=12, fontweight='bold',
                color='white' if val > heatmap_data.max()*0.6 else 'black')

plt.colorbar(im, ax=ax, label='Avg waiting time (s)')
plt.tight_layout()
plt.savefig('results/figures/fig5_heatmap.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved")


print("\n" + "="*50)
print("ALL RESULTS GENERATED!")
print("="*50)
print("Files in results/figures/:")
print("  summary_table.xlsx")
print("  fig1_waiting_time_comparison.png")
print("  fig2_temporal_evolution.png")
print("  fig3_grouped_metrics.png")
print("  fig4_improvement_percentage.png")
print("  fig5_heatmap.png")