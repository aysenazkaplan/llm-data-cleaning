
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = {
    'direct_llm': ('Direct LLM', '#4878A8'),
    'synthesized_operator': ('Synthesized', '#E18D39'),
    'guided_operator': ('Guided', '#4D9775'),
}


def plot_new_dataset_results(comparison_csv, figures_dir=None):

    comparison_csv = Path(comparison_csv)
    if not comparison_csv.is_file():
        raise FileNotFoundError(comparison_csv)
    results = pd.read_csv(comparison_csv)
    needed = {'operator_type', 'selected_columns', 'method'}
    if results.empty or not needed.issubset(results.columns):
        raise ValueError(f'Invalid comparison CSV; requires {sorted(needed)}')

    out = Path(figures_dir) if figures_dir else comparison_csv.parent / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    results = results.copy()
    results['task'] = results['operator_type'].astype(str) + ' / ' + results['selected_columns'].astype(str)
    tasks = list(dict.fromkeys(results['task']))
    active = [(key, *info) for key, info in METHODS.items() if key in set(results['method'])]
    if not active:
        raise ValueError('No recognized cleaning methods in comparison CSV')

    def metric(column, method):
        subset = results.loc[results['method'] == method].drop_duplicates('task', keep='last')
        if column not in subset:
            return np.full(len(tasks), np.nan)
        series = pd.to_numeric(subset.set_index('task')[column], errors='coerce')
        return series.reindex(tasks).to_numpy(dtype=float)

    def base(title, ylabel):
        fig, ax = plt.subplots(figsize=(max(9, len(tasks)*2.1), 5.5))
        ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(len(tasks)))
        ax.set_xticklabels(tasks, rotation=15, ha='right')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        return fig, ax

    def save(fig, filename):
        fig.tight_layout()
        path = out / filename
        fig.savefig(path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f'  [Chart] {path}', flush=True)

    x = np.arange(len(tasks))
    width = min(.24, .78 / max(1, len(active)))
    offsets = {key: (i-(len(active)-1)/2)*width for i,(key,_,_) in enumerate(active)}


    fig, ax = base('Error Correction Accuracy (Dirty Cells)', 'Error correction accuracy (%)')
    for key, label, color in active:
        dirty = metric('error_correction_accuracy', key)*100
        pos = x + offsets[key]
        good = np.isfinite(dirty)
        ax.bar(pos[good], dirty[good], width, color=color, label=label)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=9)
    save(fig, '01_error_correction_accuracy.png')


    fig, ax = base('Token Consumption', 'Input + output tokens')
    for key, label, color in active:
        ins, outs = metric('input_tokens', key), metric('output_tokens', key)
        values = np.where(np.isnan(ins) & np.isnan(outs), np.nan,
                          np.nan_to_num(ins) + np.nan_to_num(outs))
        good = np.isfinite(values)
        ax.bar((x+offsets[key])[good], values[good], width, label=label, color=color)
    ax.legend(frameon=False, fontsize=9)
    save(fig, '02_token_consumption.png')


    fig, ax = base('Execution Time (End-to-End)', 'Elapsed seconds')
    for key, label, color in active:
        values = metric('elapsed_seconds', key)
        good = np.isfinite(values)
        ax.bar((x+offsets[key])[good], values[good], width, label=label, color=color)
    ax.legend(frameon=False, fontsize=9)
    save(fig, '03_execution_time.png')


    generated = [(key, label, color) for key,label,color in active if key != 'direct_llm']
    fig, ax = base('Generated Operator Success Rates', 'Success rate (%)')
    for i, (key, label, color) in enumerate(generated):

        w = .18
        centers = x + (i-(len(generated)-1)/2)*.40
        gen = metric('attempt_success_rate', key)*100
        exe = metric('execution_success_rate', key)*100
        good_gen = np.isfinite(gen)
        good_exe = np.isfinite(exe)
        ax.bar((centers-w/2)[good_gen], gen[good_gen], w, color=color,
               label=f'{label} · generation')
        ax.bar((centers+w/2)[good_exe], exe[good_exe], w, color=color,
               alpha=.45, hatch='//', label=f'{label} · execution')
    ax.set_ylim(0, 105)
    if generated:
        ax.legend(fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(.5,-.18), frameon=False)
    else:
        ax.text(.5, .5, 'Direct LLM does not generate operators', ha='center',
                va='center', transform=ax.transAxes)
    fig.subplots_adjust(bottom=.28)
    save(fig, '04_operator_success.png')
    return out
