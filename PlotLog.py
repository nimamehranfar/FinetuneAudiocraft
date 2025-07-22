import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import pandas as pd

log_file = f'S:/Xsmall_0.1/solver.log.0'

# Data containers
epochs = []
train_ce = []
train_ppl = []
train_dur = []
val_ce = []
val_ppl = []
val_dur = []

# Patterns
train_pat = re.compile(r'Train Summary \| Epoch (\d+) .*? ce=([\d.]+) \| ppl=([\d.]+) \| duration=([\d.]+)')
val_pat = re.compile(r'Valid Summary \| Epoch (\d+) \| ce=([\d.]+) \| ppl=([\d.]+) \| duration=([\d.]+)')

with open(log_file, encoding='utf-8') as f:
    for line in f:
        train_match = train_pat.search(line)
        if train_match:
            epochs.append(int(train_match.group(1)))
            train_ce.append(float(train_match.group(2)))
            train_ppl.append(float(train_match.group(3)))
            train_dur.append(float(train_match.group(4)))
            continue  # Move to next line

        val_match = val_pat.search(line)
        if val_match:
            # It's possible validation epoch matches the last appended one
            val_ce.append(float(val_match.group(2)))
            val_ppl.append(float(val_match.group(3)))
            val_dur.append(float(val_match.group(4)))

# --- Plotting Loss (Cross-Entropy) ---
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_ce, label='Train CE Loss', marker='o')
plt.plot(epochs, val_ce, label='Validation CE Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.title('Training and Validation CE Loss per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('ce_loss_curve.png')
plt.close()

# --- Plotting Perplexity ---
min_ppl = min(min(train_ppl), min(val_ppl))
max_ppl = max(max(train_ppl), max(val_ppl))

# Add a little margin so the data doesn't sit on the edge
ymin = min_ppl * 0.95
ymax = max_ppl * 1.05

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_ppl, label='Train Perplexity', marker='o')
plt.plot(epochs, val_ppl, label='Validation Perplexity', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Perplexity (log scale)')
plt.title('Training and Validation Perplexity per Epoch')
plt.legend()
plt.yscale('log')

ax = plt.gca()
ax.set_ylim(ymin, ymax)

# Use LogLocator for both major and minor ticks
major_ticks = np.unique(
    np.concatenate((
        np.logspace(np.floor(np.log10(ymin)), np.ceil(np.log10(ymax)), num=10),
        [min_ppl, max_ppl]
    ))
)
ax.yaxis.set_major_locator(ticker.FixedLocator(major_ticks))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*.1))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

plt.grid(True, which='both', ls='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('ppl_curve_log.png')
plt.close()

# --- Plotting Epoch Duration ---
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_dur, label='Train Duration (s)', marker='o')
plt.plot(epochs, val_dur, label='Valid Duration (s)', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Duration (seconds)')
plt.title('Training and Validation Epoch Duration')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('duration_curve.png')
plt.close()

print('Plots saved as ce_loss_curve.png, ppl_curve.png, and duration_curve.png')


# Reuse variables from your script:
# epochs, train_ce, val_ce, train_ppl, val_ppl

df = pd.DataFrame({
    'Epoch': epochs,
    'Train CE Loss': train_ce,
    'Validation CE Loss': val_ce,
    'Train Perplexity': train_ppl,
    'Validation Perplexity': val_ppl,
})

# Save as CSV for inspection
df.to_csv('training_results.csv', index=False)

# Save as LaTeX table for direct copy-paste in your report
with open('training_results.tex', 'w') as f:
    f.write(df.to_latex(index=False, float_format="%.3f"))