import numpy as np
import torch
from pathlib import Path

import re
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('Agg')  # Use TkAgg for live display; change to 'Agg' if headless
import matplotlib.gridspec as gridspec
import os
from collections import deque
from datetime import datetime
import sys
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import zip_longest


plt.style.use('default')


class TrainingVisualizer:
    """
    Headless (disk-only) visualizer for SAC prosthetic controller training.
    Tracks: actor/Q1/Q2/alpha losses, episode rewards, and Q-value trends.
    Plots are only rendered when save() or close() is called.

    Usage (inside rl_train):
        viz = TrainingVisualizer(save_dir='./plots')

        # per step:
        viz.log_step(reward)
        viz.log_losses(training_losses)   # reads losses + q1_mean + q2_mean

        # per episode end:
        viz.log_episode()

        # optional periodic snapshot every N episodes:
        viz.save(tag=f'ep_{episode}')

        # final save + cleanup:
        viz.close()
    """

    def __init__(self, save_dir: str = './plots', window: int = 200,num_workers: int = 1):
        self.save_dir = save_dir
        self.window   = window
        os.makedirs(save_dir, exist_ok=True)
        self.num_workers = num_workers
        # --- raw logs ---
        self.step_rewards:    list[float] = []
        self.q1_vals:         list[float] = []
        self.q2_vals:         list[float] = []
        self.actor_losses:    list[float] = []
        self.q1_losses:       list[float] = []
        self.q2_losses:       list[float] = []
        self.alpha_losses:    list[float] = []
        self.alpha_vals:      list[float] = []
        self.policy_entropy:  list[float] = []
        self._episode_reward_acc: dict[int, float] = {i: 0.0 for i in range(num_workers)}
        self.episode_rewards:dict [int,list[float]] = {i:[] for i in range(self.num_workers)}
        self._last_saved: dict[int, str] = {}  # tracks last saved plot path per worker

    # ------------------------------------------------------------------
    # Logging API
    # ------------------------------------------------------------------

    def log_step(self, reward: float, wid: int):
        """Call once per environment step."""
        self._episode_reward_acc[wid] += reward
        self.step_rewards.append(reward)

    def log_losses(self, training_losses: dict):
        """
        Call after each train_sac() invocation.
        Reads the latest entry ([-1]) from each key — safe because
        train_sac is called with training_epochs=1 per env step,
        so exactly one new value is appended per call.
        Skips silently if a list is empty (replay buffer not yet full).

        Expected keys: actor_loss, q1_loss, q2_loss, alpha_loss,
                       q1_mean, q2_mean
        """
        def _latest(key):
            lst = training_losses.get(key, [])
            return lst[-1] if lst else None

        actor = _latest('actor_loss')
        q1l   = _latest('q1_loss')
        q2l   = _latest('q2_loss')
        alpha = _latest('alpha_loss')
        alpha_v = _latest('alpha_val')
        q1m   = _latest('q1_mean')
        q2m   = _latest('q2_mean')
        policy_entropy = _latest('policy_entropy')

        if actor is not None: self.actor_losses.append(actor)
        if q1l   is not None: self.q1_losses.append(q1l)
        if q2l   is not None: self.q2_losses.append(q2l)
        if alpha is not None: self.alpha_losses.append(alpha)
        if alpha_v is not None: self.alpha_vals.append(alpha_v)
        if q1m   is not None: self.q1_vals.append(q1m)
        if q2m   is not None: self.q2_vals.append(q2m)
        if policy_entropy is not None : self.policy_entropy.append(policy_entropy)

    def log_episode(self,wid: int):
        """Call at the end of each episode to flush accumulated reward."""
        self.episode_rewards[wid].append(self._episode_reward_acc[wid])
        self._episode_reward_acc[wid] = 0.0


    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _smooth(data: list, w: int) -> np.ndarray:
        if len(data) < 2:
            return np.array(data, dtype=float)
        arr    = np.array(data, dtype=float)
        padded = np.pad(arr,(w//2,w-1-w//2),mode='edge')
        kernel = np.ones(w) /w
        return np.convolve(padded, kernel, mode='valid')

    def _plot_line(self, ax, data: list, color: str, label: str,
                smooth_w: int = 50, alpha_raw: float = 0.25, clear: bool = True):
        if clear:
            ax.cla()
        if not data:
            return
        raw    = np.array(data, dtype=float)
        xs_raw = np.arange(len(raw))
        ax.plot(xs_raw, raw, color=color, alpha=alpha_raw, linewidth=0.8)
        if len(raw) >= 2:
            w     = min(smooth_w, max(1, len(raw) // 5))
            sm    = self._smooth(data, w)
            xs_sm = np.arange(len(raw))
            ax.plot(xs_sm, sm, color=color, linewidth=1.8, label=label)       



    def _restyle_axes(self, axes, titles):
        for ax, title in zip(axes, titles):
            ax.set_title(title, color='#a0aec0', fontsize=9, pad=6)
            ax.tick_params(colors='#606878', labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('#2d333b')
            ax.grid(True, color='#21262d', linewidth=0.6, linestyle='--')
            ax.set_facecolor('#161b22')

    def _build_figure(self, wid: int):
        """Build and populate the figure fresh each time save() is called."""
        fig = plt.figure(figsize=(16, 10), facecolor='#0e1117')
        fig.suptitle('SAC Prosthetic Controller — Training Dashboard',
                     color='#e0e0e0', fontsize=14, fontweight='bold', y=0.98)

        gs = gridspec.GridSpec(4, 2, figure=fig,
                               hspace=0.45, wspace=0.35,
                               left=0.07, right=0.97,
                               top=0.93, bottom=0.07)

        ax_style = dict(facecolor='#161b22', frameon=True)

        ax_reward = fig.add_subplot(gs[0, :], **ax_style)
        ax_actor  = fig.add_subplot(gs[1, 0], **ax_style)
        ax_q_loss = fig.add_subplot(gs[1, 1], **ax_style)
        ax_alpha  = fig.add_subplot(gs[2, 0], **ax_style)
        ax_qval   = fig.add_subplot(gs[2, 1], **ax_style)
        ax_entropy   = fig.add_subplot(gs[3, 0], **ax_style)
        axes   = [ax_reward, ax_actor, ax_q_loss, ax_alpha,ax_entropy,ax_qval]
        titles = [
            'Episode Reward',
            'Actor Loss',
            'Critic Loss  (Q1 & Q2)',
            'Alpha (Entropy Coefficient) Value',
            'Policy Entropy',
            'Q-Value Trend  (batch mean, Q1 & Q2)',
        ]

        # ---- episode reward ----
        if self.episode_rewards[wid]:
            ep = np.array(self.episode_rewards[wid], dtype=float)
            xs = np.arange(len(ep))
            ax_reward.fill_between(xs, ep.min(), ep, color='#38bdf8', alpha=0.12)
            ax_reward.plot(xs, ep, color='#38bdf8', alpha=0.35, linewidth=0.9)
            if len(ep) >= 2:
                sm    = self._smooth(self.episode_rewards[wid], max(1, len(ep) // 10))
                xs_sm = np.arange(len(ep) - len(sm), len(ep))
                ax_reward.plot(xs_sm, sm, color='#38bdf8', linewidth=2.2,
                               label='Smoothed reward')
            ax_reward.set_xlabel('Episode', color='#606878', fontsize=7)

        # ---- actor loss ----
        self._plot_line(ax_actor, self.actor_losses,
                        '#f97316', 'Actor loss', self.window)
        self._plot_line(ax_entropy, self.policy_entropy,
                        '#f97316', 'Actor Entropy', self.window)

        # ---- critic losses ----
        if self.q1_losses:
            self._plot_line(ax_q_loss, self.q1_losses, '#34d399', 'Q1 loss', self.window,clear=True)
            for line in ax_q_loss.get_lines():
                line.set_linestyle('--')
        if self.q2_losses:
            self._plot_line(ax_q_loss, self.q2_losses, '#a78bfa', 'Q2 loss', self.window,clear=False)
        if self.q1_losses or self.q2_losses:
            ax_q_loss.legend(fontsize=7, facecolor='#161b22', labelcolor='#a0aec0',
                             framealpha=0.7, loc='upper right')
        
        # ---- alpha loss ----
        self._plot_line(ax_alpha, self.alpha_vals,
                        '#fb7185', 'Alpha value', self.window)

        # ---- Q-value trend ----
        if self.q1_vals and self.q2_vals:
            w    = min(self.window, len(self.q1_vals))
            q1sm = self._smooth(self.q1_vals, w)
            q2sm = self._smooth(self.q2_vals, w)
            n    = max(len(self.q1_vals), len(self.q2_vals))
            xs   = np.arange(n - len(q1sm), n)
            ax_qval.plot(xs, q1sm, color='#34d399', linewidth=1.8, label='Q1 mean')
            ax_qval.plot(xs, q2sm, color='#a78bfa', linewidth=1.8, label='Q2 mean')
            ax_qval.legend(fontsize=7, facecolor='#161b22', labelcolor='#a0aec0',
                           framealpha=0.7, loc='upper right')
            ax_qval.set_xlabel('Training step', color='#606878', fontsize=7)

        self._restyle_axes(axes, titles)
        return fig

    # ------------------------------------------------------------------
    # Public save / close
    # ------------------------------------------------------------------
    
    def build_summary_figure(self):
        """Aggregates rewards across all workers and saves a summary plot."""
        # 1. Extract sequences from self.episode_rewards {worker_id: [rewards]}
        reward_sequences = list(self.episode_rewards.values())
        
        # Check if we actually have data
        if not any(reward_sequences):
            print("No episode data to plot.")
            return

        # 2. Align sequences of different lengths using NaN padding
        # zip_longest(*[[1,2], [1,2,3]]) -> [(1,1), (2,2), (None,3)]
        aligned_data = list(zip_longest(*reward_sequences, fillvalue=np.nan))
        
        # 3. Convert to numpy array: shape (max_episodes, num_workers)
        data_matrix = np.array(aligned_data)
        
        # 4. Calculate stats (ignoring NaNs from workers who finished fewer episodes)
        mean_rewards = np.nanmean(data_matrix, axis=1)
        std_rewards = np.nanstd(data_matrix, axis=1)
        episodes = np.arange(len(mean_rewards))

        # 5. Create the Plot
        plt.style.use('dark_background') # Keeping your dashboard aesthetic
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the shaded standard deviation area
        ax.fill_between(episodes, 
                        mean_rewards - std_rewards, 
                        mean_rewards + std_rewards, 
                        color='#00b4d8', alpha=0.2, label='Worker Std Dev')
        
        # Plot the average line
        ax.plot(episodes, mean_rewards, color='#00b4d8', lw=2, label='Mean Reward (All Workers)')
        
        ax.set_title("Aggregated Training Performance (Multi-Worker Mean)", fontsize=14, pad=15)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Total Reward", fontsize=12)
        ax.grid(True, alpha=0.15)
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir,"training_summary_final.png"), dpi=300)
        plt.close(fig)
        print(f"Summary plot saved to training_summary_final.png")

    def save(self, wid : int,tag: str = ''):
        """Render and save a high-res snapshot to save_dir.
        Deletes the previous plot for this worker before saving the new one.
        """
        ts    = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f'training_{tag}_{ts}.png' if tag else f'training_{ts}.png'
        fpath  = os.path.join(self.save_dir, fname)
        fig   = self._build_figure(wid)
        fig.savefig(fpath, dpi=150, facecolor=fig.get_facecolor(),
                    bbox_inches='tight')
        plt.close(fig)

        # Delete the previous plot for this worker now that the new one is saved
        prev = self._last_saved.get(wid)
        if prev and os.path.exists(prev):
            os.remove(prev)
            print(f'[Visualizer] Deleted previous → {prev}')

        self._last_saved[wid] = fpath
        print(f'[Visualizer] Saved → {fpath}')

    def close(self):
        """Save final plot."""
        self.build_summary_figure()

METRICS = ['loss', 'kin', 'gait', 'torque', 'jerk']
METRIC_LABELS = {
    'loss': 'Total Loss', 'kin': 'Avg Kin',
    'gait': 'Avg Gait',  'torque': 'Avg Torque', 'jerk': 'Avg Jerk',
}

def sanitize_filename(s: str) -> str:
    return re.sub(r'[^\w\-_.]', '_', s)

def parse_metrics(line: str):
    """
    FIX 1 — accept 3-or-5 metric lines.

    Some datasets (bacek, hu, gait120, moreira) omit Torque/Jerk and produce
    only 3 numeric tokens.  The old guard `len(nums) < 5` silently dropped
    every one of those lines.  Now we accept anything ≥ 3 and zip with
    however many METRICS are available; missing ones stay as NaN downstream.
    """
    nums = re.findall(r'\d+\.\d+', line)
    if len(nums) < 3:          # need at least loss, kin, gait
        return None
    return {m: float(v) for m, v in zip(METRICS, nums)}

def parse_training_logs(log_file_path: str) -> Dict[Tuple[str, str], Dict]:
    """
    Returns one record per (dataset, activity):
      train_x          : unrolled inner-epoch index
      train_<metric>   : mean across chunks at each step
      val_x            : unrolled x of each val measurement
      val_outer        : outer epoch index for each val point
      val_<metric>     : val values
      test_x           : unrolled x of test (or None)
      test_<metric>    : mean across test chunks (or None)
      outer_boundaries : unrolled x positions of outer epoch starts (for vlines)
    """
    records = {}

    def get_rec(dataset, activity):
        key = (dataset, activity)
        if key not in records:
            records[key] = {
                'dataset': dataset, 'activity': activity,
                'train_x': [],
                **{f'train_{m}': [] for m in METRICS},
                'val_x': [], 'val_outer': [],
                **{f'val_{m}': [] for m in METRICS},
                'test_x': None,
                **{f'test_{m}': None for m in METRICS},
                'outer_boundaries': [],
                '_unroll': 0,
                '_chunk_buf': defaultdict(lambda: defaultdict(list)),
                # FIX 2 — accumulate test chunks so we can average them
                '_test_buf': defaultdict(list),
            }
        return records[key]

    def flush_chunk_buf(rec):
        buf = rec['_chunk_buf']
        if not buf:
            return
        for inner_ep in sorted(buf.keys()):
            chunk_vals = buf[inner_ep]
            rec['train_x'].append(rec['_unroll'])
            rec['_unroll'] += 1
            for m in METRICS:
                vals = chunk_vals.get(m, [])
                rec[f'train_{m}'].append(float(np.mean(vals)) if vals else float('nan'))
        rec['_chunk_buf'] = defaultdict(lambda: defaultdict(list))

    def flush_test_buf(rec):
        """
        FIX 2 — called once at the very end (or when a second OUTER EPOCH
        block would clobber test data).  Averages all accumulated test-chunk
        values into a single representative point placed at the current
        unroll position.
        """
        buf = rec['_test_buf']
        if not buf:
            return
        rec['test_x'] = rec['_unroll']
        for m in METRICS:
            vals = buf.get(m, [])
            rec[f'test_{m}'] = float(np.mean(vals)) if vals else float('nan')
        rec['_test_buf'] = defaultdict(list)

    outer_epoch = 0
    inner_epoch = 0
    current_key = None
    current_mode = None

    with open(log_file_path, 'r') as f:
        for line in f:

            # Outer epoch
            match = re.search(r'OUTER EPOCH (\d+)/(\d+)', line)
            if match:
                outer_epoch = int(match.group(1))
                for rec in records.values():
                    flush_chunk_buf(rec)
                    rec['outer_boundaries'].append(rec['_unroll'])
                current_key = None
                continue

            # Inner epoch
            match = re.search(r'EPOCH (\d+)/(\d+) DATASET', line)
            if match:
                inner_epoch = int(match.group(1))
                current_key = None
                continue

            # TRAINING ON  (handles the double "INFO - INFO -" prefix via re.search)
            match = re.search(r'TRAINING ON\s+(.+?)\s*\|\s*activity=(.+?)\s*\|\s*chunk=(\S+)', line)
            if match:
                dataset, activity = match.group(1).strip(), match.group(2).strip()
                current_key = (dataset, activity)
                current_mode = 'train'
                get_rec(dataset, activity)
                continue

            # VALIDATING ON
            match = re.search(r'VALIDATING ON\s+(.+?)\s*\|\s*activity=(.+?)\s*\|\s*chunk=(\S+)', line)
            if match:
                dataset, activity = match.group(1).strip(), match.group(2).strip()
                current_key = (dataset, activity)
                current_mode = 'val'
                flush_chunk_buf(get_rec(dataset, activity))
                continue

            # TESTING ON
            match = re.search(r'TESTING ON\s+(.+?)\s*\|\s*activity=(.+?)\s*\|\s*chunk=(\S+)', line)
            if match:
                dataset, activity = match.group(1).strip(), match.group(2).strip()
                current_key = (dataset, activity)
                current_mode = 'test'
                continue

            if current_key is None:
                continue

            rec = get_rec(*current_key)

            if current_mode == 'train' and 'train Loss:' in line:
                vals = parse_metrics(line)
                if vals:
                    for m_name, v in vals.items():
                        rec['_chunk_buf'][inner_epoch][m_name].append(v)
                current_key = None

            elif current_mode == 'val' and 'val Loss:' in line:
                vals = parse_metrics(line)
                if vals:
                    rec['val_x'].append(rec['_unroll'])
                    rec['val_outer'].append(outer_epoch)
                    for m in METRICS:                           # ← iterate ALL metrics
                        v = vals.get(m, float('nan'))           # ← NaN if absent
                        rec[f'val_{m}'].append(v)
                current_key = None

            elif current_mode == 'test' and 'test Loss:' in line:
                vals = parse_metrics(line)
                if vals:
                    for m in METRICS:                           # ← same fix
                        rec['_test_buf'][m].append(vals.get(m, float('nan')))
                current_key = None

    # Final flush
    for rec in records.values():
        flush_chunk_buf(rec)
        flush_test_buf(rec)          # FIX 2 — finalize averaged test values
        del rec['_unroll']
        del rec['_chunk_buf']
        del rec['_test_buf']         # FIX 2 — clean up temp buffer

    return records

def plot_training_runs(training_runs: List[Dict], save_dir: str = 'plots'):
    """
    Create plots for each training run showing loss components over epochs.
    
    Args:
        training_runs: List of training run dictionaries from parse_training_logs
        save_dir: Directory to save plots
    """

    Path(save_dir).mkdir(exist_ok=True)
    
    for i, run in enumerate(training_runs):
        if not run['train_x']:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Training Run {i+1}: {run['dataset']} - {run['activity']}")

        
        epochs = run['epochs']
        
        # Plot 1: Total Loss
        axes[0, 0].plot(epochs, run['train_loss'], 'b-o', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, run['val_loss'], 'r-s', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Kinematic Loss
        axes[0, 1].plot(epochs, run['train_kin'], 'b-o', label='Train Kin', linewidth=2)
        axes[0, 1].plot(epochs, run['val_kin'], 'r-s', label='Val Kin', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Kinematic Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Gait Loss
        axes[1, 0].plot(epochs, run['train_gait'], 'b-o', label='Train Gait', linewidth=2)
        axes[1, 0].plot(epochs, run['val_gait'], 'r-s', label='Val Gait', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Gait Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Torque Loss
        axes[1, 1].plot(epochs, run['train_torque'], 'b-o', label='Train Torque', linewidth=2)
        axes[1, 1].plot(epochs, run['val_torque'], 'r-s', label='Val Torque', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Torque Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sanitize filename components
        dataset = sanitize_filename(run['dataset'])
        activity = sanitize_filename(run['activity'])
        chunk = sanitize_filename(run['chunk'])
        
        # Save plot
        filename = f"{save_dir}/run_{i+1}_{dataset}_{activity}_{chunk}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {filename}")

def plot_activity(rec: Dict, save_dir: str = 'plots'):
    """2x3 subplot figure for one (dataset, activity)."""
    Path(save_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"{rec['dataset']}  |  activity={rec['activity']}",
        fontsize=14, fontweight='bold'
    )

    for ax, metric in zip(axes.flat, METRICS):
        train_x = rec['train_x']
        train_y = rec[f'train_{metric}']
        val_x   = rec['val_x']
        val_y   = rec[f'val_{metric}']
        test_x  = rec['test_x']
        test_y  = rec[f'test_{metric}']

        ax.plot(train_x, train_y, color='steelblue', linewidth=1.8,
                label='Train', zorder=3)

        if val_x:
            ax.scatter(val_x, val_y, color='tomato', s=60, zorder=5,
                       label='Val', marker='D')
            ax.plot(val_x, val_y, color='tomato', linewidth=1,
                    linestyle='--', zorder=4, alpha=0.6)

        if test_x is not None and test_y is not None:
            ax.scatter([test_x], [test_y], color='seagreen', s=100,
                       zorder=6, marker='*', label='Test')

        for bx in rec['outer_boundaries']:
            ax.axvline(bx, color='gray', linewidth=1.2,
                       linestyle='--', alpha=0.6, zorder=1)

        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel('Inner Epoch (unrolled)')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes.flat[5].set_visible(False)
    plt.tight_layout()

    fname = (f"{save_dir}/"
             f"{sanitize_filename(rec['dataset'])}_"
             f"{sanitize_filename(rec['activity'])}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def plot_combined(records: Dict, save_dir: str = 'plots'):
    """Overview — total loss for all (dataset, activity) series."""
    Path(save_dir).mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("All Activities — Total Loss Overview",
                 fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(records)))

    for ((dataset, activity), rec), color in zip(records.items(), colors):
        label = f"{dataset} / {activity}"
        ax.plot(rec['train_x'], rec['train_loss'],
                color=color, linewidth=1.8, label=label)
        if rec['val_x']:
            ax.scatter(rec['val_x'], rec['val_loss'],
                       color=color, s=50, marker='D', zorder=5)
        if rec['test_x'] is not None:
            ax.scatter([rec['test_x']], [rec['test_loss']],
                       color=color, s=80, marker='*', zorder=6)

    ref_rec = max(records.values(), key=lambda r: len(r['outer_boundaries']))
    for bx in ref_rec['outer_boundaries']:
        ax.axvline(bx, color='gray', linewidth=1.2,
                   linestyle='--', alpha=0.6, zorder=1)

    ax.set_xlabel('Inner Epoch (unrolled)')
    ax.set_ylabel('Total Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()

    fname = f"{save_dir}/combined_overview.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

def create_plots(log_file: str, save_dir: str = 'plots'):
    print("Parsing log file...")
    records = parse_training_logs(log_file)
    print(f"Found {len(records)} (dataset, activity) series:")
    for (ds, act), rec in records.items():
        print(f"  {ds} / {act}  —  {len(rec['train_x'])} inner epochs, "
              f"{len(rec['val_x'])} val points, "
              f"test={'yes' if rec['test_x'] is not None else 'no'}")

    print("\nGenerating plots...")
    for rec in records.values():
        plot_activity(rec, save_dir)
    plot_combined(records, save_dir)
    print("Done.")

def plot_test_data(model, test_obj):
    emg_mask = test_obj.data['test']['masks']['emg']  # shape = 13
    kinetic_mask = test_obj.data['test']['masks']['kinetic']
    kinematic_mask = test_obj.data['test']['masks']['kinematic']

    stride_sequences = []
    current_stride_start = 0

    curr_min = 1e9
    curr_max = 1e-9
    
    for i in range(len(test_obj)):
        data_t = test_obj[i]  # Use __getitem__
        timestep = data_t['metadata']['window_idx']
        curr_min = min(timestep,curr_min)
        curr_max = max(timestep,curr_max)
        
        # When we hit window_idx=0 and we're not at the first sample
        if timestep == 0 and i != 0:
            # Previous stride ended at i-1
            data_prev = test_obj[i-1]
            timestep_prev = data_prev['metadata']['window_idx']
            
            stride_info = {
                'start_idx': current_stride_start,
                'end_idx': i - 1,
                'length': i - current_stride_start,
                'end_window_idx': timestep_prev,
                'metadata': data_prev['metadata']  # Capture patient/activity info
            }

            stride_sequences.append(stride_info)
            
            # New stride starts here
            current_stride_start = i
    
    # Don't forget the last stride!
    if len(test_obj) > 0:
        data_last = test_obj[len(test_obj) - 1]
        timestep_last = data_last['metadata']['window_idx']
        
        stride_info = {
            'start_idx': current_stride_start,
            'end_idx': len(test_obj) - 1,
            'length': len(test_obj) - current_stride_start,
            'end_window_idx': timestep_last,
            'metadata': data_last['metadata']
        }
        stride_sequences.append(stride_info)
    
    if len(stride_sequences) == 0:
        print('No strides found!')
        return
    
    # Find complete strides (window_idx goes 0 to 199, so length=200)
    complete_strides = []
    for stride in stride_sequences:
        if stride['length'] >=198 and stride['end_window_idx'] >=197:
            complete_strides.append(stride)
    
    print(f'Total test strides found: {len(stride_sequences)}')
    print(f'Complete test strides (length=200): {len(complete_strides)}')
    print(f'Max test stride idx found {curr_max}, \n Min test stride idx found {curr_min}')
    
    # for i, stride in enumerate(complete_strides):
    #     print(f'\nComplete stride {i}:')
    #     print(f"  Indices: {stride['start_idx']} to {stride['end_idx']}")
    #     print(f"  Patient: {stride['metadata']['patient_id']}")
    #     print(f"  Activity: {stride['metadata']['activity']}")
    #     print(f"  Direction: {stride['metadata']['direction']}")
    #     print(f"  Dataset: {stride['metadata']['dataset']}")
    #     print(f"  Has torque: {stride['metadata']['has_torque']}")
    
    # TODO: Now visualize one of these complete strides
    if len(complete_strides) > 0:
        # Pick the first complete stride for now
        selected_stride = complete_strides[0]
        visualize_stride(model, test_obj, selected_stride, emg_mask, kinetic_mask, kinematic_mask)
    
    return complete_strides

def visualize_stride(model, test_obj, stride_info, emg_mask, kinematic_mask, kinetic_mask, 
                     save_path=None, fps=30):
    """
    Visualize a complete stride with EMG, kinematics, and kinetics in 3D
    
    Args:
        model: Your trained model
        test_obj: SplitDataset object
        stride_info: Dict with 'start_idx', 'end_idx', 'metadata'
        emg_mask: Boolean mask for EMG channels (13,)
        kinematic_mask: Boolean mask for kinematic features (27,)
        kinetic_mask: Boolean mask for kinetic features (9,)
        save_path: Optional path to save video
        fps: Frames per second for animation
    """
    
    # EMG channel names mapping
    EMG_CHANNEL_NAMES = [
        'VL',      # 0
        'RF',      # 1
        'VM.',      # 2
        'TA',    # 3
        'BF',      # 4
        'ST,SM',        # 5 - Semitendinosus/Semimembranosus
        'GastM',    # 6
        'GastL.',    # 7
        'SL',           # 8
        'PL.',   # 9
        'PB',   # 10
        'Gluteus Med.',     # 11
        'Gluteus Max.'      # 12
    ]
    
    # Extract the full stride data
    start_idx = stride_info['start_idx']
    end_idx = stride_info['end_idx']
    n_frames = end_idx - start_idx + 1
    
    print(f"Loading stride from indices {start_idx} to {end_idx} ({n_frames} frames)")
    
    # Collect all data for this stride
    emg_data = []
    gt_kin_states = []
    gt_torques = []
    gait_pcts = []
    input_kin_states = []
    input_gait_pcts = []
    
    for i in range(start_idx, end_idx + 1):
        sample = test_obj[i]
        emg_data.append(sample['emg'])
        gt_kin_states.append(sample['target_kin_state'])
        gt_torques.append(sample['target_torque'])
        gait_pcts.append(sample['target_gait_pct'])
        input_kin_states.append(sample['input_kin_state'])
        input_gait_pcts.append(sample['input_gait_pct'])
    
    # Stack into tensors
    emg_data = torch.stack(emg_data).to(model.device)  # (200, 13, 100)
    gt_kin_states = torch.stack(gt_kin_states).to(model.device)  # (200, 27)
    gt_torques = torch.stack(gt_torques).to(model.device)  # (200, 9)
    gait_pcts = torch.stack(gait_pcts).to(model.device)  # (200,)
    input_kin_states = torch.stack(input_kin_states).to(model.device)  # (200, 27)
    input_gait_pcts = torch.stack(input_gait_pcts).to(model.device)  # (200,)
    
    # Run model inference
    print("Running model inference...")
    model.eval()
    with torch.no_grad():
        # Run inference for each timepoint
        pred_kin_states = []
        pred_gait_pcts = []
        pred_impedances = []
        
        for i in range(n_frames):
            outputs = model(
                emg_data[i:i+1], 
                input_kin_states[i:i+1], 
                input_gait_pcts[i:i+1]
            )
            pred_kin_states.append(outputs['pred_kin_state'])
            pred_gait_pcts.append(outputs['pred_gait_pct'])
            if 'pred_impedance' in outputs:
                pred_impedances.append(outputs['pred_impedance'])
        
        pred_kin_states = torch.cat(pred_kin_states, dim=0)  # (200, 27)
        pred_gait_pcts = torch.cat(pred_gait_pcts, dim=0)  # (200,)
        if len(pred_impedances) > 0:
            pred_impedances = torch.cat(pred_impedances, dim=0)  # (200, 9)
        else:
            pred_impedances = None
    
    # Convert to numpy
    emg_np = emg_data.cpu().numpy()
    gt_kin_np = gt_kin_states.cpu().numpy()
    pred_kin_np = pred_kin_states.cpu().numpy()
    gt_torque_np = gt_torques.cpu().numpy()
    pred_impedance_np = pred_impedances.cpu().numpy() if pred_impedances is not None else None
    gait_pct_np = gait_pcts.cpu().numpy()
    
    # Parse kinematic data into angles, velocities, accelerations
    def parse_kinematics(kin_state):
        """
        Parse 27D kinematic state into structured dict
        Layout: [angles(9), omega(9), alpha(9)]
        Each group of 9: [hip_roll, hip_yaw, hip_pitch, knee_roll, knee_yaw, knee_pitch, ankle_roll, ankle_yaw, ankle_pitch]
        """
        angles = kin_state[:, 0:9]
        omega = kin_state[:, 9:18]
        alpha = kin_state[:, 18:27]
        
        return {
            'hip': {
                'angles': angles[:, 0:3],    # roll, yaw, pitch
                'omega': omega[:, 0:3],
                'alpha': alpha[:, 0:3]
            },
            'knee': {
                'angles': angles[:, 3:6],    # roll, yaw, pitch
                'omega': omega[:, 3:6],
                'alpha': alpha[:, 3:6]
            },
            'ankle': {
                'angles': angles[:, 6:9],    # roll, yaw, pitch
                'omega': omega[:, 6:9],
                'alpha': alpha[:, 6:9]
            }
        }
    
    gt_parsed = parse_kinematics(gt_kin_np)
    pred_parsed = parse_kinematics(pred_kin_np)
    
    # Select active EMG channels
    active_emg_channels = np.where(emg_mask)[0]
    print(f"Active EMG channels: {active_emg_channels}")
    
    # Create figure with subplots - INCREASED FIGURE SIZE AND ADJUSTED SPACING
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.35, 
                          height_ratios=[1.8, 1, 1, 1, 1],
                          left=0.08, right=0.96, top=0.92, bottom=0.08)
    
    # 3D stick figure subplots (top row)
    ax_stick_pred = fig.add_subplot(gs[0, 0], projection='3d')
    ax_stick_gt = fig.add_subplot(gs[0, 1], projection='3d')
    
    # Joint angles (all three axes)
    ax_angles = fig.add_subplot(gs[1, :])
    
    # Torques/Impedance
    ax_torques = fig.add_subplot(gs[2, :])
    
    # EMG (show up to 6 most active channels)
    ax_emg = fig.add_subplot(gs[3, :])
    
    # Gait percentage
    ax_gait = fig.add_subplot(gs[4, :])
    
    # Setup 3D stick figures
    def setup_stick_figure_3d(ax, title):
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_zlim(0, 1.2)
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.view_init(elev=15, azim=45)  # Good viewing angle
        ax.grid(True, alpha=0.3)
        return ax
    
    setup_stick_figure_3d(ax_stick_pred, 'Predicted')
    setup_stick_figure_3d(ax_stick_gt, 'Ground Truth')
    
    # 3D stick figure helper using rotation matrices
    def rotation_matrix_x(angle):
        """Rotation matrix around X-axis (roll)"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0],
                        [0, c, -s],
                        [0, s, c]])
    
    def rotation_matrix_y(angle):
        """Rotation matrix around Y-axis (yaw)"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])
    
    def rotation_matrix_z(angle):
        """Rotation matrix around Z-axis (pitch)"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])
    
    def apply_rotation(roll, yaw, pitch):
        """Apply roll-yaw-pitch rotation"""
        R = rotation_matrix_z(pitch) @ rotation_matrix_y(yaw) @ rotation_matrix_x(roll)
        return R
    
    def draw_stick_figure_3d(ax, hip_angles, knee_angles, ankle_angles, color='blue'):
        """
        Draw a 3D stick figure
        hip_angles, knee_angles, ankle_angles: [roll, yaw, pitch] in radians
        """
        # Segment lengths
        thigh_length = 0.4
        shank_length = 0.4
        foot_length = 0.15
        
        # Start at hip (origin)
        hip_pos = np.array([0.0, 0.0, 1.0])
        
        # Initial thigh direction (pointing down)
        thigh_dir = np.array([0.0, 0.0, -thigh_length])
        
        # Apply hip rotation
        R_hip = apply_rotation(hip_angles[0], hip_angles[1], hip_angles[2])
        thigh_rotated = R_hip @ thigh_dir
        knee_pos = hip_pos + thigh_rotated
        
        # Initial shank direction (continuing down)
        shank_dir = np.array([0.0, 0.0, -shank_length])
        
        # Apply knee rotation (relative to thigh orientation)
        R_knee = apply_rotation(knee_angles[0], knee_angles[1], knee_angles[2])
        shank_rotated = R_hip @ R_knee @ shank_dir
        ankle_pos = knee_pos + shank_rotated
        
        # Initial foot direction (forward)
        foot_dir = np.array([foot_length, 0.0, 0.0])
        
        # Apply ankle rotation (relative to shank orientation)
        R_ankle = apply_rotation(ankle_angles[0], ankle_angles[1], ankle_angles[2])
        foot_rotated = R_hip @ R_knee @ R_ankle @ foot_dir
        toe_pos = ankle_pos + foot_rotated
        
        # Draw segments
        lines = []
        # Thigh
        line1, = ax.plot([hip_pos[0], knee_pos[0]], 
                         [hip_pos[1], knee_pos[1]], 
                         [hip_pos[2], knee_pos[2]], 
                         'o-', color=color, linewidth=4, markersize=10)
        # Shank
        line2, = ax.plot([knee_pos[0], ankle_pos[0]], 
                         [knee_pos[1], ankle_pos[1]], 
                         [knee_pos[2], ankle_pos[2]], 
                         'o-', color=color, linewidth=4, markersize=10)
        # Foot
        line3, = ax.plot([ankle_pos[0], toe_pos[0]], 
                         [ankle_pos[1], toe_pos[1]], 
                         [ankle_pos[2], ankle_pos[2]], 
                         's-', color=color, linewidth=3, markersize=8)
        
        return [line1, line2, line3]
    
    # Initialize stick figures
    stick_pred_lines = draw_stick_figure_3d(ax_stick_pred, 
                                            pred_parsed['hip']['angles'][0], 
                                            pred_parsed['knee']['angles'][0],
                                            pred_parsed['ankle']['angles'][0],
                                            color='red')
    stick_gt_lines = draw_stick_figure_3d(ax_stick_gt,
                                          gt_parsed['hip']['angles'][0],
                                          gt_parsed['knee']['angles'][0],
                                          gt_parsed['ankle']['angles'][0],
                                          color='blue')
    
    # Setup joint angles plot (showing pitch primarily, but can show all)
    time_vec = np.arange(n_frames)
    ax_angles.set_xlim(0, n_frames)
    ax_angles.set_xlabel('', fontsize=11)
    ax_angles.set_ylabel('Joint Angle (rad)', fontsize=11)
    ax_angles.set_title('Joint Angles (Pitch)', fontsize=12, fontweight='bold', pad=10)
    ax_angles.grid(True, alpha=0.3)
    
    # Plot pitch angles (most important for gait) - index 2
    line_hip_gt, = ax_angles.plot(time_vec, gt_parsed['hip']['angles'][:, 2], 
                                   'b-', label='Hip GT', linewidth=2)
    line_hip_pred, = ax_angles.plot(time_vec, pred_parsed['hip']['angles'][:, 2], 
                                     'r--', label='Hip Pred', linewidth=2)
    
    line_knee_gt, = ax_angles.plot(time_vec, gt_parsed['knee']['angles'][:, 2], 
                                    'g-', label='Knee GT', linewidth=2)
    line_knee_pred, = ax_angles.plot(time_vec, pred_parsed['knee']['angles'][:, 2], 
                                      'orange', linestyle='--', label='Knee Pred', linewidth=2)
    
    line_ankle_gt, = ax_angles.plot(time_vec, gt_parsed['ankle']['angles'][:, 2], 
                                     'purple', label='Ankle GT', linewidth=2)
    line_ankle_pred, = ax_angles.plot(time_vec, pred_parsed['ankle']['angles'][:, 2], 
                                       'brown', linestyle='--', label='Ankle Pred', linewidth=2)
    
    vline_angles = ax_angles.axvline(0, color='black', linewidth=2, linestyle='-', alpha=0.5)
    ax_angles.legend(loc='upper right', ncol=6, fontsize=9, framealpha=0.9)
    
    # Setup torques plot
    ax_torques.set_xlim(0, n_frames)
    ax_torques.set_xlabel('', fontsize=11)
    ax_torques.set_ylabel('Torque (N·m)', fontsize=11)
    ax_torques.set_title('Joint Torques (Pitch)', fontsize=12, fontweight='bold', pad=10)
    ax_torques.grid(True, alpha=0.3)
    
    # Show pitch torques primarily (indices 2, 5, 8)
    if pred_impedance_np is not None:
        line_hip_torque_gt, = ax_torques.plot(time_vec, gt_torque_np[:, 2], 
                                              'b-', label='Hip GT', linewidth=2)
        line_hip_torque_pred, = ax_torques.plot(time_vec, pred_impedance_np[:, 2], 
                                                'r--', label='Hip Pred', linewidth=2)
        
        line_knee_torque_gt, = ax_torques.plot(time_vec, gt_torque_np[:, 5], 
                                               'g-', label='Knee GT', linewidth=2)
        line_knee_torque_pred, = ax_torques.plot(time_vec, pred_impedance_np[:, 5], 
                                                 'orange', linestyle='--', label='Knee Pred', linewidth=2)
        
        line_ankle_torque_gt, = ax_torques.plot(time_vec, gt_torque_np[:, 8], 
                                                'purple', label='Ankle GT', linewidth=2)
        line_ankle_torque_pred, = ax_torques.plot(time_vec, pred_impedance_np[:, 8], 
                                                  'brown', linestyle='--', label='Ankle Pred', linewidth=2)
    else:
        # Only ground truth available
        ax_torques.plot(time_vec, gt_torque_np[:, 2], 'b-', label='Hip', linewidth=2)
        ax_torques.plot(time_vec, gt_torque_np[:, 5], 'g-', label='Knee', linewidth=2)
        ax_torques.plot(time_vec, gt_torque_np[:, 8], 'purple', label='Ankle', linewidth=2)
    
    vline_torques = ax_torques.axvline(0, color='black', linewidth=2, linestyle='-', alpha=0.5)
    ax_torques.legend(loc='upper right', ncol=6, fontsize=9, framealpha=0.9)
    
    # Setup EMG plot (show top 6 active channels)
    emg_rms = np.sqrt(np.mean(emg_np**2, axis=(0, 2)))  # (13,)
    active_rms = emg_rms[active_emg_channels]
    top_indices = np.argsort(active_rms)[-6:]
    top_channels = active_emg_channels[top_indices]
    
    ax_emg.set_xlim(0, n_frames)
    ax_emg.set_xlabel('', fontsize=11)
    ax_emg.set_ylabel('Normalized Amplitude', fontsize=11)
    
    # Use proper EMG channel names in title
    channel_names_display = ', '.join([EMG_CHANNEL_NAMES[ch] for ch in top_channels])
    ax_emg.set_title(f'EMG Activity: {channel_names_display}', fontsize=12, fontweight='bold', pad=10)
    ax_emg.grid(True, alpha=0.3)
    
    # Compute EMG envelopes (RMS over the 100-sample window)
    emg_envelopes = np.sqrt(np.mean(emg_np**2, axis=2))  # (200, 13)
    
    # Normalize and plot each channel with proper names
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_channels)))
    for i, ch in enumerate(top_channels):
        envelope = emg_envelopes[:, ch]
        envelope_norm = (envelope - envelope.min()) / (envelope.max() - envelope.min() + 1e-8)
        offset = i * 1.3  # Increased spacing to prevent overlap
        ax_emg.plot(time_vec, envelope_norm + offset, 
                   label=EMG_CHANNEL_NAMES[ch],  # Use proper names
                   linewidth=1.5, color=colors[i])
    
    vline_emg = ax_emg.axvline(0, color='black', linewidth=2, linestyle='-', alpha=0.5)
    ax_emg.legend(loc='upper right', ncol=3, fontsize=9, framealpha=0.9)  # Changed to 3 columns for better spacing
    ax_emg.set_ylim(-0.5, len(top_channels) * 1.3 + 0.5)
    ax_emg.set_yticks([])  # Remove y-ticks since they're normalized and stacked
    
    # Setup gait percentage plot
    ax_gait.set_xlim(0, n_frames)
    ax_gait.set_ylim(0, 105)
    ax_gait.set_xlabel('', fontsize=11)
    ax_gait.set_ylabel('Gait Cycle (%)', fontsize=11)
    ax_gait.set_title('Gait Cycle Progression', fontsize=12, fontweight='bold', pad=10)
    ax_gait.grid(True, alpha=0.3)
    ax_gait.plot(time_vec, gait_pct_np, 'k-', linewidth=2, label='Gait %')
    ax_gait.fill_between(time_vec, 0, gait_pct_np, alpha=0.3, color='blue')
    vline_gait = ax_gait.axvline(0, color='red', linewidth=2, linestyle='-', alpha=0.7)
    ax_gait.legend(fontsize=9)
    
    # Add metadata text - IMPROVED FORMATTING
    metadata = stride_info['metadata']
    title_text = (f"Patient: {metadata['patient_id']}  |  "
                 f"Activity: {metadata['activity']}  |  "
                 f"Direction: {metadata['direction']}  |  "
                 f"Dataset: {metadata['dataset']}")
    fig.suptitle(title_text, fontsize=13, fontweight='bold', y=0.96)
    
    # Animation update function
    current_frame = [0]
    is_playing = [True]
    
    def update(frame):
        current_frame[0] = frame
        
        # Update 3D stick figures
        for line in stick_pred_lines:
            line.remove()
        for line in stick_gt_lines:
            line.remove()
        
        stick_pred_lines.clear()
        stick_gt_lines.clear()
        
        pred_lines = draw_stick_figure_3d(
            ax_stick_pred,
            pred_parsed['hip']['angles'][frame],
            pred_parsed['knee']['angles'][frame],
            pred_parsed['ankle']['angles'][frame],
            color='red'
        )
        stick_pred_lines.extend(pred_lines)
        
        gt_lines = draw_stick_figure_3d(
            ax_stick_gt,
            gt_parsed['hip']['angles'][frame],
            gt_parsed['knee']['angles'][frame],
            gt_parsed['ankle']['angles'][frame],
            color='blue'
        )
        stick_gt_lines.extend(gt_lines)
        
        # Update vertical lines
        vline_angles.set_xdata([frame, frame])
        vline_torques.set_xdata([frame, frame])
        vline_emg.set_xdata([frame, frame])
        vline_gait.set_xdata([frame, frame])
        
        return stick_pred_lines + stick_gt_lines + [vline_angles, vline_torques, vline_emg, vline_gait]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True, repeat=True)
    
    # Add interactive controls - ADJUSTED POSITIONS FOR BETTER LAYOUT
    ax_button = plt.axes([0.45, 0.03, 0.1, 0.025])
    btn_play_pause = Button(ax_button, 'Pause')
    
    def toggle_play_pause(event):
        if is_playing[0]:
            anim.event_source.stop()
            btn_play_pause.label.set_text('Play')
            is_playing[0] = False
        else:
            anim.event_source.start()
            btn_play_pause.label.set_text('Pause')
            is_playing[0] = True
    
    btn_play_pause.on_clicked(toggle_play_pause)
    
    # Add frame slider
    ax_slider = plt.axes([0.2, 0.015, 0.6, 0.015])
    slider_frame = Slider(ax_slider, 'Frame', 0, n_frames-1, valinit=0, valstep=1)
    
    def update_slider(val):
        frame = int(slider_frame.val)
        fig.canvas.draw_idle()
    
    slider_frame.on_changed(update_slider)
    
    # Save if requested
    if save_path is not None:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=fps, dpi=100)
        print("Saved!")
    
    plt.show()
    
    return anim

def create_plots(log_file: str, save_dir: str = 'plots'):
    records = parse_training_logs(log_file)
    for rec in records.values():
        plot_activity(rec, save_dir)


def main():
    create_plots('C:/EMG/software/plots/server/training_20260309_004842.log')

if __name__ == '__main__':
    main()