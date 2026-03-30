print('program')

import deprl
import sconegym
import gym
import argparse
import multiprocessing as mp
from trainFM import EMGTransformer, ReplayBuffer, QNetwork, compute_impedance_torque, soft_update
import os
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque
from visualizer import TrainingVisualizer
from typing import List, Dict, Tuple, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Noise Configuration + Helpers   (UNCHANGED)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NoiseConfig:
    """
    Sim-to-real noise configuration. All magnitude params are UPPER BOUNDS —
    actual noise is domain-randomized fresh each draw so the policy learns
    robustness across the full distribution [0, max], including the clean case.
    """
    emg_noise_std_max:  float = 0.0
    emg_noise_mean_max: float = 0.0
    kin_noise_std_max:  float = 0.0
    kin_noise_mean_max: float = 0.0
    emg_jitter_max: int = 0
    kin_jitter_max: int = 0
    jitter_warmup_steps: int = 0
    noise_on_rollout: bool = False
    noise_on_replay:  bool = False

    def effective_jitter(self, curr_step: int) -> Tuple[int, int]:
        if self.jitter_warmup_steps <= 0:
            return self.emg_jitter_max, self.kin_jitter_max
        scale = min(1.0, curr_step / self.jitter_warmup_steps)
        return (
            int(round(self.emg_jitter_max * scale)),
            int(round(self.kin_jitter_max * scale)),
        )

    @property
    def any_emg_noise(self) -> bool:
        return self.emg_noise_std_max > 0.0 or self.emg_noise_mean_max > 0.0

    @property
    def any_kin_noise(self) -> bool:
        return self.kin_noise_std_max > 0.0 or self.kin_noise_mean_max > 0.0


# ── Single-sample noise (rollout)  (UNCHANGED) ───────────────────────────────

def _signal_noise_emg_single(emg: torch.Tensor, cfg: NoiseConfig) -> torch.Tensor:
    if not cfg.any_emg_noise:
        return emg
    C = emg.shape[0]
    sigmas = torch.tensor(
        np.random.uniform(0.0, cfg.emg_noise_std_max, C),
        dtype=torch.float32, device=emg.device,
    ).unsqueeze(-1)
    means = torch.tensor(
        np.random.uniform(-cfg.emg_noise_mean_max, cfg.emg_noise_mean_max, C),
        dtype=torch.float32, device=emg.device,
    ).unsqueeze(-1)
    noise = means + sigmas * torch.randn_like(emg)
    return torch.clamp(emg + noise, -1.0, 1.0)


def _signal_noise_kin_single(kin: torch.Tensor, cfg: NoiseConfig) -> torch.Tensor:
    if not cfg.any_kin_noise:
        return kin
    D = kin.shape[0]
    sigmas = torch.tensor(
        np.random.uniform(0.0, cfg.kin_noise_std_max, D),
        dtype=torch.float32, device=kin.device,
    )
    means = torch.tensor(
        np.random.uniform(-cfg.kin_noise_mean_max, cfg.kin_noise_mean_max, D),
        dtype=torch.float32, device=kin.device,
    )
    return kin + means + sigmas * torch.randn_like(kin)


def _emg_window_from_frame_buffer(emg_frame_buf: deque, delta: int) -> torch.Tensor:
    buf = list(emg_frame_buf)
    if delta == 0:
        frames = buf[-100:]
    else:
        frames = buf[-(100 + delta):-delta]
    return torch.stack(frames, dim=1)  # (13, 100)


def _rollout_emg_noise(
    emg_frame_buf: deque,
    cfg: NoiseConfig,
    eff_jitter: int,
    device: torch.device,
) -> torch.Tensor:
    delta = int(np.random.randint(0, eff_jitter + 1)) if eff_jitter > 0 else 0
    w = _emg_window_from_frame_buffer(emg_frame_buf, delta).to(device)
    return _signal_noise_emg_single(w, cfg)


def _rollout_kin_noise(
    kin: torch.Tensor,
    cfg: NoiseConfig,
    kin_buffer: deque,
    eff_jitter: int,
) -> torch.Tensor:
    k = kin
    if eff_jitter > 0 and len(kin_buffer) > 1:
        max_lookback = min(eff_jitter, len(kin_buffer) - 1)
        delta = int(np.random.randint(0, max_lookback + 1))
        if delta > 0:
            k = kin_buffer[-(delta + 1)].to(kin.device)
    return _signal_noise_kin_single(k, cfg)


# ── Batch noise (replay)  (UNCHANGED) ────────────────────────────────────────

def _batch_signal_noise_emg(emg: torch.Tensor, cfg: NoiseConfig) -> torch.Tensor:
    if not cfg.any_emg_noise:
        return emg
    B, C, _ = emg.shape
    sigmas = torch.tensor(
        np.random.uniform(0.0, cfg.emg_noise_std_max, (B, C)),
        dtype=torch.float32, device=emg.device,
    ).unsqueeze(-1)
    means = torch.tensor(
        np.random.uniform(-cfg.emg_noise_mean_max, cfg.emg_noise_mean_max, (B, C)),
        dtype=torch.float32, device=emg.device,
    ).unsqueeze(-1)
    noise = means + sigmas * torch.randn_like(emg)
    return torch.clamp(emg + noise, -1.0, 1.0)


def _batch_signal_noise_kin(kin: torch.Tensor, cfg: NoiseConfig) -> torch.Tensor:
    if not cfg.any_kin_noise:
        return kin
    B, D = kin.shape
    sigmas = torch.tensor(
        np.random.uniform(0.0, cfg.kin_noise_std_max, (B, D)),
        dtype=torch.float32, device=kin.device,
    )
    means = torch.tensor(
        np.random.uniform(-cfg.kin_noise_mean_max, cfg.kin_noise_mean_max, (B, D)),
        dtype=torch.float32, device=kin.device,
    )
    return kin + means + sigmas * torch.randn_like(kin)


# ──────────────────────────────────────────────────────────────────────────────
# NoisyReplayBuffer  (MODIFIED — adds worker_id tracking + new jitter walk)
# ──────────────────────────────────────────────────────────────────────────────

class NoisyReplayBuffer(ReplayBuffer):
    """
    Extends ReplayBuffer with multi-env awareness.

    Changes vs. original:
      - worker_id_memory: int array parallel to all other memory arrays.
        Stores which env worker wrote each slot, set in store_transition().

      - _walk_back(): replaces _clamp_to_episode(). Instead of clamping a raw
        delta, walks backward through the buffer counting only transitions that
        belong to the same worker and the same episode (no done crossed).
        Returns the absolute buffer index that is `delta` valid steps ago.

      - sample_with_jitter(): uses _walk_back() so that temporal jitter δ
        always means "δ steps ago in *this env's* episode history", regardless
        of how other workers' transitions are interleaved in the buffer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One int per slot — which worker wrote it (-1 = uninitialised)
        self.worker_id_memory = np.full(self.mem_size, -1, dtype=np.int32)

    # ── store ─────────────────────────────────────────────────────────────────

    def store_transition(self, state, action, reward, state_, done, worker_id: int = 0):
        """
        Drop-in replacement for ReplayBuffer.store_transition with an extra
        worker_id argument. Records worker_id at the same circular slot.
        """
        idx = self.ptr                          # capture slot before parent advances ptr
        super().store_transition(state, action, reward, state_, done)
        self.worker_id_memory[idx] = worker_id

    # ── jitter walk ───────────────────────────────────────────────────────────

    def _walk_back(self, idx: int, delta: int, worker_id: int) -> int:
        """
        Walk backward from `idx`, counting only steps that:
          (a) belong to `worker_id`, AND
          (b) do not cross a done=True signal (episode boundary).

        Returns the absolute buffer index that is `delta` valid steps ago,
        or the furthest reachable index if fewer than `delta` valid steps exist.

        This replaces the old _clamp_to_episode + raw subtraction pattern.
        With multiple workers interleaved in the buffer, raw index arithmetic
        is wrong — skipping over another worker's slots is mandatory.
        """
        if delta == 0:
            return idx

        valid_steps = 0
        sample_ptr = idx
        prev = idx

        # Safety: never scan more slots than the filled buffer
        max_scan = self.mem_size // self.num_workers

        for _ in range(max_scan):
            prev = (prev - 1) % self.mem_size

            if (self.terminal_memory[prev] and self.worker_id_memory[prev] == worker_id) or (self.worker_id_memory[prev] == -1):
                break

            if self.worker_id_memory[prev] != worker_id:
                continue

            if self.worker_id_memory[prev] == worker_id:
                sample_ptr = prev
                valid_steps += 1

                if valid_steps == delta:
                    break

        return sample_ptr                              # absolute buffer index, δ valid steps back

    # ── sample ────────────────────────────────────────────────────────────────

    def sample_with_jitter(
        self,
        batch_size: int,
        noise_cfg: 'NoiseConfig',
        bilateral: bool = False,
        curr_step: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch with per-element temporal jitter that is worker-aware.

        For each element b:
          δ_emg ~ U[0, eff_emg_jitter]
          δ_kin ~ U[0, eff_kin_jitter]

        src index is found via _walk_back() which skips interleaved slots from
        other workers and stops at episode boundaries — so δ always means
        genuine temporal distance within that env's own episode.
        """
        max_mem = min(self.size, self.mem_size)
        assert max_mem > 0, 'Buffer is empty!'
        batch_size = min(batch_size, max_mem)
        base_indices = np.random.choice(max_mem, batch_size, replace=(max_mem < batch_size))

        eff_emg_jitter, eff_kin_jitter = noise_cfg.effective_jitter(curr_step)

        states  = self.state_memory[base_indices].copy()
        states_ = self.new_state_memory[base_indices].copy()

        if eff_emg_jitter > 0 or eff_kin_jitter > 0:
            sides_slices = (
                [(0, 1300, 1300, 1327), (1327, 2627, 2627, 2654)]
                if bilateral else
                [(0, 1300, 1300, 1327)]
            )

            for b, idx in enumerate(base_indices):
                wid = int(self.worker_id_memory[idx])

                for es, ee, ks, ke in sides_slices:

                    if eff_emg_jitter > 0:
                        # δ = desired temporal distance in this env's history
                        d_emg = int(np.random.randint(0, eff_emg_jitter + 1))
                        src   = self._walk_back(int(idx), d_emg, wid)
                        if src != idx:
                            states[b, es:ee] = self.state_memory[src, es:ee]

                    if eff_kin_jitter > 0:
                        d_kin = int(np.random.randint(0, eff_kin_jitter + 1))
                        src   = self._walk_back(int(idx), d_kin, wid)
                        if src != idx:
                            states[b, ks:ke] = self.state_memory[src, ks:ke]

        actions = self.action_memory[base_indices]
        rewards = self.reward_memory[base_indices]
        dones   = self.terminal_memory[base_indices]

        return states, states_, actions, rewards, dones


# ──────────────────────────────────────────────────────────────────────────────
# SAC Training  (UNCHANGED except N gradient steps driven by caller)
# ──────────────────────────────────────────────────────────────────────────────

def parse_sides(states, actions=None, bilateral=False):
    B = states.shape[0]
    if bilateral:
        emg_R = states[:, :1300].reshape(B, 13, 100)
        kin_R = states[:, 1300:1327]
        emg_L = states[:, 1327:2627].reshape(B, 13, 100)
        kin_L = states[:, 2627:2654]
        sides = [(emg_R, kin_R), (emg_L, kin_L)]
        if actions is not None:
            return sides, [actions[:, :54], actions[:, 54:]]
    else:
        emg = states[:, :1300].reshape(B, 13, 100)
        kin = states[:, 1300:1327]
        sides = [(emg, kin)]
        if actions is not None:
            return sides, [actions]
    return sides


def train_sac(optimizer_and_scheduler, policy_args, critic_args, Policy,
              QNetwork_base1, QNetwork_base2,
              QNetwork_target1, QNetwork_target2,
              replay_buff, training_epochs, training_losses,
              bilateral=False, sample_batch_size=256,
              noise_cfg: Optional[NoiseConfig] = None,
              curr_step: int = 0):

    gamma, tau = 0.99, 0.005
    training_iterations = 0

    if replay_buff.size < sample_batch_size:
        return

    while training_iterations < training_epochs:

        if noise_cfg is not None and noise_cfg.noise_on_replay and \
                isinstance(replay_buff, NoisyReplayBuffer):
            states, states_, actions, rewards, dones = replay_buff.sample_with_jitter(
                sample_batch_size, noise_cfg, bilateral=bilateral, curr_step=curr_step
            )
        else:
            states, states_, actions, rewards, dones = replay_buff.sample_buffer(sample_batch_size)

        states  = torch.tensor(states,  dtype=torch.float32).to('cuda')
        states_ = torch.tensor(states_, dtype=torch.float32).to('cuda')
        actions = torch.tensor(actions, dtype=torch.float32).to('cuda')
        rewards = torch.tensor(rewards, dtype=torch.float32).to('cuda').unsqueeze(-1)
        dones   = torch.tensor(dones,   dtype=torch.float32).to('cuda').unsqueeze(-1)

        sides,  act_list = parse_sides(states,  actions,  bilateral)
        sides_           = parse_sides(states_,           bilateral=bilateral)

        if noise_cfg is not None and noise_cfg.noise_on_replay:
            sides = [
                (_batch_signal_noise_emg(emg, noise_cfg),
                 _batch_signal_noise_kin(kin, noise_cfg))
                for emg, kin in sides
            ]
            sides_ = [
                (_batch_signal_noise_emg(emg, noise_cfg),
                 _batch_signal_noise_kin(kin, noise_cfg))
                for emg, kin in sides_
            ]

        # ── Critic Update ─────────────────────────────────────────────────────
        for p in QNetwork_base1.parameters(): p.requires_grad = True
        for p in QNetwork_base2.parameters(): p.requires_grad = True

        q1_loss = q2_loss = torch.tensor(0.0, device='cuda')
        cq1_list, cq2_list = [], []

        with torch.no_grad():
            ys = []
            for (emg_, kin_) in sides_:

                out_ = Policy(emg_.to(Policy.device), kin_.to(Policy.device), sample=True)
                next_act = torch.cat([
                    out_['pred_kin_state']  * Policy.kinematic_mask.unsqueeze(0),
                    out_['pred_impedance']  * Policy.kinematic_mask.unsqueeze(0)
                ], dim=-1)
                tq = torch.min(
                    QNetwork_target1(emg_, kin_, next_act),
                    QNetwork_target2(emg_, kin_, next_act)
                )
                log_pdf_ = out_['pred_kin_log_pdf'] + out_['pred_impedance_log_pdf']
                ys.append(rewards + gamma * (1 - dones) * (tq - Policy.log_alpha.exp().detach() * log_pdf_))

        for (emg, kin), act, y in zip(sides, act_list, ys):
            cq1 = QNetwork_base1(emg, kin, act)
            cq2 = QNetwork_base2(emg, kin, act)
            cq1_list.append(cq1)
            cq2_list.append(cq2)
            q1_loss += F.mse_loss(cq1, y)
            q2_loss += F.mse_loss(cq2, y)

        optimizer_and_scheduler['q1b']['optimizer'].zero_grad()
        optimizer_and_scheduler['q2b']['optimizer'].zero_grad()
        q1_loss.backward(retain_graph=True)
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(QNetwork_base1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(QNetwork_base2.parameters(), 1.0)
        optimizer_and_scheduler['q1b']['optimizer'].step()
        optimizer_and_scheduler['q2b']['optimizer'].step()

        # ── Actor Update ──────────────────────────────────────────────────────
        for p in QNetwork_base1.parameters(): p.requires_grad = False
        for p in QNetwork_base2.parameters(): p.requires_grad = False

        actor_loss = torch.tensor(0.0, device='cuda')
        log_pdfs_all = []

        for (emg, kin) in sides:

            out = Policy(emg.to(Policy.device), kin.to(Policy.device), sample=True)
            sampled_act = torch.cat([
                out['pred_kin_state']  * Policy.kinematic_mask.unsqueeze(0),
                out['pred_impedance']  * Policy.kinematic_mask.unsqueeze(0)
            ], dim=-1)
            q = torch.min(QNetwork_base1(emg, kin, sampled_act),
                          QNetwork_base2(emg, kin, sampled_act))
            log_pdf = out['pred_kin_log_pdf'] + out['pred_impedance_log_pdf']
            log_pdfs_all.append(log_pdf)
            actor_loss += (Policy.log_alpha.exp().detach() * log_pdf - q).mean()

        optimizer_and_scheduler['policy']['optimizer'].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(Policy.parameters(), 1.0)
        optimizer_and_scheduler['policy']['optimizer'].step()

        for p in QNetwork_base1.parameters(): p.requires_grad = True
        for p in QNetwork_base2.parameters(): p.requires_grad = True

        # ── Alpha Update ──────────────────────────────────────────────────────
        avg_log_pdf = torch.stack(log_pdfs_all).mean(0).detach()
        alpha_loss = -(Policy.log_alpha * (avg_log_pdf + Policy.target_entropy)).mean()

        optimizer_and_scheduler['policy_log_alpha']['optimizer'].zero_grad()
        alpha_loss.backward()
        optimizer_and_scheduler['policy_log_alpha']['optimizer'].step()
        
        with torch.no_grad():
            Policy.log_alpha.clamp_(min=-10.0, max=2.0)
        # ── Soft Updates + Logging ────────────────────────────────────────────
        soft_update(QNetwork_base1, QNetwork_target1, tau)
        soft_update(QNetwork_base2, QNetwork_target2, tau)

        training_losses['alpha_val'].append(Policy.log_alpha.exp().detach().item())
        training_losses['policy_entropy'].append(-avg_log_pdf.mean().item())
        training_losses['actor_loss'].append(actor_loss.item())
        training_losses['q1_loss'].append(q1_loss.item())
        training_losses['q2_loss'].append(q2_loss.item())
        training_losses['alpha_loss'].append(alpha_loss.item())
        training_losses['q1_mean'].append(torch.stack(cq1_list).mean().item())
        training_losses['q2_mean'].append(torch.stack(cq2_list).mean().item())

        training_iterations += 1

    print("\n--- Training Phase Complete ---")
    print(f"  Actor Loss: {np.mean(training_losses['actor_loss']):.4f}")
    print(f"  Q1 Loss:    {np.mean(training_losses['q1_loss']):.4f}")
    print(f"  Q2 Loss:    {np.mean(training_losses['q2_loss']):.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# Amputation Config  (UNCHANGED)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AmputationConfig:
    name: str
    env_id: str
    action_indices: List[int]
    obs_rearrange_tag: str
    concat_tag: str
    bilateral: bool
    agent_obs_slices: Tuple
    n_prosthetic_joints: int
    emg_side_slices: Tuple
    tibial: bool = False
    subfolder: str = ''


AMPUTATION_CONFIGS = {
    'transtibial_left': AmputationConfig(
        name='transtibial_left', env_id='sconewalk_h0111_osim-v1', subfolder='tb_left',
        action_indices=[0], obs_rearrange_tag='tibial_left', concat_tag='tibial_left',
        bilateral=False, agent_obs_slices=(slice(0, 45), slice(46, None)),
        n_prosthetic_joints=1, emg_side_slices=(slice(9, None),), tibial=True
    ),
    'transtibial_right': AmputationConfig(
        name='transtibial_right', env_id='sconewalk_h0222_osim-v1', subfolder='tb_right',
        action_indices=[0], obs_rearrange_tag='tibial_right', concat_tag='tibial_right',
        bilateral=False, agent_obs_slices=(slice(0, 45), slice(46, None)),
        n_prosthetic_joints=1, emg_side_slices=(slice(0, 9),), tibial=True
    ),
    'transfemoral_left': AmputationConfig(
        name='transfemoral_left', env_id='sconewalk_h0444_osim-v1', subfolder='tf_left',
        action_indices=[0, 1], obs_rearrange_tag='left', concat_tag='trans_left',
        bilateral=False, agent_obs_slices=(slice(0, 45), slice(47, None)),
        n_prosthetic_joints=2, emg_side_slices=(slice(9, None),)
    ),
    'transfemoral_right': AmputationConfig(
        name='transfemoral_right', env_id='sconewalk_h0555_osim-v1', subfolder='tf_right',
        action_indices=[0, 1], obs_rearrange_tag='right', concat_tag='trans_right',
        bilateral=False, agent_obs_slices=(slice(0, 45), slice(47, None)),
        n_prosthetic_joints=2, emg_side_slices=(slice(0, 9),)
    ),
    'transfemoral_both': AmputationConfig(
        name='transfemoral_both', env_id='sconewalk_h0333_osim-v1', subfolder='tf_dual',
        action_indices=[0, 1, 2, 3], obs_rearrange_tag='trans_both', concat_tag='trans_both',
        bilateral=True, agent_obs_slices=(slice(0, 45), slice(49, None)),
        n_prosthetic_joints=4, emg_side_slices=(slice(0, 9), slice(9, None))
    ),
    'transtibial_both': AmputationConfig(
        name='transtibial_both', env_id='sconewalk_h0888_osim-v1', subfolder='tb_dual',
        action_indices=[0, 1], obs_rearrange_tag='tibial_both', concat_tag='tibial_both',
        bilateral=True, agent_obs_slices=(slice(0, 45), slice(47, None)),
        n_prosthetic_joints=2, emg_side_slices=(slice(0, 9), slice(9, None)), tibial=True
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers  (UNCHANGED)
# ──────────────────────────────────────────────────────────────────────────────

def map_excitation_window(exc_window_9ch, tibial: bool = False):
    if exc_window_9ch.ndim == 1:
        exc_window_9ch = exc_window_9ch[:, np.newaxis]
    n = exc_window_9ch.shape[1]
    out = torch.zeros((13, n), dtype=torch.float32)
    out[0]  = torch.tensor(exc_window_9ch[5])
    out[1]  = torch.tensor(exc_window_9ch[4])
    out[2]  = torch.tensor(exc_window_9ch[5])
    out[4]  = torch.tensor(exc_window_9ch[1])
    out[5]  = torch.tensor(exc_window_9ch[0])
    out[12] = torch.tensor(exc_window_9ch[2])
    if tibial:
        out[3]  = torch.tensor(exc_window_9ch[8])
        out[6]  = torch.tensor(exc_window_9ch[6])
        out[8]  = torch.tensor(exc_window_9ch[7])
    return out


def get_sagittal(impedance_values):
    sagittal_impedances = np.zeros(9,)
    counter = 0
    for i in range(impedance_values.shape[-1]):
        if (i + 1) % 3 == 0:
            sagittal_impedances[counter] = impedance_values[i]
            counter += 1
    return sagittal_impedances


def init_loss_dict():
    return {
        'actor_loss': [], 'q1_loss': [], 'q2_loss': [],
        'alpha_loss': [], 'alpha_val': [], 'policy_entropy': [],
        'q1_mean': [], 'q2_mean': []
    }


def build_padded_emg_window(seed_emg, steps, device):
    pad_size = max(0, 100 - steps)
    window = F.pad(seed_emg[:, 0:steps], (pad_size, 0), mode='replicate')
    return window.to(device)


def rearrange_obs(obs: torch.Tensor, direction_of_control='left'):
    def expand_to_plane(joints):
        out = []
        for v in joints:
            out.extend([0.0, 0.0, v.item()])
        return out

    if direction_of_control.lower() == 'right':
        pos, vel, acc, leg = obs[3:6], obs[12:15], obs[21:24], obs[27:36]
    elif direction_of_control.lower() == 'left':
        pos, vel, acc, leg = obs[6:9], obs[15:18], obs[24:27], obs[36:45]
    elif 'tibial' in direction_of_control.lower() or 'trans' in direction_of_control.lower():
        pos_r, vel_r, acc_r, leg_r = obs[3:6],  obs[12:15], obs[21:24], obs[27:36]
        pos_l, vel_l, acc_l, leg_l = obs[6:9],  obs[15:18], obs[24:27], obs[36:45]

        dof_r = torch.tensor(expand_to_plane(pos_r) + expand_to_plane(vel_r) + expand_to_plane(acc_r), dtype=torch.float32)
        dof_l = torch.tensor(expand_to_plane(pos_l) + expand_to_plane(vel_l) + expand_to_plane(acc_l), dtype=torch.float32)

        tibial = 'tibial' in direction_of_control.lower()

        def make_emg(leg):
            below_knee = [leg[6].item(), 0.0, leg[7].item()] if tibial else [0.0, 0.0, 0.0]
            return torch.tensor([
                leg[5].item(), leg[4].item(), leg[5].item(),
                leg[8].item() if tibial else 0.0,
                leg[1].item(), leg[0].item(),
                below_knee[0], below_knee[1], below_knee[2],
                0.0, 0.0, 0.0, leg[2].item(),
            ], dtype=torch.float32)

        tag = direction_of_control.lower()
        if tag.endswith('_both'):
            return (dof_r, make_emg(leg_r)), (dof_l, make_emg(leg_l))
        elif tag.endswith('_left'):
            return dof_l, make_emg(leg_l)
        elif tag.endswith('_right'):
            return dof_r, make_emg(leg_r)

    dof_tensor = torch.tensor(expand_to_plane(pos) + expand_to_plane(vel) + expand_to_plane(acc), dtype=torch.float32)
    emg_tensor = torch.tensor([
        leg[5].item(), leg[4].item(), leg[5].item(), leg[8].item(),
        leg[1].item(), leg[0].item(), leg[6].item(), 0.0, leg[7].item(),
        0.0, 0.0, 0.0, leg[2].item(),
    ], dtype=torch.float32)
    return dof_tensor, emg_tensor


def concatenate_actions(pred_torque, muscle_action, direction):
    curr_ptr = 0
    if direction == 'right' or direction == 'left':
        full_action = np.zeros((21,))
        for i in range(pred_torque.shape[-1]):
            if (i + 1) % 3 == 0:
                full_action[curr_ptr] = pred_torque[:, i]
                curr_ptr += 1
        if direction == 'left':
            full_action[(curr_ptr + 9):] = muscle_action[9:]
        else:
            full_action[(curr_ptr):(curr_ptr + 9)] = muscle_action[:9]
        return full_action
    elif direction in ('trans_right', 'trans_left'):
        full_action = np.zeros((20,))
        for i in range(pred_torque.shape[-1]):
            if (i + 1) % 3 == 0 and i > 2:
                full_action[curr_ptr] = pred_torque[:, i]
                curr_ptr += 1
        if direction == 'trans_left':
            full_action[2:11] = muscle_action[9:]
            ZERO = [17, 18, 19]
        else:
            full_action[11:] = muscle_action[:9]
            ZERO = [8, 9, 10]
        full_action[ZERO] = 0.0
        return full_action
    elif direction.lower() in ('tibial_right', 'tibial_left'):
        full_action = np.zeros((19,))
        full_action[0] = pred_torque[:, -1]
        ZERO = [8, 9] if direction == 'tibial_left' else [17, 18]
        full_action[1:] = muscle_action
        full_action[ZERO] = 0.0
        return full_action
    elif direction.lower() == 'trans_both':
        full_action = np.zeros((22,))
        for j in range(2):
            for i in range(pred_torque[j].shape[-1]):
                if (i + 1) % 3 == 0 and i > 2:
                    full_action[curr_ptr] = pred_torque[j][:, i]
                    curr_ptr += 1
        full_action[4:] = muscle_action
        full_action[[10, 11, 12, 19, 20, 21]] = 0.0
        return full_action
    elif direction.lower() == 'tibial_both':
        full_action = np.zeros((20,))
        for j in range(2):
            for i in range(pred_torque[j].shape[-1]):
                if (i + 1) % 3 == 0 and i > 5:
                    full_action[curr_ptr] = pred_torque[j][:, i]
                    curr_ptr += 1
        full_action[2:] = muscle_action
        full_action[[8, 9, 10, 17, 18, 19]] = 0.0
        return full_action


def setup_env_and_agent(cfg: AmputationConfig, deprl_checkpoint: str):
    env = gym.make(cfg.env_id, clip_actions=True)
    env.action_indices = cfg.action_indices
    n_skip = len(cfg.action_indices)
    trimmed_action_space = gym.spaces.Box(
        low=env.action_space.low[:-n_skip],
        high=env.action_space.high[:-n_skip],
        dtype=env.action_space.dtype
    )
    trimmed_obs_space = gym.spaces.Box(
        low=env.observation_space.low[:-n_skip],
        high=env.observation_space.high[:-n_skip],
        dtype=env.observation_space.dtype
    )
    agent = deprl.custom_agents.dep_factory(3, deprl.custom_mpo_torch.TunedMPO())(
        replay=deprl.custom_replay_buffers.AdaptiveEnergyBuffer(
            return_steps=1, batch_size=256, steps_between_batches=1000,
            batch_iterations=30, steps_before_batches=2e5, num_acts=18
        )
    )
    agent.initialize(trimmed_obs_space, trimmed_action_space, seed=0)
    agent.load(deprl_checkpoint)
    print(f'[worker] agent loaded for {cfg.name}')
    return env, agent


# ──────────────────────────────────────────────────────────────────────────────
# EnvWorkerState
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EnvWorkerState:
    """
    All mutable per-environment state for one episode.
    Owned entirely by the worker subprocess — never shared.
    """
    emg_windows:       list   # List[(13,100) CPU tensor] — one per side
    emg_frame_buffers: list   # List[deque of (13,) CPU tensors]
    kin_buffers:       list   # List[deque of (27,) CPU tensors]
    sides:             list   # List[(dof_tensor, emg_tensor)] current obs per side
    steps:             int
    done:              bool
    episode_reward:    float


# ──────────────────────────────────────────────────────────────────────────────
# Worker subprocess
# ──────────────────────────────────────────────────────────────────────────────

def worker_loop(
    worker_id:        int,
    cfg:              AmputationConfig,
    deprl_checkpoint: str,
    noise_cfg:        Optional[NoiseConfig],
    max_env_steps:    int,
    conn,                           # mp.Connection — child end of Pipe
):
    """
    Subprocess target.  One process per parallel environment.

    Protocol with main process
    ──────────────────────────
    Each step the worker sends a 'step_obs' dict, then blocks on conn.recv()
    waiting for the action reply.  After env.step() it sends a 'transition'
    dict.  At episode end it sends an 'episode_end' dict (no reply expected)
    and immediately starts the next episode.

    Shutdown: main sends None → worker exits cleanly.

    Message schemas
    ───────────────
    Worker → Main  'step_obs':
        emg_fwd   : List[np.ndarray (13,100)]   noisy EMG per side
        kin_fwd   : List[np.ndarray (27,)]       noisy kin per side
        curr_states: List[np.ndarray]            clean state per side (for replay)
        muscle_action: np.ndarray                deprl agent output
        steps     : int                          local step counter

    Main → Worker  (action reply):
        full_action : np.ndarray                 concatenated env action
        action_bufs : List[np.ndarray]           [pred_kin|pred_imp] per side
        curr_step   : int                        global step from main (for noise)

    Worker → Main  'transition':
        curr_state  : np.ndarray                 flattened clean state
        next_state  : np.ndarray                 flattened clean next state
        action      : np.ndarray                 flattened action buf
        reward      : float
        done        : bool

    Worker → Main  'episode_end':
        episode_reward : float
        steps          : int
    """
    env, agent = setup_env_and_agent(cfg, deprl_checkpoint)

    _emg_jitter = noise_cfg.emg_jitter_max if noise_cfg else 0
    _kin_jitter = noise_cfg.kin_jitter_max if noise_cfg else 0
    cpu = torch.device('cpu')

    while True:                                         # ── episode loop ──

        obs = env.reset()
        env.unwrapped.store_next_episode()

        raw   = rearrange_obs(obs, cfg.obs_rearrange_tag)
        sides = list(raw) if cfg.bilateral else [raw]

        steps          = 1
        done           = False
        episode_reward = 0.0

        # ── init EMG windows ──────────────────────────────────────────────────
        seed_emgs   = [torch.zeros(13, 100) for _ in sides]
        for i, (_, emg) in enumerate(sides):
            seed_emgs[i][:, 0] = emg
        emg_windows = [build_padded_emg_window(s, steps, cpu) for s in seed_emgs]

        # ── init circular buffers ─────────────────────────────────────────────
        emg_frame_buffers = []
        for i, (_, emg) in enumerate(sides):
            buf = deque(maxlen=100 + max(_emg_jitter, 0))
            seed_frame = emg.cpu()
            for _ in range(buf.maxlen):
                buf.append(seed_frame)
            emg_frame_buffers.append(buf)

        kin_buffers = [
            deque([s[0].detach().cpu()], maxlen=max(_kin_jitter + 1, 1))
            for s in sides
        ]

        excitation_buffer = None

        while not done and steps < max_env_steps:       # ── step loop ──

            # ── update EMG windows (excitation buffer from previous step) ─────
            if steps > 1:
                emg_windows = [
                    map_excitation_window(excitation_buffer[sl], tibial=cfg.tibial)
                    for sl in cfg.emg_side_slices
                ]

            kinematics = [s[0].to(cpu) for s in sides]

            # ── push clean frames into circular buffers (always) ──────────────
            for i in range(len(sides)):
                emg_frame_buffers[i].append(emg_windows[i][:, -1].detach().cpu())
                kin_buffers[i].append(kinematics[i].detach().cpu())

            # ── build noisy obs for forward pass; clean obs for replay ─────────
            # curr_step will arrive in the action reply from main
            # On the very first step we don't have it yet — use steps as proxy.
            # After the first action reply curr_step is always available.
            if noise_cfg is not None and noise_cfg.noise_on_rollout:
                # Use local steps as a conservative proxy until main replies
                _eff_emg, _eff_kin = noise_cfg.effective_jitter(steps)
                emg_windows_fwd = [
                    _rollout_emg_noise(emg_frame_buffers[i], noise_cfg, _eff_emg, cpu)
                    for i in range(len(sides))
                ]
                kinematics_fwd = [
                    _rollout_kin_noise(kinematics[i], noise_cfg, kin_buffers[i], _eff_kin)
                    for i in range(len(sides))
                ]
            else:
                emg_windows_fwd = emg_windows
                kinematics_fwd  = kinematics

            # ── clean curr_state for replay ───────────────────────────────────
            curr_states = [
                np.concatenate([
                    w.detach().numpy().flatten(),
                    k.detach().numpy().flatten()
                ])
                for w, k in zip(emg_windows, kinematics)
            ]

            # ── deprl muscle action (CPU, local to worker) ────────────────────
            agent_obs     = np.concatenate([obs[cfg.agent_obs_slices[0]],
                                            obs[cfg.agent_obs_slices[1]]])
            muscle_action = agent.test_step(agent_obs, steps)

            # ── send obs to main; block for action ────────────────────────────
            conn.send({
                'type':         'step_obs',
                'emg_fwd':      [w.numpy() for w in emg_windows_fwd],
                'kin_fwd':      [k.numpy() for k in kinematics_fwd],
                'curr_states':  curr_states,
                'muscle_action': muscle_action,
                'steps':        steps,
            })

            reply = conn.recv()
            if reply is None:                           # shutdown signal
                env.close()
                return

            full_action  = reply['full_action']
            action_bufs  = reply['action_bufs']         # list of arrays per side
            # Update noise curriculum using main's global step counter
            if noise_cfg is not None and noise_cfg.noise_on_rollout:
                _eff_emg, _eff_kin = noise_cfg.effective_jitter(reply['curr_step'])
                # Recompute with corrected global step (minor quality improvement)
                emg_windows_fwd = [
                    _rollout_emg_noise(emg_frame_buffers[i], noise_cfg, _eff_emg, cpu)
                    for i in range(len(sides))
                ]
                kinematics_fwd = [
                    _rollout_kin_noise(kinematics[i], noise_cfg, kin_buffers[i], _eff_kin)
                    for i in range(len(sides))
                ]

            # ── step environment ──────────────────────────────────────────────
            obs, reward, done, excitation_buffer = env.step(full_action)

            # ── build next_state ──────────────────────────────────────────────
            raw_next = rearrange_obs(obs, cfg.obs_rearrange_tag)
            sides    = list(raw_next) if cfg.bilateral else [raw_next]

            next_emg_windows = [
                map_excitation_window(excitation_buffer[sl], tibial=cfg.tibial)
                for sl in cfg.emg_side_slices
            ]
            next_states = [
                np.concatenate([
                    w.detach().numpy().flatten(),
                    s[0].detach().numpy().flatten()
                ])
                for w, s in zip(next_emg_windows, sides)
            ]

            _r             = float(reward.item() if isinstance(reward, torch.Tensor) else reward)
            episode_reward += _r

            # ── send transition to main ───────────────────────────────────────
            conn.send({
                'type':       'transition',
                'curr_state': np.concatenate(curr_states),
                'next_state': np.concatenate(next_states),
                'action':     np.concatenate(action_bufs),
                'reward':     _r,
                'done':       bool(done),
            })

            emg_windows = next_emg_windows
            steps      += 1

        # ── episode over ──────────────────────────────────────────────────────
        conn.send({
            'type':           'episode_end',
            'episode_reward': episode_reward,
            'steps':          steps,
        })

        if not done:
            try:
                env.unwrapped.model.write_results(
                    env.unwrapped.output_dir,
                    f"{env.unwrapped.episode:05d}_{env.unwrapped.total_reward:.3f}"
                )
            except Exception:
                pass
        # Loop back to env.reset() automatically


# ──────────────────────────────────────────────────────────────────────────────
# Vectorized training loop  (REPLACES rl_train)
# ──────────────────────────────────────────────────────────────────────────────

def rl_train(
    cfg:                      AmputationConfig,
    prosthetic_controller,
    replay_buffer,
    Q1_b, Q2_b, Q1_m, Q2_m,
    args,
    critic_config,
    optimizers_and_schedulers,
    max_training_steps:       int = 100_000,
    max_env_steps:            int = 10_000,
    noise_cfg:                Optional[NoiseConfig] = None,
    save_interval:            int = 10,
    num_envs:                 int = 1,
    save_sac_interval: int = 10000
):
    """
    Vectorized SAC training loop over `num_envs` parallel sconegym environments.

    Architecture
    ────────────
    - num_envs worker subprocesses each own one env + deprl agent.
    - Each step: all workers send their (noisy) observations simultaneously.
    - Main stacks them into a single batch → one GPU forward pass → split
      actions back to each worker.
    - Workers step their envs and return (s, a, r, s', done) transitions.
    - Main writes transitions to the shared NoisyReplayBuffer (sequential,
      preserving per-worker episode ordering via worker_id tags).
    - SAC gradient steps = num_envs per collect cycle to keep the
      data/update ratio constant regardless of parallelism.
    """
    device = prosthetic_controller.device

    # ── spawn worker subprocesses ─────────────────────────────────────────────
    pipes     = []   # parent-end connections, indexed by worker_id
    processes = []

    for wid in range(num_envs):
        parent_conn, child_conn = mp.Pipe(duplex=True)
        p = mp.Process(
            target=worker_loop,
            args=(wid, cfg, args.deprl_checkpoint, noise_cfg, max_env_steps, child_conn),
            daemon=True,
        )
        p.start()
        child_conn.close()               # parent doesn't need child end
        pipes.append(parent_conn)
        processes.append(p)

    print(f'[main] spawned {num_envs} worker processes')

    viz             = TrainingVisualizer(save_dir=args.save_plot_dir, window=200,num_workers=args.num_envs)
    training_losses = init_loss_dict()
    curr_step       = 0
    last_save_step = 0
    worker_episode_nums: dict[int, int] = {i: 0 for i in range(args.num_envs)}

    # ── main collect / update loop ────────────────────────────────────────────
    while curr_step < max_training_steps:

        # ── Phase 1: collect step_obs from every worker ───────────────────────
        # Workers may send 'step_obs' or 'episode_end'.
        # Accumulate episode_end messages and re-poll those workers for their
        # next episode's first step_obs.

        step_obs_by_wid = {}    # wid → step_obs dict (only 'step_obs' msgs)

        for wid, conn in enumerate(pipes):
            msg = conn.recv()

            while msg['type'] == 'episode_end':
                # Log the completed episode
                print(
                    f'[worker {wid}] episode {worker_episode_nums[wid]} | '
                    f'steps {msg["steps"]} | '
                    f'reward {msg["episode_reward"]:.3f} | '
                    f'avg {msg["episode_reward"] / max(msg["steps"], 1):.4f}'
                )

                worker_episode_nums[wid] += 1
                viz.log_episode(wid)

                if worker_episode_nums[wid] % save_interval == 0:
                    viz.save(wid,tag=f'episode{worker_episode_nums[wid]}_worker{wid}_end')

                # Worker auto-resets; grab the first step_obs of the new episode
                msg = conn.recv()

            # msg is now guaranteed to be 'step_obs'
            step_obs_by_wid[wid] = msg

        # ── Phase 2: batched GPU forward pass ─────────────────────────────────
        # Stack all workers' observations into a single (N*n_sides, 13, 100)
        # and (N*n_sides, 27) batch, run one forward pass, then split by worker.

        all_emg_np = []   # accumulate as numpy, move to GPU once
        all_kin_np = []
        worker_order    = list(step_obs_by_wid.keys())
        sides_per_worker = []

        for wid in worker_order:
            obs_msg = step_obs_by_wid[wid]
            n_sides = len(obs_msg['emg_fwd'])
            sides_per_worker.append(n_sides)
            for emg_np, kin_np in zip(obs_msg['emg_fwd'], obs_msg['kin_fwd']):
                all_emg_np.append(emg_np)
                all_kin_np.append(kin_np)

        emg_batch = torch.tensor(np.stack(all_emg_np), dtype=torch.float32).to(device)  # (total_sides, 13, 100)
        kin_batch = torch.tensor(np.stack(all_kin_np), dtype=torch.float32).to(device)  # (total_sides, 27)

        with torch.no_grad():

            batch_out = prosthetic_controller(emg_batch, kin_batch, sample=True)

        # ── Phase 3: split results and send actions back to each worker ────────
        slot = 0
        for wid, n_sides in zip(worker_order, sides_per_worker):
            obs_msg      = step_obs_by_wid[wid]
            muscle_action = obs_msg['muscle_action']

            torques     = []
            action_bufs = []

            for s in range(n_sides):
                idx      = slot + s
                kin_s    = kin_batch[idx]                               # (27,) on device
                pred_kin = batch_out['pred_kin_state'][idx : idx + 1]  # (1, 27)
                pred_imp = batch_out['pred_impedance'][idx : idx + 1]  # (1, 27)

                torque = compute_impedance_torque(
                    input_kin_state = kin_s.unsqueeze(0),
                    pred_kin_state  = pred_kin,
                    pred_impedance  = pred_imp,
                )
                torques.append(torque)
                action_bufs.append(
                    np.concatenate([
                        pred_kin.detach().cpu().numpy().flatten(),
                        pred_imp.detach().cpu().numpy().flatten(),
                    ])
                )

            torque_arg  = torques if cfg.bilateral else torques[0]
            full_action = concatenate_actions(torque_arg, muscle_action, cfg.concat_tag)

            pipes[wid].send({
                'full_action': full_action,
                'action_bufs': action_bufs,
                'curr_step':   curr_step,
            })

            slot += n_sides

        # ── Phase 4: collect transitions from every worker ────────────────────
        # Workers have now called env.step().  Drain their 'transition' dicts
        # and write sequentially to the shared replay buffer.

        n_collected = 0
        for wid in worker_order:
            msg = pipes[wid].recv()
            assert msg['type'] == 'transition', \
                f'expected transition from worker {wid}, got {msg["type"]}'

            replay_buffer.store_transition(
                state     = msg['curr_state'],
                action    = msg['action'],
                reward    = msg['reward'],
                state_    = msg['next_state'],
                done      = msg['done'],
                worker_id = wid,
            )

            viz.log_step(msg['reward'],wid)
            viz.log_losses(training_losses)
            n_collected += 1

        curr_step += n_collected

        # ── Phase 5: SAC gradient update ──────────────────────────────────────
        # Run one gradient step per collected transition so the data/update
        # ratio stays identical to the single-env baseline.
        if replay_buffer.size >= args.min_replay_size:
            train_sac(
                optimizer_and_scheduler = optimizers_and_schedulers,
                policy_args      = args,
                critic_args      = critic_config,
                Policy           = prosthetic_controller,
                QNetwork_base1   = Q1_b,
                QNetwork_base2   = Q2_b,
                QNetwork_target1 = Q1_m,
                QNetwork_target2 = Q2_m,
                replay_buff      = replay_buffer,
                training_epochs  = args.num_envs,     # N steps → N gradient updates
                training_losses  = training_losses,
                bilateral        = cfg.bilateral,
                sample_batch_size = args.batch_size,
                noise_cfg        = noise_cfg,
                curr_step        = curr_step,
            )

            if (curr_step // save_sac_interval) > (last_save_step // save_sac_interval):
                    
                Q1_b.save_checkpoint('Q1B',optimizer_and_scheduler['q1b']['optimizer'],optimizer_and_scheduler['q1b']['scheduler'])
                Q2_b.save_checkpoint('Q2B',optimizer_and_scheduler['q2b']['optimizer'],optimizer_and_scheduler['q2b']['scheduler'])
                Q1_m.save_checkpoint('Q1T',Q_config,optimizer_and_scheduler['q1t']['optimizer'],optimizer_and_scheduler['q1t']['scheduler'])
                Q2_m.save_checkpoint('Q2T',Q_config,optimizer_and_scheduler['q2t']['optimizer'],optimizer_and_scheduler['q2t']['scheduler'])
                replay_buff.save()

                prosthetic_controller.save_checkpoint(
                    optimizer_and_scheduler['policy']['optimizer'],optimizer_and_scheduler['policy']['scheduler'],
                    args,best_val_loss,curr_step // save_sac_interval,
                    optimizer_and_scheduler['policy_log_alpha']['optimizer'],alpha_scheduler=optimizer_and_scheduler['policy_log_alpha']['scheduler'])

                last_save_step = curr_step


    # ── shutdown ──────────────────────────────────────────────────────────────
    for conn in pipes:
        conn.send(None)                  # poison pill
    for p in processes:
        p.join(timeout=10)

    viz.close()
    print('training complete.')


def build_q_network(config, device, lr, epochs):
    net = QNetwork(**config).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 100)
    return net, opt, sch


def build_networks_and_optimizers(args, prosthetic_controller, Q_config, from_checkpoint=False):
    lr, epochs = args.lr, args.epochs
    device     = args.device
    n_sides    = 2 if args.bilateral else 1

    save_dir = os.path.join(args.checkpoint_dir, args.amputation_type)
    os.makedirs(save_dir, exist_ok=True)

    if from_checkpoint:
        print('loading from checkpoint')
        policy_ckpt = torch.load(os.path.join(save_dir, 'best_RL_transformer_model.pth'))
        mc = policy_ckpt['model_config']

        policy = EMGTransformer(
            emg_channels=13, emg_window_size=100, kin_state_dim=27,
            d_model=mc['d_model'], nhead=mc['nhead'],
            num_encoder_layers=mc['num_layers'], num_decoder_layers=mc['num_layers'],
            predict_impedance=True,
            emg_mask=args.emg_mask, kinematic_mask=args.kinematic_mask
        ).to(device)
        policy.load_state_dict(policy_ckpt['model_state_dict'])
        policy.log_alpha = policy_ckpt['log_alpha'].to(device).requires_grad_(True)

        p_opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
        p_opt.load_state_dict(policy_ckpt['policy_optimizer_state_dict'])
        p_sch = torch.optim.lr_scheduler.CosineAnnealingLR(p_opt, T_max=epochs, eta_min=lr / 100)
        p_sch.load_state_dict(policy_ckpt['policy_scheduler_state_dict'])

        a_opt = torch.optim.AdamW([policy.log_alpha], lr=lr, weight_decay=0.01, eps=1e-8)
        a_opt.load_state_dict(policy_ckpt['log_alpha_optimizer'])
        a_sch = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=epochs, eta_min=lr / 100)
        a_sch.load_state_dict(policy_ckpt['log_alpha_scheduler'])
        print('policy + log_alpha loaded')

        q_nets, q_opts, q_schs = [], [], []
        for tag in ['Q1B', 'Q2B', 'Q1T', 'Q2T']:
            ckpt = torch.load(os.path.join(save_dir, tag))
            net  = QNetwork(**ckpt['config']).to(device)
            net.load_checkpoint(os.path.join(save_dir, tag))
            opt  = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
            opt.load_state_dict(ckpt['optimizer'])
            sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 100)
            sch.load_state_dict(ckpt['scheduler'])
            q_nets.append(net); q_opts.append(opt); q_schs.append(sch)
        print(f'loaded {len(q_nets)} Q networks')

        replay_buffer = NoisyReplayBuffer(
            max_size    = int(1e7),
            input_shape = int(n_sides * (13 * 100 + 27)),
            n_actions   = int(n_sides * 54),
            checkpoint_dir = save_dir,
            save_name   = args.amputation_type,
            num_workers = args.num_envs
        )
        replay_buffer.load()
        print(f'loaded replay buffer from: {save_dir}')
        print('finished loading checkpoint')

    else:
        policy = prosthetic_controller

        q_nets, q_opts, q_schs = [], [], []
        for _ in range(4):
            net, opt, sch = build_q_network(Q_config, device, lr, epochs)
            q_nets.append(net); q_opts.append(opt); q_schs.append(sch)

        p_opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
        p_sch = torch.optim.lr_scheduler.CosineAnnealingLR(p_opt, T_max=epochs, eta_min=lr / 100)
        a_opt = torch.optim.Adam([policy.log_alpha], lr=1e-3)
        a_sch = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=epochs, eta_min=lr / 100)

        replay_buffer = NoisyReplayBuffer(
            max_size    = int(1e5),
            input_shape = int(n_sides * (13 * 100 + 27)),
            n_actions   = int(n_sides * 54),
            checkpoint_dir = save_dir,
            save_name   = args.amputation_type,
            num_workers = args.num_envs
        )

    policy.checkpoint_dir = save_dir
    for net in q_nets:
        net.checkpoint_dir = save_dir
    replay_buffer.checkpoint_dir = save_dir

    optimizers_and_schedulers = {
        'policy':           {'optimizer': p_opt,        'scheduler': p_sch},
        'policy_log_alpha': {'optimizer': a_opt,        'scheduler': a_sch},
        'q1b':              {'optimizer': q_opts[0],    'scheduler': q_schs[0]},
        'q2b':              {'optimizer': q_opts[1],    'scheduler': q_schs[1]},
        'q1t':              {'optimizer': q_opts[2],    'scheduler': q_schs[2]},
        'q2t':              {'optimizer': q_opts[3],    'scheduler': q_schs[3]},
    }

    print('models initialized')

    return policy, q_nets, replay_buffer, optimizers_and_schedulers


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print('running')
    parser = argparse.ArgumentParser()

    # ── data / model paths ────────────────────────────────────────────────────
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--checkpoint_dir',  type=str,
                        default='/gpfs/data/s001/vwulfek1/software/models/SAC')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--replay_buffer_tag', type=str, default=None)
    parser.add_argument('--deprl_checkpoint',  type=str,
                        default='/gpfs/data/s001/vwulfek1/software/models/step_13500000')
    parser.add_argument('--save_model_interval', type=int,
                        default=1e5)
    parser.add_argument('--save_plot_interval', type=int,
                        default=10)
    parser.add_argument('--save_plot_dir', type=str,
                        default='/gpfs/data/s001/vwulfek1/software/plots/SAC')

    # ── environment ───────────────────────────────────────────────────────────
    parser.add_argument('--amputation_type', type=str,
                        choices=list(AMPUTATION_CONFIGS.keys()),
                        default='transfemoral_left')
    parser.add_argument('--num_envs', type=int, default=4,  
                        help='Number of parallel environment workers')

    # ── training hyperparams ──────────────────────────────────────────────────
    parser.add_argument('--batch_size',          type=int,   default=256)
    parser.add_argument('--epochs',              type=int,   default=100)
    parser.add_argument('--lr',                  type=float, default=1e-4)
    parser.add_argument('--max_training_steps',  type=int,   default=100000)
    parser.add_argument('--max_env_steps',       type=int,   default=20000)
    parser.add_argument('--min_replay_size',     type=int,   default=2048)

    # ── model architecture ────────────────────────────────────────────────────
    parser.add_argument('--d_model',    type=int, default=512)
    parser.add_argument('--nhead',      type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--device',     type=str, default='cuda')

    # ── sim-to-real noise ─────────────────────────────────────────────────────
    parser.add_argument('--emg_noise_std_max',  type=float, default=1.0)
    parser.add_argument('--emg_noise_mean_max', type=float, default=0.0)
    parser.add_argument('--kin_noise_std_max',  type=float, default=1.0)
    parser.add_argument('--kin_noise_mean_max', type=float, default=0.0)
    parser.add_argument('--emg_jitter_max',     type=int,   default=200)
    parser.add_argument('--kin_jitter_max',     type=int,   default=5)
    parser.add_argument('--jitter_warmup_steps',type=int,   default=0)
    parser.add_argument('--noise_on_rollout',   action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--noise_on_replay',    action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    cfg = AMPUTATION_CONFIGS[args.amputation_type]
    args.bilateral    = cfg.bilateral
    args.cfg_subfolder = cfg.subfolder

    emg_masks = {
        'transtibial_left':   np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1]),
        'transtibial_right':  np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1]),
        'transfemoral_left':  np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
        'transfemoral_right': np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
        'transfemoral_both':  np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
        'transtibial_both':   np.array([1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1]),
    }
    kin_masks = {
        'transtibial_left':   np.array([[0,0,0],[0,0,0],[0,0,1]]),
        'transtibial_right':  np.array([[0,0,0],[0,0,0],[0,0,1]]),
        'transfemoral_left':  np.array([[0,0,0],[0,0,1],[0,0,1]]),
        'transfemoral_right': np.array([[0,0,0],[0,0,1],[0,0,1]]),
        'transfemoral_both':  np.array([[0,0,1],[0,0,1],[0,0,1]]),
        'transtibial_both':   np.array([[0,0,1],[0,0,0],[0,0,1]]),
    }

    args.emg_mask      = emg_masks[args.amputation_type]
    args.kinematic_mask = kin_masks[args.amputation_type]

    # ── build controller ──────────────────────────────────────────────────────
    print('checking for checkpoints')
    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location=args.device)
        mc   = ckpt.get('model_config', {})
        d_model    = mc.get('d_model',    args.d_model)
        nhead      = mc.get('nhead',      args.nhead)
        num_layers = mc.get('num_layers', args.num_layers)
        if mc:
            print(f'checkpoint model_config: d_model={d_model}, nhead={nhead}, num_layers={num_layers}')
        else:
            print('WARNING: checkpoint has no model_config — falling back to CLI args.')
    else:
        ckpt       = None
        d_model    = args.d_model
        nhead      = args.nhead
        num_layers = args.num_layers

    prosthetic_controller = EMGTransformer(
        emg_channels=13, emg_window_size=100, kin_state_dim=27,
        d_model=d_model, nhead=nhead,
        num_encoder_layers=num_layers, num_decoder_layers=num_layers,
        predict_impedance=True,
        emg_mask=args.emg_mask, kinematic_mask=args.kinematic_mask
    ).to(args.device)

    if ckpt is not None:
        missing, unexpected = prosthetic_controller.load_state_dict(
            ckpt['model_state_dict'], strict=False
        )
        new_heads    = [k for k in missing if 'log_std' in k]
        real_missing = [k for k in missing if 'log_std' not in k]
        if new_heads:
            print(f'log_std heads not in checkpoint ({len(new_heads)} keys) — initialising fresh.')
        if real_missing:
            print(f'WARNING: unexpected missing keys: {real_missing}')
        if unexpected:
            print(f'WARNING: {len(unexpected)} unexpected keys — first few: {unexpected[:3]}')
        print(f'loaded controller from {args.checkpoint_path}')

    Q_config = {
        'h_dim': 512, 'num_bins': 1,
        'emg_channels': 13, 'emg_window_size': 100,
        'kin_state_dim': 27, 'action_dim': 54,
        'd_model': 512, 'nhead': 2,
        'num_encoder_layers': 2, 'num_decoder_layers': 2,
        'dim_feedforward': 1024, 'dropout': 0.1
    }

    policy, q_nets, replay_buffer, optimizers_and_schedulers = build_networks_and_optimizers(
        args, prosthetic_controller, Q_config, from_checkpoint=args.resume
    )

    noise_cfg = NoiseConfig(
        emg_noise_std_max   = args.emg_noise_std_max,
        emg_noise_mean_max  = args.emg_noise_mean_max,
        kin_noise_std_max   = args.kin_noise_std_max,
        kin_noise_mean_max  = args.kin_noise_mean_max,
        emg_jitter_max      = args.emg_jitter_max,
        kin_jitter_max      = args.kin_jitter_max,
        jitter_warmup_steps = args.jitter_warmup_steps,
        noise_on_rollout    = args.noise_on_rollout,
        noise_on_replay     = args.noise_on_replay,
    )
    print(f'noise config: {noise_cfg}')
    print(f'num_envs: {args.num_envs}')

    # NoisyReplayBuffer is always used in the vec version (worker_id required)
    if not isinstance(replay_buffer, NoisyReplayBuffer):
        replay_buffer.__class__ = NoisyReplayBuffer
        replay_buffer.worker_id_memory = np.full(replay_buffer.mem_size, -1, dtype=np.int32)
        print('replay buffer upgraded to NoisyReplayBuffer')

    rl_train(
        cfg                      = cfg,
        prosthetic_controller    = policy,
        replay_buffer            = replay_buffer,
        Q1_b                     = q_nets[0],
        Q2_b                     = q_nets[1],
        Q1_m                     = q_nets[2],
        Q2_m                     = q_nets[3],
        args                     = args,
        critic_config            = Q_config,
        optimizers_and_schedulers = optimizers_and_schedulers,
        max_training_steps       = args.max_training_steps,
        max_env_steps            = args.max_env_steps,
        noise_cfg                = noise_cfg,
        save_interval            =args.save_plot_interval,
        num_envs                 = args.num_envs,
        save_sac_interval        = args.save_model_interval,
    )


if __name__ == '__main__':
    mp.set_start_method('spawn')      # required for CUDA + multiprocessing
    main()