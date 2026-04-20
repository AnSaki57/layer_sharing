"""FedAgent: Federated Learning with DRL-based Client Selection and Knowledge Distillation.

Implementation of the FedAgent framework from:
    Sun, Song, Tu, Liu — "FedAgent: Federated learning with deep reinforcement learning
    based client selection and knowledge distillation" (Expert Systems with Applications, 2025).

Architecture
------------
Server-client (star) topology:

  * Server  : holds the global model w_G, a Dueling Double-DQN agent (main + target
              networks, replay buffer), the latest known per-client weights and
              metadata, and the global F1 history. Each round it
                (1) builds the state s^t from per-client features (PCA-compressed
                    weights + n_i, m_i, F_i, B_i + global F1),
                (2) selects K clients via top-K of Q(s, .) (epsilon-greedy),
                (3) ships the global model to those K clients in parallel,
                (4) receives updated weights, performs FedAvg aggregation,
                (5) computes per-client rewards (Eq. 15), stores the transition
                    in the replay buffer, and runs a Dueling Double-DQN update
                    (Algorithm 1).
  * Clients : on each "train" request they reset to the received global model,
              run E local epochs of SGD with the FedAgent KD loss
                L = L_CE + mu * L_KL (with the true-class column masked out,
                Eqs. 21-24)
              and return the updated weights + metadata + measured local-training
              latency to the server.

This implementation deliberately removes the fault-tolerance machinery from
``layer_sharing.py`` (no fault injection, no crash detection, no
random-layer-stacking, no convergence-based termination): every client is
assumed reachable and reliable, and the federation runs for a fixed number of
rounds T.

Input file (``fedagent_input.txt`` by default)
----------------------------------------------
Set ``FEDAGENT_INPUT=/path/to/file.txt`` before launching to use another file
(e.g. ``fedagent_input_localhost.txt`` for an all-``127.0.0.1`` single-laptop run).
Or use ``./run_fedagent_local.sh`` which exports that for you.

::

    N M K
    SERVER_IP
    CURRENT_MACHINE_IP
    client_ip_1,client_ip_2,...,client_ip_N

where N = number of clients, M = number of machines, K = number of clients
selected per round. The process whose ``CURRENT_MACHINE_IP`` equals ``SERVER_IP``
runs the orchestrator; others only host listeners for matching client IDs.

For **one laptop**, set ``SERVER_IP``, ``CURRENT_MACHINE_IP``, and every entry in
the client list to ``127.0.0.1`` so a single process runs the server and all
client threads.
"""

import argparse
import logging
import os
import pickle
import random
import socket
import struct
import sys
import threading
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# =============================================================================
# Reproducibility
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# Constants / hyperparameters (overridable via env vars or CLI)
# =============================================================================

# --- Federated learning (FL_* env vars are shared with layer_sharing.py) ---
BATCH_SIZE = int(os.environ.get("FL_BATCH_SIZE", "64"))
EPOCHS_PER_ROUND = int(os.environ.get("FL_EPOCHS_PER_ROUND", "5"))   # E (paper default)
TOTAL_ROUNDS = int(os.environ.get("FL_ROUNDS", os.environ.get("FED_ROUNDS", "100")))  # T
FIXED_DATA_PER_CLIENT = 5000
DIRICHLET_ALPHA = float(os.environ.get("FL_DIRICHLET_ALPHA", "0.5"))
DEVICE = torch.device("cpu")
LOCAL_LR = 0.01

# --- Knowledge distillation (Eq. 21) ---
KD_TEMPERATURE = 1.0            # tau
KD_MU = 1.5                     # mu, weight on the KL term

# --- Deep reinforcement learning (Algorithm 1) ---
DRL_GAMMA = 0.95                # discount factor
DRL_LR = 0.01                   # delta
DRL_BATCH_SIZE = 32             # N_b
DRL_BUFFER_SIZE = 1000          # N_e
DRL_TARGET_UPDATE_EVERY = 10    # N_t (rounds)
DRL_UPDATES_PER_ROUND = 4       # b in Algorithm 1
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_ROUNDS = 50
LAMBDA_REWARD = 0.5             # lambda in Eq. 15

# --- Networking ---
CONNECT_TIMEOUT = float(os.environ.get("FED_CONNECT_TIMEOUT", "120"))
SERVER_BACKLOG = int(os.environ.get("FED_SERVER_BACKLOG", "128"))
CLIENT_BASE_PORT = 8650
SERVER_PORT = 8600              # not strictly used: server only initiates, never listens

# Per-client static metadata (m_i, F_i, B_i in the paper). In a simulation we
# generate these once from a fixed-seed RNG so every machine sees the same
# values and the agent gets a meaningful (deterministic) signal.
_META_RNG = np.random.default_rng(SEED)


# =============================================================================
# Logging
# =============================================================================
logger = logging.getLogger("fedagent")
logger.setLevel(logging.INFO)
_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
logger.addHandler(_console_handler)


class _LoggerWriter:
    def __init__(self, lg, level):
        self.lg = lg
        self.level = level

    def write(self, message):
        if message.strip():
            self.lg.log(self.level, message.strip())

    def flush(self):
        pass


sys.stdout = _LoggerWriter(logger, logging.INFO)


# =============================================================================
# Input parsing
# =============================================================================
def parse_input_file(path: str = "fedagent_input.txt"):
    """Parse the FedAgent input file (see module docstring for format)."""
    try:
        with open(path, "r") as fh:
            lines = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
        if len(lines) < 4:
            raise ValueError("Input file must contain at least 4 non-empty lines.")
        n, m, k = map(int, lines[0].split())
        server_ip = lines[1]
        current_machine_ip = lines[2]
        client_ips = [ip.strip() for ip in lines[3].split(",")]
        if len(client_ips) != n:
            raise ValueError(
                f"Expected {n} client IPs on line 4, got {len(client_ips)}."
            )
        if not (1 <= k <= n):
            raise ValueError(f"K must satisfy 1 <= K <= N, got K={k}, N={n}.")
        return n, m, k, server_ip, current_machine_ip, client_ips
    except FileNotFoundError:
        print(f"Input file '{path}' not found.")
    except ValueError as ve:
        print(f"Error parsing input file: {ve}")
    except Exception as e:
        print(f"Unexpected error parsing input file: {e}")
    return None, None, None, None, None, None


_INPUT_PATH = os.environ.get("FEDAGENT_INPUT", "fedagent_input.txt")
NUM_CLIENTS, NUM_MACHINES, K_PER_ROUND, SERVER_IP, CURRENT_MACHINE_IP, ips = (
    parse_input_file(_INPUT_PATH)
)
if NUM_CLIENTS is None:
    print("Failed to parse the input file. Exiting.")
    sys.exit(1)


# =============================================================================
# Networking utilities (length-prefixed pickle messages)
# =============================================================================
def send_message(conn, message):
    data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack("!I", len(data)) + data)


def _recv_exact(conn, nbytes: int):
    data = b""
    while len(data) < nbytes:
        chunk = conn.recv(nbytes - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def receive_message(conn):
    hdr = _recv_exact(conn, 4)
    if not hdr:
        return None
    msg_len = struct.unpack("!I", hdr)[0]
    body = _recv_exact(conn, msg_len)
    if body is None:
        return None
    return pickle.loads(body)


def _new_socket():
    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sk.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        sk.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    except Exception:
        pass
    return sk


# =============================================================================
# Dataset (CIFAR-10 with Dirichlet non-IID partition)
# =============================================================================
_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=_transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=_transform)


def create_dirichlet_non_iid_splits_fixed(dataset, num_clients, alpha=0.5, fixed_per_client=5000):
    num_classes = 10
    class_indices = {c: np.where(np.array(dataset.targets) == c)[0] for c in range(num_classes)}
    client_indices = {i: [] for i in range(num_clients)}
    for c, idxs in class_indices.items():
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(idxs)).astype(int)
        start = 0
        for i, count in enumerate(proportions):
            client_indices[i].extend(idxs[start:start + count])
            start += count

    final = {}
    for cid, idxs in client_indices.items():
        np.random.shuffle(idxs)
        if len(idxs) > fixed_per_client:
            final[cid] = list(idxs[:fixed_per_client])
        else:
            final[cid] = list(np.random.choice(idxs, fixed_per_client, replace=True))
    return [torch.utils.data.Subset(dataset, final[i]) for i in range(num_clients)]


client_data = create_dirichlet_non_iid_splits_fixed(
    train_dataset, NUM_CLIENTS, alpha=DIRICHLET_ALPHA, fixed_per_client=FIXED_DATA_PER_CLIENT
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)


# =============================================================================
# Model definitions
# =============================================================================
class PaperCNN(nn.Module):
    """Two-layer 5x5 CNN + 2x2 max-pool + 2 FC layers with ReLU (paper default)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SimpleCNN(nn.Module):
    """The original small CNN from layer_sharing.py (kept for comparison)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def build_model(choice: int) -> nn.Module:
    if choice == 1:
        return PaperCNN()
    if choice == 2:
        return SimpleCNN()
    raise ValueError("Model choice must be 1 (PaperCNN) or 2 (SimpleCNN).")


MODEL_NAME_MAP = {1: "PaperCNN", 2: "SimpleCNN"}


# =============================================================================
# State-dict / parameter helpers
# =============================================================================
def state_dict_to_numpy(model: nn.Module):
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def numpy_to_state_dict_torch(state_np):
    return {k: torch.tensor(v).to(DEVICE) for k, v in state_np.items()}


def flatten_state_np(state_np):
    """Flatten only the floating-point tensors (skip integer BN counters)."""
    parts = []
    for k in sorted(state_np.keys()):
        arr = state_np[k]
        if np.issubdtype(arr.dtype, np.floating):
            parts.append(arr.ravel())
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def fedavg_aggregate(state_dicts, weights):
    """Weighted FedAvg over a list of numpy state dicts (Eq. 4)."""
    total = float(sum(weights))
    if total <= 0 or not state_dicts:
        return None
    norm_w = [w / total for w in weights]
    out = {}
    keys = state_dicts[0].keys()
    for key in keys:
        ref = state_dicts[0][key]
        if np.issubdtype(ref.dtype, np.floating):
            acc = np.zeros_like(ref, dtype=np.float64)
            for sd, w in zip(state_dicts, norm_w):
                acc += sd[key].astype(np.float64) * w
            out[key] = acc.astype(ref.dtype)
        else:
            # integer buffers (e.g. BN num_batches_tracked) -> just take the
            # first one; they are not part of FedAvg in any meaningful sense.
            out[key] = ref.copy()
    return out


def compute_accuracy_and_f1(model, loader, num_classes: int = 10):
    """Returns (accuracy %, macro-F1 in [0, 1])."""
    model.eval()
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            preds = model(data).argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            for c in range(num_classes):
                pc = preds == c
                tc = target == c
                tp[c] += int((pc & tc).sum().item())
                fp[c] += int((pc & ~tc).sum().item())
                fn[c] += int((~pc & tc).sum().item())
    acc = 100.0 * correct / max(1, total)
    f1_per_class = []
    for c in range(num_classes):
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1_per_class.append((2 * tp[c]) / denom if denom > 0 else 0.0)
    return acc, float(np.mean(f1_per_class))


# =============================================================================
# Knowledge distillation loss (Eqs. 21-24)
# =============================================================================
def kd_local_loss(student_logits, teacher_logits, targets, tau: float, mu: float):
    """L = CE(student, y) + mu * KL_excl_y( softmax(teacher/tau) || softmax(student/tau) ).

    The KL is computed with the true-class column masked out of the softmax (per
    Eqs. 22-24): the contribution of the true label is removed from both
    distributions before normalisation.
    """
    ce = F.cross_entropy(student_logits, targets)

    num_classes = student_logits.size(1)
    # Mask: True for "keep this column", False for "drop the true-class column".
    keep_mask = torch.ones_like(student_logits, dtype=torch.bool)
    keep_mask.scatter_(1, targets.view(-1, 1), False)
    neg_inf = torch.finfo(student_logits.dtype).min

    t_masked = teacher_logits.masked_fill(~keep_mask, neg_inf) / tau
    s_masked = student_logits.masked_fill(~keep_mask, neg_inf) / tau

    log_q_s = F.log_softmax(s_masked, dim=1)
    q_t = F.softmax(t_masked, dim=1)
    # KL(q_t || q_s) = sum q_t * (log q_t - log q_s); using KLDiv with
    # log_target=False expects input=log q_s, target=q_t, so the result is
    # exactly the KL divergence above (per-batch mean).
    kl = F.kl_div(log_q_s, q_t, reduction="batchmean")
    return ce + mu * kl, ce.detach().item(), kl.detach().item()


# =============================================================================
# Dueling DQN architecture (Section 4.2, Eqs. 19-20)
# =============================================================================
class DuelingQNet(nn.Module):
    """Dueling network: shared trunk -> (V scalar) + (A vector of length n_actions),

    Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)   (Eq. 20)
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(True),
            nn.Linear(hidden // 2, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(True),
            nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, s):
        h = self.trunk(s)
        v = self.value_head(h)                          # (B, 1)
        a = self.advantage_head(h)                      # (B, n_actions)
        return v + (a - a.mean(dim=1, keepdim=True))    # (B, n_actions)


# =============================================================================
# Replay buffer
# =============================================================================
class ReplayBuffer:
    """Stores (s, action_set, reward_total, s', done) transitions."""

    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, action_set, reward, s_next, done):
        self.buf.append((s.astype(np.float32), tuple(sorted(action_set)),
                         float(reward), s_next.astype(np.float32), bool(done)))

    def __len__(self):
        return len(self.buf)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.stack(s), dtype=torch.float32),
            list(a),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.stack(s2), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )


# =============================================================================
# Dueling Double DQN agent (Section 4.2, Algorithm 1)
# =============================================================================
class DuelingDoubleDQNAgent:
    """Selects K-of-N clients via top-K of per-client Q-values.

    The combinatorial Q-function is decomposed as
        Q(s, a) = sum_{i in a} Q_i(s)
    where Q_i(s) is the network's i-th output. Top-K under this decomposition
    is exact, and the standard Double-DQN target becomes
        y = sum_{i in selected} r_i  +  gamma * sum_{i in TopK(s'; W)} Q_i(s'; W^-).
    """

    def __init__(self, state_dim: int, n_clients: int, k_per_round: int):
        self.n_clients = n_clients
        self.k = k_per_round

        self.main_net = DuelingQNet(state_dim, n_clients).to(DEVICE)
        self.target_net = DuelingQNet(state_dim, n_clients).to(DEVICE)
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.main_net.parameters(), lr=DRL_LR)
        self.buffer = ReplayBuffer(DRL_BUFFER_SIZE)
        self.update_steps = 0

    # ------- selection -------
    def epsilon(self, round_idx: int) -> float:
        frac = min(1.0, round_idx / max(1, EPS_DECAY_ROUNDS))
        return EPS_START + frac * (EPS_END - EPS_START)

    @torch.no_grad()
    def select_clients(self, state_np, round_idx: int):
        eps = self.epsilon(round_idx)
        if random.random() < eps:
            return sorted(random.sample(range(self.n_clients), self.k))
        s = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        q = self.main_net(s).squeeze(0).cpu().numpy()
        return sorted(np.argsort(-q)[: self.k].tolist())

    # ------- training (Algorithm 1) -------
    def train_step(self) -> float:
        if len(self.buffer) < DRL_BATCH_SIZE:
            return 0.0
        s, actions, r, s2, d = self.buffer.sample(DRL_BATCH_SIZE)
        s, s2, r, d = s.to(DEVICE), s2.to(DEVICE), r.to(DEVICE), d.to(DEVICE)

        # Q_pred = sum over selected clients (per-sample) of Q_i(s; W).
        q_all = self.main_net(s)                              # (B, N)
        q_pred = torch.zeros(len(actions), device=DEVICE)
        for i, act in enumerate(actions):
            q_pred[i] = q_all[i, list(act)].sum()

        # Double-DQN target: pick top-K of s' with the main net, evaluate with target.
        with torch.no_grad():
            q_next_main = self.main_net(s2)                   # (B, N)
            topk_idx = torch.topk(q_next_main, self.k, dim=1).indices  # (B, K)
            q_next_target = self.target_net(s2)               # (B, N)
            q_next_sel = q_next_target.gather(1, topk_idx).sum(dim=1)
            y = r + DRL_GAMMA * (1.0 - d) * q_next_sel

        loss = F.mse_loss(q_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), 5.0)
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % DRL_TARGET_UPDATE_EVERY == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())
        return float(loss.item())


# =============================================================================
# State / reward helpers
# =============================================================================
def pca_compress(weight_matrix: np.ndarray, n_components: int) -> np.ndarray:
    """SVD-based PCA: (N, P) -> (N, n_components). Pads with zeros if needed."""
    if weight_matrix.size == 0:
        return np.zeros((0, n_components), dtype=np.float32)
    n_samples = weight_matrix.shape[0]
    centered = weight_matrix - weight_matrix.mean(axis=0, keepdims=True)
    try:
        U, S, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((n_samples, n_components), dtype=np.float32)
    feats = U * S                                       # (N, min(N, P))
    out = np.zeros((n_samples, n_components), dtype=np.float32)
    take = min(n_components, feats.shape[1])
    out[:, :take] = feats[:, :take].astype(np.float32)
    return out


def build_state(latest_weights_flat, client_meta, prev_f1, n_components):
    """Concatenate per-client features + global F1 into a flat state vector."""
    n = len(latest_weights_flat)
    weight_matrix = np.stack(latest_weights_flat, axis=0)
    pca_feats = pca_compress(weight_matrix, n_components)   # (N, n_components)
    rows = []
    for i in range(n):
        meta = client_meta[i]                                # (n_i, m_i, F_i, B_i)
        rows.append(np.concatenate([pca_feats[i], np.asarray(meta, dtype=np.float32)]))
    s = np.concatenate(rows + [np.asarray([prev_f1], dtype=np.float32)])
    return s.astype(np.float32)


def normalized_divergence(w_client_flat, w_global_flat) -> float:
    g = w_global_flat
    denom = float(np.linalg.norm(g)) + 1e-8
    return float(np.linalg.norm(w_client_flat - g)) / denom


def utility_zeta(div_norm, f1_now, f1_prev) -> float:
    if f1_now > f1_prev:
        return float(np.exp(-div_norm))
    return float(1.0 - np.exp(-div_norm))


# =============================================================================
# Client side: local listener + KD training
# =============================================================================
def _train_one_round(client_id, model_choice, global_state_np, epochs, train_loader):
    """Reset local model from global, do `epochs` of SGD with KD loss, return state + latency."""
    student = build_model(model_choice).to(DEVICE)
    student.load_state_dict(numpy_to_state_dict_torch(global_state_np), strict=True)

    teacher = build_model(model_choice).to(DEVICE)
    teacher.load_state_dict(numpy_to_state_dict_torch(global_state_np), strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = optim.SGD(student.parameters(), lr=LOCAL_LR, momentum=0.9)

    t0 = time.perf_counter()
    student.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            loss, _, _ = kd_local_loss(s_logits, t_logits, y,
                                       tau=KD_TEMPERATURE, mu=KD_MU)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    train_latency = time.perf_counter() - t0
    return state_dict_to_numpy(student), train_latency


def _train_initial_epoch(client_id, model_choice, train_loader):
    """Algorithm 2 line 4: each client downloads w0, runs 1 epoch of plain SGD."""
    model = build_model(model_choice).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LOCAL_LR, momentum=0.9)
    t0 = time.perf_counter()
    model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    return state_dict_to_numpy(model), time.perf_counter() - t0


def client_listener(client_id: int, model_choice: int, stop_event: threading.Event):
    """Listens on CLIENT_BASE_PORT + client_id for server-issued work."""
    train_loader = torch.utils.data.DataLoader(
        client_data[client_id], batch_size=BATCH_SIZE, shuffle=True
    )
    n_local = len(client_data[client_id])
    static_meta = (
        n_local,
        int(_META_RNG.integers(2, 9)),                  # m_i: # CPU cores
        float(_META_RNG.uniform(1.5, 4.0)),             # F_i: GHz
        float(_META_RNG.uniform(20.0, 200.0)),          # B_i: Mbps
    )

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", CLIENT_BASE_PORT + client_id))
    server.listen(SERVER_BACKLOG)
    server.settimeout(1.0)
    logger.info(f"[client {client_id}] listening on :{CLIENT_BASE_PORT + client_id}")

    try:
        while not stop_event.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            try:
                conn.settimeout(CONNECT_TIMEOUT)
                msg = receive_message(conn)
                if msg is None:
                    conn.close()
                    continue
                mtype = msg.get("type")

                if mtype == "init":
                    weights, latency = _train_initial_epoch(
                        client_id, model_choice, train_loader
                    )
                    send_message(conn, {
                        "type": "init_response",
                        "client_id": client_id,
                        "weights": weights,
                        "n_samples": static_meta[0],
                        "m_cores": static_meta[1],
                        "f_ghz": static_meta[2],
                        "b_mbps": static_meta[3],
                        "train_latency": latency,
                    })

                elif mtype == "train":
                    weights, latency = _train_one_round(
                        client_id, model_choice,
                        msg["global_weights"], msg.get("epochs", EPOCHS_PER_ROUND),
                        train_loader,
                    )
                    send_message(conn, {
                        "type": "train_response",
                        "client_id": client_id,
                        "round": msg.get("round"),
                        "weights": weights,
                        "n_samples": static_meta[0],
                        "m_cores": static_meta[1],
                        "f_ghz": static_meta[2],
                        "b_mbps": static_meta[3],
                        "train_latency": latency,
                    })

                elif mtype == "terminate":
                    logger.info(f"[client {client_id}] received terminate")
                    stop_event.set()

                else:
                    logger.warning(f"[client {client_id}] unknown message type {mtype}")
            except Exception as e:
                logger.exception(f"[client {client_id}] error handling connection: {e}")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
    finally:
        try:
            server.close()
        except Exception:
            pass
        logger.info(f"[client {client_id}] listener stopped")


# =============================================================================
# Server side
# =============================================================================
def _request(target_ip: str, target_port: int, payload: dict, timeout: float):
    """Synchronous request/response on a fresh TCP connection."""
    sock = _new_socket()
    sock.settimeout(timeout)
    try:
        sock.connect((target_ip, target_port))
        send_message(sock, payload)
        resp = receive_message(sock)
        return resp
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _request_with_retry(target_ip, target_port, payload, timeout, attempts=20, delay=2.0):
    last_exc = None
    for _ in range(attempts):
        try:
            return _request(target_ip, target_port, payload, timeout)
        except (ConnectionRefusedError, OSError) as e:
            last_exc = e
            time.sleep(delay)
    raise RuntimeError(
        f"Could not reach {target_ip}:{target_port} after {attempts} attempts: {last_exc}"
    )


def _ship_train_request(cid, payload):
    """Returns (cid, response, round_trip_latency) or (cid, None, latency on fail)."""
    target_ip = ips[cid]
    port = CLIENT_BASE_PORT + cid
    t0 = time.perf_counter()
    try:
        resp = _request(target_ip, port, payload, timeout=CONNECT_TIMEOUT)
    except Exception as e:
        logger.warning(f"[server] training request to client {cid} failed: {e}")
        resp = None
    return cid, resp, time.perf_counter() - t0


def server_main(model_choice: int, total_rounds: int, k_per_round: int,
                stop_event: threading.Event):
    """Implements Algorithm 2 (server side) with the DRL training loop."""
    logger.info(
        f"[server] starting | N={NUM_CLIENTS} K={k_per_round} T={total_rounds} "
        f"model={MODEL_NAME_MAP[model_choice]}"
    )

    server_t0 = time.time()

    # ---- Initialise the global model (same seed everywhere -> same w^0). ----
    global_model = build_model(model_choice).to(DEVICE)
    global_state_np = state_dict_to_numpy(global_model)

    # ---- (Algorithm 2 line 4) ask every client to do 1 epoch and report. ----
    logger.info("[server] initialisation phase: requesting 1-epoch warm-up from all clients")
    latest_weights = {}
    client_meta = {}
    init_threads = []
    init_results = {}
    init_lock = threading.Lock()

    def _do_init(cid):
        try:
            resp = _request_with_retry(
                ips[cid], CLIENT_BASE_PORT + cid,
                {"type": "init"}, timeout=CONNECT_TIMEOUT,
            )
            with init_lock:
                init_results[cid] = resp
        except Exception as e:
            logger.error(f"[server] init failed for client {cid}: {e}")
            with init_lock:
                init_results[cid] = None

    for cid in range(NUM_CLIENTS):
        th = threading.Thread(target=_do_init, args=(cid,), daemon=True)
        th.start()
        init_threads.append(th)
    for th in init_threads:
        th.join()

    for cid in range(NUM_CLIENTS):
        resp = init_results.get(cid)
        if resp is None or resp.get("type") != "init_response":
            logger.error(f"[server] missing init response from client {cid}; aborting")
            stop_event.set()
            return
        latest_weights[cid] = resp["weights"]
        client_meta[cid] = (
            float(resp["n_samples"]),
            float(resp["m_cores"]),
            float(resp["f_ghz"]),
            float(resp["b_mbps"]),
        )
    logger.info("[server] init phase complete; all clients reported")

    # Pick a random client's init weights as the initial global model -- this
    # is fine because the warm-up epoch is identical in spirit to the first FL
    # round, and it gives the agent a non-trivial starting point.
    global_state_np = fedavg_aggregate(
        [latest_weights[c] for c in range(NUM_CLIENTS)],
        [client_meta[c][0] for c in range(NUM_CLIENTS)],
    )
    global_model.load_state_dict(numpy_to_state_dict_torch(global_state_np), strict=True)
    init_acc, prev_f1 = compute_accuracy_and_f1(global_model, test_loader)
    logger.info(f"[server] post-init global model accuracy={init_acc:.2f}% F1={prev_f1:.4f}")

    # ---- Build the agent. ----
    state_dim = NUM_CLIENTS * (NUM_CLIENTS + 4) + 1     # PCA(N) + 4 meta + 1 global F1
    agent = DuelingDoubleDQNAgent(state_dim, NUM_CLIENTS, k_per_round)

    # Per-client running reward (for the EMA term in Eq. 15).
    prev_reward = np.zeros(NUM_CLIENTS, dtype=np.float32)

    def _current_state(prev_f1_val):
        flats = [flatten_state_np(latest_weights[c]) for c in range(NUM_CLIENTS)]
        return build_state(flats, [client_meta[c] for c in range(NUM_CLIENTS)],
                           prev_f1_val, n_components=NUM_CLIENTS)

    state = _current_state(prev_f1)

    history = []
    best_acc, best_round = 0.0, -1

    # ---- Main FL loop (Algorithm 2 lines 6-25) ----
    for t in range(total_rounds):
        if stop_event.is_set():
            break

        # (1) Agent picks K clients.
        selected = agent.select_clients(state, t)
        logger.info(f"[server] round {t}: eps={agent.epsilon(t):.3f}, selected={selected}")

        # (2) Ship the global model to selected clients in parallel.
        train_payload = {
            "type": "train",
            "round": t,
            "global_weights": global_state_np,
            "epochs": EPOCHS_PER_ROUND,
        }
        results = {}
        latencies = {}
        threads, lock = [], threading.Lock()

        def _worker(cid):
            cid2, resp, rtt = _ship_train_request(cid, train_payload)
            with lock:
                results[cid2] = resp
                latencies[cid2] = rtt

        for cid in selected:
            th = threading.Thread(target=_worker, args=(cid,), daemon=True)
            th.start()
            threads.append(th)
        for th in threads:
            th.join()

        # (3) Collect weights from successful responders. Anyone we failed to
        # reach is dropped from this round (no fault-tolerance machinery, but
        # we don't crash either -- we just skip them for aggregation).
        valid = [cid for cid in selected
                 if results.get(cid) is not None
                 and results[cid].get("type") == "train_response"]
        if not valid:
            logger.warning(f"[server] round {t}: no valid responses, skipping aggregation")
            continue

        for cid in valid:
            r = results[cid]
            latest_weights[cid] = r["weights"]
            client_meta[cid] = (float(r["n_samples"]), float(r["m_cores"]),
                                float(r["f_ghz"]), float(r["b_mbps"]))

        # (4) FedAvg aggregation over the selected (and successful) clients.
        agg_states = [latest_weights[cid] for cid in valid]
        agg_weights = [client_meta[cid][0] for cid in valid]
        new_global = fedavg_aggregate(agg_states, agg_weights)
        global_state_np = new_global
        global_model.load_state_dict(numpy_to_state_dict_torch(global_state_np), strict=True)

        acc, f1_now = compute_accuracy_and_f1(global_model, test_loader)
        if acc > best_acc:
            best_acc, best_round = acc, t

        # (5) Per-client rewards (Eq. 15) for the selected clients.
        max_lat = max(latencies[cid] for cid in valid) + 1e-8
        global_flat = flatten_state_np(global_state_np)
        round_rewards = np.zeros(NUM_CLIENTS, dtype=np.float32)
        for cid in valid:
            div = normalized_divergence(flatten_state_np(latest_weights[cid]), global_flat)
            zeta = utility_zeta(div, f1_now, prev_f1)
            l_norm = latencies[cid] / max_lat
            r_i = LAMBDA_REWARD * (zeta - l_norm) + (1.0 - LAMBDA_REWARD) * prev_reward[cid]
            round_rewards[cid] = r_i
            prev_reward[cid] = r_i
        total_reward = float(round_rewards.sum())

        # (6) Build s', store transition, train agent.
        next_state = _current_state(f1_now)
        done = (t == total_rounds - 1)
        agent.buffer.push(state, valid, total_reward, next_state, done)

        loss_vals = []
        for _ in range(DRL_UPDATES_PER_ROUND):
            loss_vals.append(agent.train_step())
        mean_loss = float(np.mean([v for v in loss_vals if v > 0])) if any(v > 0 for v in loss_vals) else 0.0

        history.append({
            "round": t,
            "selected": valid,
            "acc": acc,
            "f1": f1_now,
            "reward": total_reward,
            "drl_loss": mean_loss,
            "buffer": len(agent.buffer),
        })
        logger.info(
            f"[server] round {t}: acc={acc:.2f}% F1={f1_now:.4f} reward={total_reward:+.3f} "
            f"drl_loss={mean_loss:.4f} buf={len(agent.buffer)}/{DRL_BUFFER_SIZE}"
        )

        prev_f1 = f1_now
        state = next_state

    # ---- Done: tell every client to shut down. ----
    logger.info(f"[server] training complete; best_acc={best_acc:.2f}% (round {best_round})")
    for cid in range(NUM_CLIENTS):
        try:
            sock = _new_socket()
            sock.settimeout(CONNECT_TIMEOUT)
            sock.connect((ips[cid], CLIENT_BASE_PORT + cid))
            send_message(sock, {"type": "terminate"})
            sock.close()
        except Exception as e:
            logger.warning(f"[server] failed to terminate client {cid}: {e}")

    stop_event.set()

    total_time = time.time() - server_t0

    # ---- Summary ----
    print("\nFedAgent run complete")
    print(f"  Model            : {MODEL_NAME_MAP[model_choice]}")
    print(f"  N / K / T        : {NUM_CLIENTS} / {k_per_round} / {total_rounds}")
    print(f"  Best accuracy    : {best_acc:.2f}% (round {best_round})")
    if history:
        print(f"  Final accuracy   : {history[-1]['acc']:.2f}%")
        print(f"  Final F1         : {history[-1]['f1']:.4f}")
        print(f"  Last DRL loss    : {history[-1]['drl_loss']:.4f}")
    print(f"  Total time       : {total_time:.2f}s")

    # ---- Optional: dump a JSON results file for comparison harnesses. ----
    results_path = os.environ.get("FL_RESULTS_JSON")
    if results_path:
        import json
        summary = {
            "framework": "fedagent",
            "model": MODEL_NAME_MAP[model_choice],
            "n_clients": NUM_CLIENTS,
            "k_per_round": k_per_round,
            "rounds": total_rounds,
            "batch_size": BATCH_SIZE,
            "epochs_per_round": EPOCHS_PER_ROUND,
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "total_time_s": float(total_time),
            "final_acc": float(history[-1]["acc"]) if history else 0.0,
            "final_f1": float(history[-1]["f1"]) if history else 0.0,
            "best_acc": float(best_acc),
            "best_round": int(best_round),
            "history": history,
        }
        try:
            with open(results_path, "w") as fh:
                json.dump(summary, fh, indent=2)
            print(f"  Wrote results JSON to {results_path}")
        except Exception as e:
            print(f"  Failed to write results JSON to {results_path}: {e}")


# =============================================================================
# Main entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="FedAgent: DRL-based client selection + KD for federated learning."
    )
    parser.add_argument(
        "--model", type=int, default=1, choices=[1, 2],
        help="1 = PaperCNN (5x5 conv x2 + 2 FC, paper default), 2 = SimpleCNN (3x3 baseline)",
    )
    parser.add_argument("--rounds", type=int, default=TOTAL_ROUNDS,
                        help="Total federated communication rounds T")
    parser.add_argument("--k", type=int, default=K_PER_ROUND,
                        help="Number of clients selected per round (default from input file)")
    args = parser.parse_args()

    model_name = MODEL_NAME_MAP[args.model]
    log_filename = (
        f"fedagent_log_{model_name.lower()}_N{NUM_CLIENTS}_K{args.k}_T{args.rounds}"
        f"_M{NUM_MACHINES}.txt"
    )
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(_formatter)
    logger.addHandler(file_handler)

    print(f"Starting FedAgent | model={model_name} | N={NUM_CLIENTS} K={args.k} T={args.rounds}")
    print(f"  current_machine_ip = {CURRENT_MACHINE_IP}")
    print(f"  server_ip          = {SERVER_IP}")
    print(f"  client_ips         = {ips}")

    stop_event = threading.Event()

    # ---- Spawn local-client listeners for clients hosted on this machine. ----
    local_threads = []
    local_client_ids = [i for i in range(NUM_CLIENTS) if ips[i] == str(CURRENT_MACHINE_IP)]
    for cid in local_client_ids:
        th = threading.Thread(
            target=client_listener,
            args=(cid, args.model, stop_event),
            daemon=True,
        )
        th.start()
        local_threads.append(th)
    if local_client_ids:
        print(f"Hosting {len(local_client_ids)} local client(s): {local_client_ids}")
        # Give the OS a moment to bind the listening sockets before any peer
        # tries to connect.
        time.sleep(1.0)

    # ---- If this machine is the server, run the orchestrator. ----
    if str(CURRENT_MACHINE_IP) == str(SERVER_IP):
        server_main(args.model, args.rounds, args.k, stop_event)
    else:
        print(
            "This machine is not the server; local client listeners stay up until the "
            "orchestrator at SERVER_IP finishes and sends terminate (Ctrl+C to quit)."
        )
        try:
            stop_event.wait()
        except KeyboardInterrupt:
            stop_event.set()

    # ---- Wait for the listeners to exit (server sets stop_event via terminate msg). ----
    for th in local_threads:
        th.join(timeout=5.0)
    print("FedAgent process exiting.")


if __name__ == "__main__":
    main()
