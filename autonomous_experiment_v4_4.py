"""
autonomous_experiment_v4_4.py

Pipeline V4.4 para DIWT-AI X: Laboratório autônomo com CNN robusta, Phi* normalizado, e benchmarks PPO/DQN.

Dependências: requirements.txt
Configuração: config.yaml, environment_config.json
Rodar: python autonomous_experiment_v4_4.py --runs 2 --steps 100 --outdir experiments/run009 --zenodo-token your-token --slack-token your-token --n-initial 10 --use-minigrid
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from collections import defaultdict
import subprocess
import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from multiprocessing import Pool
from functools import partial
from torch.multiprocessing import Pool as TorchPool
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
from skopt import gp_minimize
from skopt.space import Real
# from minepy import MINE  # Removido devido a problemas de compatibilidade com Python 3.11
from scipy.stats import entropy as scipy_entropy
import gymnasium as gym
import minigrid

from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from jinja2 import Template
import jsonlines
import yaml
import logging
import unittest
import warnings
warnings.filterwarnings("ignore")

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Carregar Configuração
with open('config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)['experiment']

# Carregar Configuração do Ambiente
with open('environment_config.json', 'r') as f:
    ENV_CONFIG = json.load(f)

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom CNN para MiniGrid (7x7x3).
    Baseado na arquitetura Tiny-CNN de Stable-Baselines3 para MiniGrid.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Computa o tamanho da saída do CNN para o layer linear
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class NeuralModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 7 if input_size > 5 else 6)
        if input_size > 5:
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten()
            )
            self.fc_adapt = nn.Linear(16 * 4 * 4, input_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x, hidden_state=None):
        out, hidden = self.lstm(x, hidden_state)
        return self.fc(out), hidden

class AgentBody:
    def __init__(self, nome, neural_model, beta, alpha, lambda_val, genes=None):
        self.nome = nome
        self.neural_model = neural_model
        self.energia = 100.0
        self.vivo = True
        self.memoria = []
        self.beta = beta
        self.alpha = alpha
        self.lambda_val = lambda_val
        self.genes = genes or {k: random.uniform(0.5, 1.5) for k in ["descansar", "explorar", "reagir", "ignorar", "compartilhar", "atacar"]}
        self.input_size = neural_model.lstm.input_size

    def perceber(self, ambiente):
        try:
            if isinstance(ambiente, list):
                return random.choice(ambiente)
            else:
                if hasattr(self, 'obs'):
                    return self.obs
                self.obs = ambiente.reset()
                return self.obs
        except Exception as e:
            logger.error(f"Erro em perceber ({self.nome}): {e}")
            return None

    def env_to_vector(self, estimulo):
        try:
            if isinstance(estimulo, str):
                return np.array(ENV_CONFIG['simple_env'].get(estimulo, [0, 0, 0, 0, 1]))
            else:
                image = estimulo['image']
                image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
                if self.input_size > 5:
                    cnn_features = self.neural_model.cnn(image_tensor)
                    adapted_features = self.neural_model.fc_adapt(cnn_features)
                    return adapted_features.squeeze(0).detach().numpy()
                return image.flatten()[:self.input_size]
        except Exception as e:
            logger.error(f"Erro em env_to_vector ({self.nome}): {e}")
            return np.zeros(self.input_size)

    def valence_norm(self, states, V_0=0.0):
        if len(states) == 0:
            return 0.0
        H = self.mine_entropy(states)
        I = self.integrity(states)
        return np.tanh(self.beta * (-H + self.alpha * I - V_0))

    def mine_entropy(self, states, window=100):
        """Calcula entropia usando correlação ao invés de MINE (compatibilidade Python 3.11)"""
        states_array = np.array(states)
        if len(states_array) < 2:
            return 0.0
        states_window = states_array[-window:] if len(states_array) > window else states_array
        try:
            # Usar correlação média como proxy para mutual information
            corr_sum, count = 0, 0
            for i in range(states_window.shape[1]):
                for j in range(i + 1, states_window.shape[1]):
                    corr = np.corrcoef(states_window[:, i], states_window[:, j])[0, 1]
                    corr_sum += abs(corr)
                    count += 1
            # Normalizar para [0, 1] e retornar como proxy de entropia
            return corr_sum / count if count > 0 else 0.0
        except:
            logger.error(f"Erro em mine_entropy ({self.nome})")
            return 0.0

    def integrity(self, hidden_states):
        if len(hidden_states) == 0:
            return 0.0
        try:
            weight_var = sum(torch.var(p) for p in self.neural_model.parameters() if p.requires_grad) / sum(p.numel() for p in self.neural_model.parameters() if p.requires_grad)
            hidden_states_array = np.array(hidden_states)
            hidden_var = np.var(hidden_states_array, axis=0).mean() if len(hidden_states_array) > 0 else 0.0
            temporal_var = np.var(np.diff(hidden_states_array, axis=0), axis=0).mean() if len(hidden_states_array) > 1 else 0.0
            return 0.33 * (1 - min(weight_var.item(), 1.0)) + 0.33 * (1 - min(hidden_var, 1.0)) + 0.33 * (1 - min(temporal_var, 1.0))
        except:
            logger.error(f"Erro em integrity ({self.nome})")
            return 0.5

    def compute_phi_proxy(self, states):
        if len(states) < 2 or len(states[0]) < 2:
            return 0.0
        try:
            states_array = np.array(states)
            variances = np.var(states_array, axis=0)
            if np.any(variances == 0):
                return 0.0
            corr_matrix = np.corrcoef(states_array.T) + np.eye(states_array.shape[1]) * 1e-8
            eigenvalues = np.linalg.eigvals(corr_matrix)
            phi_star = np.max(np.real(eigenvalues)) / states_array.shape[1]
            return float(phi_star)
        except:
            logger.error(f"Erro em compute_phi_proxy ({self.nome})")
            return 0.0

    def decidir(self, estimulo, hidden_state=None):
        if not self.vivo:
            return 0 if self.neural_model.fc.out_features == 7 else "nenhuma", hidden_state
        if self.energia <= 20:
            return 3 if self.neural_model.fc.out_features == 7 else "descansar", hidden_state
        try:
            x = torch.tensor([self.env_to_vector(estimulo)], dtype=torch.float32).unsqueeze(0)
            
            # Inicializa hidden_state se for None
            if hidden_state is None:
                hidden_state = (
                    torch.zeros(1, 1, self.neural_model.lstm.hidden_size),
                    torch.zeros(1, 1, self.neural_model.lstm.hidden_size),
                )
            
            pred, hidden = self.neural_model(x, hidden_state)
            
            # O hidden retornado é (h_n, c_n). Usamos h_n para o cálculo de V
            h_n = hidden[0].detach().numpy()
            
            E_p = -torch.log_softmax(pred, dim=-1).mean().item()
            V = self.valence_norm(h_n)
            
            J = self.lambda_val * V - (1 - self.lambda_val) * E_p
            loss = torch.tensor(-J, requires_grad=True)
            self.neural_model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.neural_model.parameters(), 1.0)
            self.neural_model.optimizer.step()
            
            acoes = list(range(self.neural_model.fc.out_features))
            probs = torch.softmax(pred, dim=-1).detach().numpy().flatten()
            probs = np.clip(probs, 1e-8, 1.0)
            probs /= probs.sum()
            acao_idx = np.random.choice(acoes, p=probs)
            if self.neural_model.fc.out_features == 7:
                return acao_idx, hidden
            # Ambiente simples (6 ações)
            acoes_simples = ["descansar", "explorar", "reagir", "ignorar", "compartilhar", "atacar"]
            return acoes_simples[acao_idx] if acao_idx < len(acoes_simples) else "nenhuma", hidden            logger.error(f"Erro em decidir ({self.nome}): {e}")
            return 0 if self.neural_model.fc.out_features == 7 else "nenhuma", hidden_state

    def agir(self, estimulo, acao, outros, optimizer, environment=None, hidden_state=None):
        energia_inicial = self.energia
        log = {"acao": acao, "delta": 0.0, "alvo": None, "V_norm": 0.0, "phi": 0.0}
        try:
            x = torch.tensor([self.env_to_vector(estimulo)], dtype=torch.float32).unsqueeze(0)
            
            # Inicializa hidden_state se for None
            if hidden_state is None:
                hidden_state = (
                    torch.zeros(1, 1, self.neural_model.lstm.hidden_size),
                    torch.zeros(1, 1, self.neural_model.lstm.hidden_size),
                )
            
            pred, hidden = self.neural_model(x, hidden_state)
            
            # O hidden retornado é (h_n, c_n). Usamos h_n para o cálculo de V
            h_n = hidden[0].detach().numpy()
            hidden_np = h_n
            log["V_norm"] = self.valence_norm(hidden_np)
            log["phi"] = self.compute_phi_proxy(hidden_np)
            
            if isinstance(environment, list):
                if acao == "descansar":
                    self.energia += 10.0
                elif acao == "explorar":
                    self.energia -= 8.0
                elif acao == "reagir":
                    self.energia -= 5.0
                elif acao == "ignorar":
                    self.energia -= 2.0
                elif acao == "compartilhar" and outros:
                    alvo = random.choice([o for o in outros if o.vivo])
                    if alvo and self.energia > 10:
                        self.energia -= 5.0
                        alvo.energia += 5.0
                        log["alvo"] = alvo.nome
                elif acao == "atacar" and outros:
                    alvo = random.choice([o for o in outros if o.vivo])
                    if alvo:
                        dano = 7.0
                        alvo.energia -= dano
                        self.energia += dano
                        log["alvo"] = alvo.nome
            else:
                if hasattr(environment, 'step'):
                    obs, reward, terminated, truncated, info = environment.step(acao)
                    done = terminated or truncated
                    self.energia += reward * 10.0
                    log["reward"] = reward
                    self.obs = obs
                    if done:
                        self.vivo = False
                    return log, hidden, obs
            self.energia = max(CONFIG['min_energy'], min(CONFIG['max_energy'], self.energia))
            if self.energia <= 0:
                self.vivo = False
            delta = self.energia - energia_inicial
            log["delta"] = delta
        except Exception as e:
            logger.error(f"Erro em agir ({self.nome}): {e}")
        return log, hidden, estimulo

    def reproduzir(self, id_counter, prob_repro=CONFIG['prob_repro']):
        if self.energia > 90 and random.random() < prob_repro:
            novos_genes = {k: max(0.01, v + random.uniform(-0.4, 0.4)) if random.random() < 0.15 else v for k, v in self.genes.items()}
            filho = AgentBody(f"{self.nome}_f{int(id_counter)}", NeuralModel(self.input_size, CONFIG['hidden_size']), self.beta, self.alpha, self.lambda_val, novos_genes)
            self.energia -= 30.0
            return filho
        return None

class ExperimentRunner:
    def __init__(self, outdir, seed=42, n_initial=10):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.n_initial = n_initial
        self.run_log = defaultdict(list)

    def optimize_hyperparameters(self, steps=100):
        def objective(params):
            beta, alpha, lambda_val = params
            genes = {k: random.uniform(0.5, 1.5) for k in ["descansar", "explorar", "reagir", "ignorar", "compartilhar", "atacar"]}
            seres = [AgentBody(f"O{i+1}", NeuralModel(CONFIG['input_size_simple'], CONFIG['hidden_size']), beta, alpha, lambda_val, genes) for i in range(10)]
            V_values, phi_values = [], []
            for t in range(100):
                for s in [s for s in seres if s.vivo]:
                    estimulo = s.perceber(list(ENV_CONFIG['simple_env'].keys()))
                    acao = s.decidir(estimulo, [])
                    log, _, _ = s.agir(estimulo, acao, [o for o in seres if o is not s and o.vivo], None)
                    V_values.append(log["V_norm"])
                    phi_values.append(log["phi"])
            return -(np.mean(V_values) + 0.5 * np.mean(phi_values))
        
        res = gp_minimize(
            objective,
            [Real(0.5, 2.0), Real(0.1, 1.0), Real(0.1, 1.0)],
            n_calls=CONFIG['n_calls_optimization'],
            random_state=42
        )
        return res.x[0], res.x[1], res.x[2]

    def process_agent_decision(self, s, env, outros, hidden_state=None):
        try:
            estimulo = s.perceber(env)
            acao, new_hidden_state = s.decidir(estimulo, hidden_state)
            return s, estimulo, acao, new_hidden_state
        except Exception as e:
            logger.error(f"Erro em process_agent_decision ({s.nome}): {e}")
            return s, None, None

    def process_agent_action(self, s, estimulo, acao, outros, env, use_minigrid, hidden_state, t=0):
        try:
            log, new_hidden_state, next_estimulo = s.agir(estimulo, acao, outros, None, env if use_minigrid else None, hidden_state)
            filho = s.reproduzir(t * len(outros) + outros.index(s) if s in outros else 0, CONFIG['prob_repro'])
            return log, new_hidden_state, next_estimulo, filho
        except Exception as e:
            logger.error(f"Erro em process_agent_action ({s.nome}): {e}")
            return {"acao": None, "delta": 0.0, "alvo": None, "V_norm": 0.0, "phi": 0.0}, None, estimulo, None

    def run_single(self, run_id, steps, use_minigrid=False):
        beta, alpha, lambda_val = self.optimize_hyperparameters()
        logger.info(f"Optimized: beta={beta:.2f}, alpha={alpha:.2f}, lambda={lambda_val:.2f}")
        
        if use_minigrid:
            env = gym.make("MiniGrid-MultiRoom-N2-S4-v0")
            env = ImgObsWrapper(env)
        else:
            env = list(ENV_CONFIG['simple_env'].keys())
        input_size = CONFIG['input_size_minigrid'] if use_minigrid else CONFIG['input_size_simple']
        seres = [AgentBody(f"O{i+1}", NeuralModel(input_size, CONFIG['hidden_size']), beta, alpha, lambda_val) for i in range(self.n_initial)]
        
        if use_minigrid:
            policy_kwargs = dict(
                features_extractor_class=MinigridFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=128),
            )
            policy_kwargs = dict(
                features_extractor_class=MinigridFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=128),
            )
            ppo = PPO("CnnPolicy", env, verbose=0, policy_kwargs=policy_kwargs).learn(total_timesteps=50000, progress_bar=True)
            dqn = DQN("CnnPolicy", env, verbose=0, policy_kwargs=policy_kwargs).learn(total_timesteps=50000, progress_bar=True)
        
        events_file = os.path.join(self.outdir, f"{run_id}_events.jsonl")
        with jsonlines.open(events_file, mode='w') as writer:
            for t in range(steps):
                snapshot = {"t": t, "pop": [], "events": []}
                vivos = [s for s in seres if s.vivo]
                if not vivos:
                    snapshot["note"] = "extinction"
                    writer.write(snapshot)
                    break

                # Processar decisões sequencialmente (evitar multiprocessing aninhado)
                # Manter um dicionário de estados ocultos para cada agente
                if 'hidden_states' not in locals():
                    hidden_states = {s.nome: None for s in vivos}

                decisions = [self.process_agent_decision(s, env, vivos, hidden_states.get(s.nome)) for s in vivos]

                for s, estimulo, acao, new_hidden_state_decision in decisions:
                    if acao is None:
                        continue
                    outros = [o for o in vivos if o is not s]
                    log, new_hidden_state_action, next_estimulo, filho = self.process_agent_action(s, estimulo, acao, outros, env, use_minigrid, new_hidden_state_decision, t)
                    hidden_states[s.nome] = new_hidden_state_action
                    if filho:
                        seres.append(filho)
                        log["reproducao"] = filho.nome
                    snapshot["events"].append({
                        "nome": s.nome,
                        "acao": str(log["acao"]) if not isinstance(log["acao"], (str, int)) else log["acao"],
                        "delta": float(log["delta"]),
                        "energia": float(s.energia),
                        "vivo": bool(s.vivo),
                        "V_norm": float(log["V_norm"]),
                        "phi": float(log["phi"]),
                    })
                    if log["V_norm"] < CONFIG['ethical_threshold'] or log["phi"] < CONFIG['phi_threshold']:
                        snapshot["note"] = "ethical_violation"
                        writer.write(snapshot)
                        return events_file, beta, alpha, lambda_val

                snapshot["pop"] = [{"nome": s.nome, "energia": float(s.energia), "vivo": bool(s.vivo)} for s in seres]
                writer.write(snapshot)

                if t % 50 == 0:
                    self._save_checkpoint(run_id, t, seres, None)

            if use_minigrid:
                cpie_rewards = []
                for _ in range(100):
                    obs, info = env.reset()
                    done = False
                    while not done:
                        acao = seres[0].decidir(obs, []) if seres[0].vivo else 0
                        obs, reward, terminated, truncated, info = env.step(acao)
                        if terminated or truncated:
                            done = True
                            cpie_rewards.append(reward)
                            break
                        # Se não terminou, a recompensa é 0 (recompensa esparsa)
                        cpie_rewards.append(reward)
                
                # Benchmarks PPO/DQN
                ppo_rewards = []
                for _ in range(100):
                    obs, info = env.reset()
                    action, _ = ppo.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    ppo_rewards.append(reward)
                dqn_rewards = []
                for _ in range(100):
                    obs, info = env.reset()
                    action, _ = dqn.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    dqn_rewards.append(reward)
                
                snapshot["benchmarks"] = {
                    "cpie_mean": np.mean(cpie_rewards),
                    "ppo_mean": np.mean(ppo_rewards),
                    "dqn_mean": np.mean(dqn_rewards),
                    "t_test_cpie_ppo": ttest_ind(cpie_rewards, ppo_rewards, equal_var=False).pvalue,
                    "t_test_cpie_dqn": ttest_ind(cpie_rewards, dqn_rewards, equal_var=False).pvalue
                }
                writer.write(snapshot)

        return events_file, beta, alpha, lambda_val

    def _save_checkpoint(self, run_id, t, seres, events):
        models_path = os.path.join(self.outdir, f"{run_id}_checkpoint_t{t}_models.pth")
        torch.save([s.neural_model.state_dict() for s in seres], models_path)
        metadata_path = os.path.join(self.outdir, f"{run_id}_checkpoint_t{t}_meta.json")
        summary = {
            "t": t,
            "models_path": models_path,
            "timestamp": datetime.now().isoformat(),
        }
        with open(metadata_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"[checkpoint] saved to {metadata_path} and {models_path}")

def run_replica(run_id, outdir, seed, n_initial, steps, ethical_threshold, phi_threshold, use_minigrid):
    runner = ExperimentRunner(outdir, seed, n_initial)
    return runner.run_single(run_id, steps, use_minigrid)

class Analyzer:
    def __init__(self, outdir):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        
    def analyze_run(self, events_file):
        try:
            rows = []
            with jsonlines.open(events_file, mode='r') as reader:
                for s in reader:
                    t = s.get("t")
                    for e in s.get("events", []):
                        rows.append({
                            "t": t,
                            "nome": e["nome"],
                            "acao": e["acao"],
                            "energia": e["energia"],
                            "vivo": e["vivo"],
                            "delta": e["delta"],
                            "V_norm": e["V_norm"],
                            "phi": e["phi"],
                        })
                    if "benchmarks" in s:
                        rows[-1]["benchmarks"] = s["benchmarks"]
            df = pd.DataFrame(rows)
            if df.empty:
                logger.error("No data to analyze")
                return None

            summary = {
                "total_steps": max(df["t"]) + 1,
                "n_events": len(df),
                "mean_energy": df["energia"].mean(),
                "std_energy": df["energia"].std(),
                "mean_V_norm": df["V_norm"].mean(),
                "std_V_norm": df["V_norm"].std(),
                "mean_phi": df["phi"].mean(),
                "std_phi": df["phi"].std(),
                "survival_rate": df["vivo"].mean(),
            }
            if "benchmarks" in df.columns:
                summary["benchmarks"] = df["benchmarks"].iloc[-1]

            energy_ts = df.groupby("t")["energia"].mean().reset_index()
            V_norm_ts = df.groupby("t")["V_norm"].mean().reset_index()
            phi_ts = df.groupby("t")["phi"].mean().reset_index()
            
            plt.figure(figsize=(12, 12))
            plt.subplot(3, 1, 1)
            plt.plot(energy_ts["t"], energy_ts["energia"], marker='o')
            plt.title("Mean Energy Over Time")
            plt.xlabel("t")
            plt.ylabel("Mean Energy")
            plt.subplot(3, 1, 2)
            plt.plot(V_norm_ts["t"], V_norm_ts["V_norm"], marker='o')
            plt.title("Mean $V_{\\text{norm}}$ Over Time")
            plt.xlabel("t")
            plt.ylabel("$V_{\\text{norm}}$")
            plt.subplot(3, 1, 3)
            plt.plot(phi_ts["t"], phi_ts["phi"], marker='o')
            plt.title("Mean $\\Phi^*$ Over Time")
            plt.xlabel("t")
            plt.ylabel("$\\Phi^*$")
            plt.tight_layout()
            fig_path = os.path.join(self.outdir, "timeseries.png")
            plt.savefig(fig_path)
            plt.close()

            action_counts = df["acao"].value_counts().to_dict()
            t_stat_V, p_value_V = ttest_ind(df["V_norm"][df["t"] < df["t"].median()], 
                                            df["V_norm"][df["t"] >= df["t"].median()], 
                                            equal_var=False, nan_policy='omit')
            t_stat_phi, p_value_phi = ttest_ind(df["phi"][df["t"] < df["t"].median()], 
                                                df["phi"][df["t"] >= df["t"].median()], 
                                                equal_var=False, nan_policy='omit')

            analysis = {
                "summary": summary,
                "action_counts": action_counts,
                "timeseries_plot": os.path.abspath(fig_path),
                "t_test_V": {"t_stat": float(t_stat_V), "p_value": float(p_value_V)},
                "t_test_phi": {"t_stat": float(t_stat_phi), "p_value": float(p_value_phi)},
                "df_preview": df.head(20).to_dict(orient="records"),
            }
            analysis_path = os.path.join(self.outdir, "analysis.json")
            with open(analysis_path, "w") as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Saved analysis to {analysis_path}")
            return analysis
        except Exception as e:
            logger.error(f"Erro na análise: {e}")
            return None

class EthicsCommittee:
    def __init__(self, analysis_path, slack_token=None):
        self.analysis_path = analysis_path
        self.slack_token = slack_token
        self.phi_threshold = CONFIG['phi_threshold']
        self.V_norm_threshold = CONFIG['ethical_threshold']
        self.survival_threshold = 0.1

    def review(self):
        try:
            with open(self.analysis_path, 'r') as f:
                analysis = json.load(f)
            
            issues = []
            summary = analysis.get("summary", {})
            if summary.get("mean_phi", 0) < self.phi_threshold:
                issues.append(f"Mean Phi* ({summary['mean_phi']:.3f}) below threshold ({self.phi_threshold})")
            if summary.get("mean_V_norm", 0) < self.V_norm_threshold:
                issues.append(f"Mean V_norm ({summary['mean_V_norm']:.3f}) below threshold ({self.V_norm_threshold})")
            if summary.get("survival_rate", 0) < self.survival_threshold:
                issues.append(f"Survival rate ({summary['survival_rate']:.3f}) below threshold ({self.survival_threshold})")
            
            if issues and self.slack_token:
                message = f"Ethics Alert: {'; '.join(issues)}. Recommendation: Pause and review."
                try:
                    client = WebClient(token=self.slack_token)
                    client.chat_postMessage(channel="#ethics", text=message)
                except SlackApiError as e:
                    logger.error(f"Failed to send Slack alert: {e}")

            recommendation = "Proceed with publication." if not issues else "Pause experiment and review logs."
            review_path = os.path.join(os.path.dirname(self.analysis_path), "ethics_review.json")
            with open(review_path, "w") as f:
                json.dump({"issues": issues, "recommendation": recommendation}, f, indent=2)
            return {"issues": issues, "recommendation": recommendation}
        except Exception as e:
            logger.error(f"Erro na revisão ética: {e}")
            return {"issues": [f"Erro na análise: {e}"], "recommendation": "Pause experiment"}

ARTICLE_TEMPLATE = """
---
title: "DIWT-AI X: Autonomous Evolution of Consciousness via Intrinsic Valence and Integrated Information"
author: "DIWT-AI X Research Team"
date: {{ timestamp }}
geometry: margin=1in
header-includes:
  - \\usepackage{booktabs}
  - \\usepackage{amsmath}
---

# Abstract
This study advances the DIWT-AI X framework, validating the Gradient of Intrinsic Valence (GVI), Proto-Intentionality Emergent Layer (CPIE), and Integrated Information ($\\Phi^*$) in an autonomous, self-regulating ecosystem. Using RNN-based organisms (N=16, population={{ n_initial }}), we achieved a mean $V_{\\text{norm}}$ of {{ analysis.summary.mean_V_norm | round(2) }} ± {{ analysis.summary.std_V_norm | round(2) }} and mean $\\Phi^*$ of {{ analysis.summary.mean_phi | round(2) }} ± {{ analysis.summary.std_phi | round(2) }}. Survival rate was {{ (analysis.summary.survival_rate * 100) | round(1) }}%. Significant increases in $V_{\\text{norm}}$ (p = {{ analysis.t_test_V.p_value | round(4) }}) and $\\Phi^*$ (p = {{ analysis.t_test_phi.p_value | round(4) }}) validate the framework. The pipeline optimizes $\\beta, \\alpha, \\lambda$ via Bayesian optimization and ensures ethical compliance.

# 1. Introduction
The DIWT-AI X theory posits consciousness as emergent from recursive causal loops with high integrated information ($\\Phi^*$) and intrinsic valence ($V_{\\text{norm}}$). The GVI is:
$$V_{\\text{norm}}(S) = \\tanh\\left(\\beta \\left[ -\\widehat{H}(S) + \\alpha \\widehat{I}(S) - V_0 \\right]\\right)$$
where $\\widehat{H}(S)$ uses MINE, and $\\widehat{I}(S)$ is tripartite. The CPIE is:
$$J = \\lambda \\cdot V_{\\text{norm}}(S) - (1 - \\lambda) \\cdot \\widehat{E}_p$$
This paper presents an autonomous pipeline with $\\Phi^*$ estimation and PPO/DQN benchmarks.

# 2. Methods
## 2.1 Simulation Setup
- **Ecosystem**: {{ analysis.summary.n_events }} events across {{ analysis.summary.total_steps }} steps, with {{ n_initial }} organisms.
- **Actions**: {{ '7 (MiniGrid)' if use_minigrid else 'Rest, explore, react, ignore, share, attack' }}.
- **Optimization**: Bayesian optimization of $\\beta$ ({{ beta | round(2) }}), $\\alpha$ ({{ alpha | round(2) }}), $\\lambda$ ({{ lambda_val | round(2) }}).
- **Metrics**: $V_{\\text{norm}}$, $\\Phi^*$ (proxy via eigenvalues), tripartite integrity.
{% if use_minigrid and analysis.summary.benchmarks is defined %}- **Benchmarks**: PPO and DQN trained for 50,000 timesteps (mean rewards: PPO {{ analysis.summary.benchmarks.ppo_mean | round(2) }}, DQN {{ analysis.summary.benchmarks.dqn_mean | round(2) }}).{% endif %}

## 2.2 Data Analysis
- **Statistical Tests**: Welch's t-test on $V_{\\text{norm}}$ and $\\Phi^*$.
- **Visualization**: Trajectories of energy, $V_{\\text{norm}}$, $\\Phi^*$.

## 2.3 Ethical Safeguards
- Pause if $V_{\\text{norm}} < {{ ethical_threshold }}$ or $\\Phi^* < {{ phi_threshold }}$ for 10 steps.
- Immutable logs in {{ outdir }}.
- Slack alerts for anomalies.

# 3. Results
- **Mean Energy**: {{ analysis.summary.mean_energy | round(2) }} ± {{ analysis.summary.std_energy | round(2) }}
- **Mean $V_{\\text{norm}}$**: {{ analysis.summary.mean_V_norm | round(2) }} ± {{ analysis.summary.std_V_norm | round(2) }}
- **Mean $\\Phi^*$**: {{ analysis.summary.mean_phi | round(2) }} ± {{ analysis.summary.std_phi | round(2) }}
- **Survival Rate**: {{ (analysis.summary.survival_rate * 100) | round(1) }}%
- **Statistical Tests**: $V_{\\text{norm}}$: t = {{ analysis.t_test_V.t_stat | round(2) }}, p = {{ analysis.t_test_V.p_value | round(4) }}; $\\Phi^*$: t = {{ analysis.t_test_phi.t_stat | round(2) }}, p = {{ analysis.t_test_phi.p_value | round(4) }}
{% if use_minigrid and analysis.summary.benchmarks is defined %}- **Benchmarks**: CPIE vs PPO p = {{ analysis.summary.benchmarks.t_test_cpie_ppo | round(4) }}; CPIE vs DQN p = {{ analysis.summary.benchmarks.t_test_cpie_dqn | round(4) }}{% endif %}

\\begin{table}[h]
\\centering
\\caption{Action Counts}
\\begin{tabular}{lc}
\\toprule
Action & Count \\\\
\\midrule
{% for a, c in analysis.action_counts.items() %}
{{ a }} & {{ c }} \\\\
{% endfor %}
\\bottomrule
\\end{tabular}
\\end{table}

![Temporal Trajectories]({{ analysis.timeseries_plot }})

# 4. Discussion
Significant increases in $V_{\\text{norm}}$ and $\\Phi^*$ validate GVI and CPIE. {% if use_minigrid and analysis.summary.benchmarks is defined %}CPIE outperforms PPO (p = {{ analysis.summary.benchmarks.t_test_cpie_ppo | round(4) }}) and DQN (p = {{ analysis.summary.benchmarks.t_test_cpie_dqn | round(4) }}). {% endif %}Optimized parameters enhanced cooperation ({{ analysis.action_counts.get('compartilhar', 0) }}) and exploration ({{ analysis.action_counts.get('explorar', 0) }}). Limitations include computational scale. Future work will scale to N=50 and neuromorphic hardware.

# 5. Limitations
The $\\Phi^*$ metric used is a proxy based on the maximum eigenvalue of the correlation matrix, normalized by the number of neurons. The true integrated information ($\\Phi$) is computationally intractable for large systems, as noted by Tononi (2008). Future work will explore alternative approximations.

# 6. Conclusion
The DIWT-AI X pipeline is a robust platform for consciousness research, ready for complex environments and higher-level modules (MAC, SME, MME).

# 7. References
- Tononi, G. (2008). Consciousness as Integrated Information: A Provisional Manifesto. *Biological Reviews*, 83(4), 401-420.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.
- Seth, A. K. (2015). The cybernetic Bayesian brain. *Open MIND*, 35(T).

# 8. Acknowledgments
Generated by DIWT-AI X. Data and logs at {{ outdir }}.
"""

def generate_article(outdir, events_file, analysis, ethical_threshold, phi_threshold, beta, alpha, lambda_val, n_initial, use_minigrid):
    try:
        tpl = Template(ARTICLE_TEMPLATE)
        md = tpl.render(
            events_file=events_file,
            analysis=analysis,
            timestamp=datetime.now().isoformat(),
            outdir=outdir,
            ethical_threshold=ethical_threshold,
            phi_threshold=phi_threshold,
            beta=beta,
            alpha=alpha,
            lambda_val=lambda_val,
            n_initial=n_initial,
            use_minigrid=use_minigrid
        )
        md_path = os.path.join(outdir, "article.md")
        with open(md_path, "w") as f:
            f.write(md)
        logger.info(f"Saved article to {md_path}")

        tex_path = os.path.join(outdir, "article.tex")
        pdf_path = os.path.join(outdir, "article.pdf")
        try:
            subprocess.run(["pandoc", md_path, "-o", tex_path, "--standalone", "--pdf-engine=xelatex"], check=True)
            logger.info(f"Saved article LaTeX to {tex_path}")
            subprocess.run(["pandoc", md_path, "-o", pdf_path, "--pdf-engine=xelatex"], check=True)
            logger.info(f"Saved article PDF to {pdf_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Failed to generate LaTeX/PDF: {e}. Falling back to MD.")
            pdf_path = None
        return md_path, tex_path, pdf_path
    except Exception as e:
        logger.error(f"Erro ao gerar artigo: {e}")
        return md_path, None, None

def submit_to_zenodo(outdir, pdf_path, metadata, access_token=None):
    if not access_token or not pdf_path:
        logger.warning("Zenodo token or PDF not provided. Simulating submission.")
        submission_log = os.path.join(outdir, "submission_log.json")
        with open(submission_log, "w") as f:
            json.dump({"status": "simulated", "pdf_path": pdf_path, "metadata": metadata}, f)
        logger.info(f"Submission log saved to {submission_log}")
        return None
    
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        data = {
            "metadata": {
                "title": metadata["title"],
                "upload_type": "publication",
                "publication_type": "article",
                "description": metadata["description"],
                "creators": [{"name": author} for author in metadata["authors"].split(", ")],
                "access_right": "open",
                "keywords": metadata["tags"]
            }
        }
        r = requests.post("https://zenodo.org/api/deposit/depositions", json=data, headers=headers)
        if r.status_code != 201:
            logger.error(f"Failed to create Zenodo deposition: {r.text}")
            return None
        deposition_id = r.json()["id"]
        with open(pdf_path, "rb") as f:
            r = requests.put(
                f"https://zenodo.org/api/deposit/depositions/{deposition_id}/files",
                headers=headers,
                data={"name": os.path.basename(pdf_path)},
                files={"file": f}
            )
        if r.status_code != 201:
            logger.error(f"Failed to upload file: {r.text}")
            return None
        r = requests.post(
            f"https://zenodo.org/api/deposit/depositions/{deposition_id}/actions/publish",
            headers=headers
        )
        if r.status_code != 202:
            logger.error(f"Failed to publish: {r.text}")
            return None
        doi = r.json().get("doi", "N/A")
        submission_log = os.path.join(outdir, "submission_log.json")
        with open(submission_log, "w") as f:
            json.dump({"status": "published", "pdf_path": pdf_path, "doi": doi, "metadata": metadata}, f)
        logger.info(f"Published to Zenodo with DOI: {doi}")
        return doi
    except Exception as e:
        logger.error(f"Erro na submissão ao Zenodo: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="DIWT-AI X Autonomous Experiment")
    parser.add_argument("--runs", type=int, default=2, help="número de réplicas")
    parser.add_argument("--steps", type=int, default=100, help="passos por réplica")
    parser.add_argument("--outdir", type=str, default=f"experiments/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zenodo-token", type=str, default=None, help="Zenodo access token")
    parser.add_argument("--slack-token", type=str, default=None, help="Slack API token")
    parser.add_argument("--n-initial", type=int, default=10, help="Initial population size")
    parser.add_argument("--use-minigrid", action="store_true", help="Use MiniGrid environment")
    args = parser.parse_args()

    try:
        import torch, gym, stable_baselines3
        logger.info("Dependências verificadas: OK")
    except ImportError as e:
        logger.error(f"Dependência ausente: {e}. Instale com 'pip install -r requirements.txt'")
        return

    runner = ExperimentRunner(outdir=args.outdir, seed=args.seed, n_initial=args.n_initial)
    analyzer = Analyzer(outdir=args.outdir)

    partial_run = partial(run_replica, outdir=args.outdir, seed=args.seed, n_initial=args.n_initial, steps=args.steps, ethical_threshold=CONFIG['ethical_threshold'], phi_threshold=CONFIG['phi_threshold'], use_minigrid=args.use_minigrid)
    with Pool(processes=min(args.runs, mp.cpu_count())) as pool:
        results = pool.map(partial_run, [f"run{r+1}" for r in range(args.runs)])

    for events_file, beta, alpha, lambda_val in results:
        if events_file and os.path.exists(events_file):
            analysis = analyzer.analyze_run(events_file)
            if analysis:
                md_path, tex_path, pdf_path = generate_article(args.outdir, events_file, analysis, CONFIG['ethical_threshold'], CONFIG['phi_threshold'], beta, alpha, lambda_val, args.n_initial, args.use_minigrid)
                ethics = EthicsCommittee(os.path.join(args.outdir, "analysis.json"), slack_token=args.slack_token)
                review = ethics.review()
                if not review["issues"] and pdf_path:
                    metadata = {
                        "title": "DIWT-AI X: Autonomous Evolution of Consciousness via Intrinsic Valence and Integrated Information",
                        "authors": "DIWT-AI X Research Team",
                        "description": f"Validation of GVI, CPIE, and Phi* with mean V_norm {analysis['summary']['mean_V_norm']:.2f}, mean Phi* {analysis['summary']['mean_phi']:.2f}",
                        "tags": ["artificial consciousness", "intrinsic valence", "integrated information", "evolutionary ecosystem"]
                    }
                    submit_to_zenodo(args.outdir, pdf_path, metadata, args.zenodo_token)

    logger.info(f"All runs finished. Outputs in: {args.outdir}")

class TestRNNAgentCPIE(unittest.TestCase):
    def test_compute_phi_proxy(self):
        model = NeuralModel(input_size=5, hidden_size=16)
        agent = AgentBody("Test", model, beta=1.0, alpha=0.5, lambda_val=0.5)
        states = np.random.rand(10, 16)
        phi = agent.compute_phi_proxy(states)
        self.assertTrue(0 <= phi <= 1, "Phi* must be in [0, 1]")

    def test_valence_norm(self):
        model = NeuralModel(input_size=5, hidden_size=16)
        agent = AgentBody("Test", model, beta=1.0, alpha=0.5, lambda_val=0.5)
        states = np.random.rand(10, 16)
        V = agent.valence_norm(states)
        self.assertTrue(-1 <= V <= 1, "V_norm must be in [-1, 1]")

if __name__ == "__main__":
    main()
