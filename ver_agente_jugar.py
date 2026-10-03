"""
ver_agente_jugar.py
====================
Carga el modelo DQN entrenado y lo pone a jugar ALE/Pong-v5 en modo visual.

Uso:
    python ver_agente_jugar.py
    python ver_agente_jugar.py --modelo dqn_pong_best.pth --episodios 5
    python ver_agente_jugar.py --velocidad 33   # 30 FPS

IMPORTANTE: Coloca este archivo junto a dqn_pong_best.pth
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import ale_py
import cv2

# ── Configuración ────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Red neuronal (idéntica a la del entrenamiento) ────────────────────────────
class DQNNetwork(nn.Module):
    def __init__(self, n_actions: int = 6):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_block(self.conv_block(x))


# ── Preprocesamiento (igual que en entrenamiento) ─────────────────────────────
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(84, 84), dtype=np.float32
        )

    def observation(self, obs):
        gray    = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0


class FrameStack(gym.Wrapper):
    """Apila los ultimos n_frames en un tensor (n_frames, 84, 84)."""
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames   = []
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(n_frames, 84, 84), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info   = self.env.reset(**kwargs)   # obs: (84, 84)
        self.frames = [obs] * self.n_frames       # 4 copias del primer frame
        return np.array(self.frames, dtype=np.float32), info  # (4, 84, 84)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)  # obs: (84, 84)
        self.frames.pop(0)
        self.frames.append(obs)
        return np.array(self.frames, dtype=np.float32), reward, term, trunc, info


def make_env(render_mode="human"):
    env = gym.make("ALE/Pong-v5", render_mode=render_mode)
    env = PreprocessFrame(env)
    env = FrameStack(env, n_frames=4)
    return env


# ── Carga del modelo ─────────────────────────────────────────────────────────
def cargar_modelo(ruta: str, n_actions: int = 6) -> DQNNetwork:
    net        = DQNNetwork(n_actions=n_actions).to(DEVICE)
    checkpoint = torch.load(ruta, map_location=DEVICE, weights_only=False)

    if "q_online_state_dict" in checkpoint:
        net.load_state_dict(checkpoint["q_online_state_dict"])
        ep = checkpoint.get("episode",     "?")
        rw = checkpoint.get("best_reward", "?")
        print(f"  Episodio guardado  : {ep}")
        print(f"  Mejor recompensa   : {rw}")
    elif "state_dict" in checkpoint:
        net.load_state_dict(checkpoint["state_dict"])
    else:
        net.load_state_dict(checkpoint)

    net.eval()
    return net


# ── Bucle de juego ────────────────────────────────────────────────────────────
def jugar(modelo_path: str, n_episodios: int = 3, delay_ms: int = 16):
    print(f"\n{'='*55}")
    print(f"  DQN Pong — Agente entrenado jugando")
    print(f"  Modelo    : {modelo_path}")
    print(f"  Dispositivo: {DEVICE}")
    print(f"{'='*55}\n")

    print("[INFO] Cargando modelo...")
    net = cargar_modelo(modelo_path)
    print("[INFO] Modelo cargado. Abriendo ventana de juego...\n")

    env         = make_env(render_mode="human")
    recompensas = []

    for ep in range(1, n_episodios + 1):
        state, _      = env.reset()
        total_reward  = 0.0
        pasos         = 0
        done          = False

        print(f"[Episodio {ep}/{n_episodios}] Jugando...")

        while not done:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                action  = int(net(state_t).argmax(dim=1).item())

            state, reward, terminated, truncated, _ = env.step(action)
            done          = terminated or truncated
            total_reward += reward
            pasos        += 1

            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

        recompensas.append(total_reward)
        emoji = "🏆 GANA" if total_reward > 0 else ("🤝 EMPATE" if total_reward == 0 else "❌ PIERDE")
        print(f"  Recompensa = {total_reward:+.0f}  {emoji}  |  Pasos: {pasos:,}\n")

    env.close()

    print(f"{'='*55}")
    print(f"  RESUMEN — {n_episodios} episodios")
    print(f"  Media   : {np.mean(recompensas):+.1f}")
    print(f"  Máximo  : {max(recompensas):+.0f}")
    print(f"  Mínimo  : {min(recompensas):+.0f}")
    print(f"{'='*55}\n")


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ver al agente DQN jugar Pong")
    parser.add_argument("--modelo",    type=str, default="dqn_pong_best.pth",
                        help="Ruta al .pth del modelo (default: dqn_pong_best.pth)")
    parser.add_argument("--episodios", type=int, default=3,
                        help="Numero de episodios a jugar (default: 3)")
    parser.add_argument("--velocidad", type=int, default=16,
                        help="Delay entre frames en ms (default: 16 ~ 60fps, 0 = max speed)")
    args = parser.parse_args()

    jugar(modelo_path=args.modelo, n_episodios=args.episodios, delay_ms=args.velocidad)


