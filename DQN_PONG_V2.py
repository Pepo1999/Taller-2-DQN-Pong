# =============================================================================
# PROYECTO: Deep Q-Network (DQN) para ALE/Pong-v5
# MAESTRÍA EN INTELIGENCIA ARTIFICIAL — TALLER 2
# Basado en: Mnih et al., 2015 — "Human-level control through deep reinforcement learning"
#
# ENTORNO: ALE/Pong-v5 (Arcade Learning Environment)
#   - Observaciones: Imágenes RGB de 210x160x3 → preprocesadas a (4, 84, 84)
#   - Acciones:      Espacio discreto de 6 acciones
#   - Recompensa:    +1 al anotar, -1 al recibir, 0 en otro caso
#   - Terminación:   Al llegar a 21 puntos (episodio terminado)
#
# ALGORITMO: DQN con Replay Buffer, Target Network, epsilon-greedy, Bellman
# COMPATIBILIDAD: Google Colab (GPU recomendada)
# =============================================================================

# SECCION 0: INSTALACION DE DEPENDENCIAS (descomentar en Google Colab)
# !pip install gymnasium[atari] ale-py AutoROM opencv-python --quiet
# !pip install torch torchvision --quiet

import os
os.environ["AUTO_ROM_ACCEPT_LICENSE"] = "true"

# SECCION 1: IMPORTACIONES

import gymnasium as gym
import ale_py # Añadido para asegurar el registro de los entornos ALE
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Dispositivo de computo  : {DEVICE}")
print(f"[INFO] PyTorch version         : {torch.__version__}")
print(f"[INFO] Gymnasium version       : {gym.__version__}")


# SECCION 2: PREPROCESAMIENTO DEL ENTORNO
#
# JUSTIFICACION DEL PIPELINE:
#
#   El entorno ALE/Pong-v5 entrega frames RGB (210,160,3) uint8 con
#   valores en [0,255]. Aplicamos el pipeline de Mnih et al. (2015):
#
#   1. ESCALA DE GRISES: Fórmula de luminancia Y=0.299R+0.587G+0.114B
#      Elimina información cromática irrelevante en Pong.
#      (210,160,3) -> (210,160)
#
#   2. REDIMENSIONADO 84x84 (INTER_AREA):
#      Comprime de 33,600 a 7,056 valores por frame.
#      (210,160) -> (84,84)
#
#   3. NORMALIZACION [0.0, 1.0]:
#      División entre 255.0 para estabilizar gradientes.
#
#   4. FRAME STACKING k=4:
#      Un frame ónico viola la propiedad de Markov (no hay velocidad).
#      4 frames consecutivos permiten inferir:
#        - Posición de la pelota
#        - Dirección y velocidad (diferencia entre frames)
#        - Posición de las paletas
#      Estado final: (4, 84, 84)

class PreprocessFrame(gym.ObservationWrapper):
    """
    Wrapper: transforma cada frame de Pong.
    Input  : RGB  (210, 160, 3) uint8  [0, 255]
    Output : Gray (  1,  84, 84) float32 [0.0, 1.0]
    """

    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(1, shape[0], shape[1]),
            dtype=np.float32
        )

    def observation(self, obs):
        """
        Convierte frame RGB a grayscale normalizado.
        Args:
            obs : np.ndarray (210, 160, 3) uint8
        Returns:
            frame: np.ndarray (1, 84, 84) float32 en [0.0, 1.0]
        """
        import cv2
        gray    = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.shape[1], self.shape[0]),
                             interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized[np.newaxis, :, :]  # (1, 84, 84)


class FrameStackWrapper(gym.ObservationWrapper):
    """
    Wrapper: apila los ultimos n_frames en el eje de canales.
    Input  : single frame (1, 84, 84)
    Output : stacked state (n_frames, 84, 84)
    """

    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames   = deque(maxlen=n_frames)
        old_shape = env.observation_space.shape   # (1, 84, 84)
        new_shape = (n_frames, old_shape[1], old_shape[2])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs[0])
        return self._get_obs(), info

    def observation(self, obs):
        self.frames.append(obs[0])
        return self._get_obs()

    def _get_obs(self):
        return np.array(self.frames, dtype=np.float32)  # (4, 84, 84)


def make_env(env_id="ALE/Pong-v5", render_mode=None):
    """
    Construye el entorno con pipeline completo de preprocesamiento.
    Pipeline: gym.make() -> PreprocessFrame (1,84,84) -> FrameStackWrapper (4,84,84)
    Returns: gymnasium.Env con obs_space.shape == (4, 84, 84)
    """
    env = gym.make(env_id, render_mode=render_mode)
    env = PreprocessFrame(env, shape=(84, 84))
    env = FrameStackWrapper(env, n_frames=4)
    return env


# SECCION 3: REPLAY BUFFER
#
# FUNDAMENTO:
#   En Q-Learning online, muestras consecutivas estan correlacionadas
#   temporalmente: rho(t_i, t_{i+1}) ~= 1.
#   El SGD asume muestras i.i.d.; violar esto causa inestabilidad.
#   El Experience Replay almacena D={(s_i,a_i,r_i,s'_i)} y muestrea
#   mini-batches aleatorios, rompiendo la correlacion temporal y
#   permitiendo reutilizar experiencias multiples veces.
#
#   Capacidad: 100,000 transiciones
#   (~3 GB RAM con estados float32 de forma (4,84,84))

class ReplayBuffer:
    """
    Memoria circular FIFO para Experience Replay.
    Almacena transiciones (s, a, r, s', done) y muestrea mini-batches.
    """

    def __init__(self, capacity=50_000):
        self.buffer   = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        """
        Almacena una transicion de 5-tupla.
        Args:
            state      : np.ndarray (4, 84, 84) float32 — estado s_t
            action     : int                            — accion a_t
            reward     : float                          — recompensa clipeada
            next_state : np.ndarray (4, 84, 84) float32 — estado s_{t+1}
            done        : bool                           — fin de episodio?
        """
        # Guardar como uint8 para reducir uso de RAM 4x (float32 -> uint8)
        state_u8      = (state      * 255).astype(np.uint8)
        next_state_u8 = (next_state * 255).astype(np.uint8)
        self.buffer.append((state_u8, action, reward, next_state_u8, done))

    def sample(self, batch_size):
        """
        Muestrea batch_size transiciones al azar.
        Returns:
            Tuple de tensores PyTorch en DEVICE:
              states      : (batch_size, 4, 84, 84) float32
              actions     : (batch_size,)            int64
              rewards     : (batch_size,)            float32
              next_states : (batch_size, 4, 84, 84) float32
              dones       : (batch_size,)            float32
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Reconvertir uint8 -> float32 normalizado [0,1]
        states_f      = np.array(states,      dtype=np.float32) / 255.0
        next_states_f = np.array(next_states, dtype=np.float32) / 255.0
        return (
            torch.tensor(states_f,              dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(actions),     dtype=torch.int64  ).to(DEVICE),
            torch.tensor(np.array(rewards),     dtype=torch.float32).to(DEVICE),
            torch.tensor(next_states_f,         dtype=torch.float32).to(DEVICE),
            torch.tensor(np.array(dones),       dtype=torch.float32).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)

    @property
    def is_ready(self):
        """
        True cuando el buffer tiene suficientes muestras (warm-up minimo).
        """
        return len(self.buffer) >= 10_000


# SECCION 4: RED NEURONAL — DQN (CNN)
#
# ARQUITECTURA (Mnih et al., 2015 — DeepMind DQN):
#
#  ENTRADA: (batch, 4, 84, 84) <- 4 frames apilados, 84x84 pixeles
#
#  Conv2d(4->32,  k=8, s=4) -> ReLU -> (batch, 32, 20, 20)
#    Calculo: floor((84-8)/4)+1 = 20
#    Proposito: features globales de bajo nivel (bordes, posicion objetos)
#    Kernel grande 8x8 cubre zonas extensas del campo de juego
#
#  Conv2d(32->64, k=4, s=2) -> ReLU -> (batch, 64, 9, 9)
#    Calculo: floor((20-4)/2)+1 = 9
#    Proposito: features de nivel medio (paleta, pelota, marcador)
#
#  Conv2d(64->64, k=3, s=1) -> ReLU -> (batch, 64, 7, 7)
#    Calculo: floor((9-3)/1)+1 = 7
#    Proposito: relaciones espaciales de alto nivel (posicion pelota-paleta)
#    Stride=1 preserva resolucion espacial fina
#
#  Flatten: (batch, 64*7*7) = (batch, 3136)
#
#  Linear(3136->512) -> ReLU
#    Representacion latente densa. 512 neuronas: estandar DeepMind.
#
#  Linear(512->n_actions) <- Sin activacion (Q-values reales)
#    Q(s,a;theta) para cada accion a en A
#
#  Activacion: ReLU — evita gradiente evanescente, rapida convergencia.
#  Total de parametros: ~1.68M

class DQNNetwork(nn.Module):
    """
    Red Q convolucional. Mapea estados visuales a Q-values.
    Input  : Tensor (batch, 4, 84, 84)
    Output : Tensor (batch, n_actions) <- Q(s,a;theta) sin activacion
    """

    def __init__(self, n_actions):
        """
        Args:
            n_actions: numero de acciones discretas. ALE/Pong-v5: 6 acciones.
        """
        super(DQNNetwork, self).__init__()

        # Bloque Convolucional
        self.conv_block = nn.Sequential(
            # Capa 1: features bajo nivel | salida: (batch, 32, 20, 20)
            nn.Conv2d(in_channels=4,  out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            # Capa 2: features nivel medio | salida: (batch, 64, 9, 9)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            # Capa 3: features alto nivel  | salida: (batch, 64, 7, 7)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        # Bloque Fully Connected
        self.fc_block = nn.Sequential(
            nn.Flatten(),                       # (batch, 3136)
            nn.Linear(64 * 7 * 7, 512),        # representacion latente
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),          # Q(s,a) por accion
        )

    def forward(self, x):
        """
        Propagacion hacia adelante.
        Args:
            x: Tensor (batch, 4, 84, 84) en [0.0, 1.0]
        Returns:
            Tensor (batch, n_actions) — Q-values estimados
        """
        x = self.conv_block(x)
        return self.fc_block(x)

    def count_parameters(self):
        """Numero total de parametros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# SECCION 5: AGENTE DQN

class DQNAgent:
    """
    Agente DQN completo (Mnih et al., 2015):

    - Red Q ONLINE (theta)  : se actualiza con SGD en cada paso.
    - Red Q TARGET (theta-) : copia periódica de theta. Genera los
                              Q-values de referencia (objetivo fijo).
                              Estabiliza el entrenamiento al evitar
                              "perseguir un objetivo móvil".

    ECUACION DE BELLMAN (objetivo de aprendizaje):
        y_i = r_i + gamma * max_{a'} Q(s'_i, a'; theta-)  si not done_i
        y_i = r_i                                          si done_i

    PERDIDA (Huber Loss / SmoothL1):
        L(theta) = E[ HuberLoss(y_i, Q(s_i, a_i; theta)) ]
        donde HuberLoss(delta) = 0.5*delta^2   si |delta| <= 1
                                 |delta| - 0.5  si |delta| > 1
    """

    def __init__(
        self,
        n_actions,
        epsilon_start      = 1.0,
        epsilon_end         = 0.1,
        epsilon_decay_steps= 1_000_000,
        learning_rate      = 1e-4,
        gamma              = 0.99,
        batch_size         = 32,
        target_update_freq = 1_000,
        buffer_capacity    = 50_000,
    ):
        """
        Args:
            n_actions          : numero de acciones del entorno
            epsilon_start      : epsilon inicial (exploracion total)
            epsilon_end        : epsilon minimo (exploracion residual)
            epsilon_decay_steps: pasos para decrecer epsilon linealmente
            learning_rate      : eta para Adam optimizer
            gamma              : factor de descuento gamma en (0,1]
            batch_size         : tamaño del mini-batch de entrenamiento
            target_update_freq : frecuencia (pasos) de sincronizacion theta- <- theta
            buffer_capacity    : tamaño maximo del replay buffer
        """
        self.n_actions           = n_actions
        self.gamma               = gamma
        self.batch_size          = batch_size
        self.target_update_freq  = target_update_freq

        # Parametros de la politica epsilon-greedy
        self.epsilon             = epsilon_start
        self.epsilon_start       = epsilon_start
        self.epsilon_end         = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Redes neuronales
        self.q_online = DQNNetwork(n_actions).to(DEVICE)
        self.q_target = DQNNetwork(n_actions).to(DEVICE)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()  # target NUNCA se entrena directamente

        # Optimizador y perdida
        # Adam: adaptativo, mas robusto que RMSProp en la practica.
        # lr=1e-4: conservador y estable para imagenes.
        # eps=1.5e-4: valor recomendado en el paper de DeepMind.
        self.optimizer = optim.Adam(
            self.q_online.parameters(),
            lr=learning_rate,
            eps=1.5e-4
        )
        # Huber Loss: mas robusta que MSE frente a errores TD grandes
        self.loss_fn = nn.SmoothL1Loss()

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Contador global de pasos
        self.total_steps = 0

        print(f"[INFO] DQNAgent inicializado")
        print(f"  Parametros de la red : {self.q_online.count_parameters():,}")
        print(f"  Dispositivo          : {DEVICE}")
        print(f"  Epsilon: {epsilon_start} -> {epsilon_end} en {epsilon_decay_steps:,} pasos")

    def select_action(self, state):
        """
        Selecciona una acción según la politica epsilon-greedy con
        decaimiento lineal.

        Decaimiento:
            eps(t) = max(eps_end, eps_start - (eps_start-eps_end)*t/T_decay)

        Args:
            state: np.ndarray (4, 84, 84) - estado actual del entorno
        Returns:
            int en [0, n_actions-1] - indice de la acción seleccionada
        """
        # Actualizar epsilon (decaimiento lineal por paso global)
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_end)
            * self.total_steps / self.epsilon_decay_steps
        )

        if random.random() < self.epsilon:
            # EXPLORACION: acción aleatoria uniforme
            return random.randrange(self.n_actions)
        else:
            # EXPLOTACION: acción con mayor Q-value estimado
            state_t = torch.tensor(
                state[np.newaxis, :], dtype=torch.float32
            ).to(DEVICE)
            with torch.no_grad():
                q_vals = self.q_online(state_t)   # (1, n_actions)
            return int(q_vals.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        """Almacena (s, a, r, s', done) en el replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        """
        Una iteración de gradiente descendente con la pérdida Huber.

        Implementa la ecuacion de Bellman:
            y_i = r_i + gamma * max_{a'} Q(s'_i,a';theta-) * (1 - done_i)
            L   = E[ HuberLoss( Q(s_i, a_i; theta), y_i ) ]

        Returns:
            float o None: valor de la pérdida si se actualizó,
                          None durante el warm-up del buffer.
        """
        if not self.replay_buffer.is_ready:
            return None

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        # Q-values actuales: Q(s, a; theta)
        q_all     = self.q_online(states)               # (B, n_actions)
        q_current = q_all.gather(
            1, actions.unsqueeze(1)                     # (B, 1)
        ).squeeze(1)                                    # (B,)

        # Q-values objetivo: y = r + gamma * max_{a'} Q(s'; theta-) -
        with torch.no_grad():
            q_next_max = self.q_target(next_states).max(1)[0]  # (B,)
            # done_i=1.0 anula el valor futuro (estado terminal)
            y = rewards + self.gamma * q_next_max * (1.0 - dones)

        # Perdida y retropropagacion
        loss = self.loss_fn(q_current, y)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: ||grad_theta||_2 <= 10
        # Previene explosion del gradiente con entradas de imagen
        torch.nn.utils.clip_grad_norm_(self.q_online.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Sincronizacion periodica: theta- <- theta
        if self.total_steps % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

        self.total_steps += 1
        return loss.item()

    def save_model(self, filepath):
        """Guarda el estado completo del agente en un checkpoint."""
        torch.save({
            "q_online_state_dict" : self.q_online.state_dict(),
            "q_target_state_dict" : self.q_target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps"         : self.total_steps,
            "epsilon"             : self.epsilon,
        }, filepath)
        print(f"[INFO] Checkpoint guardado -> {filepath}")

    def load_model(self, filepath):
        """
        Carga un checkpoint previamente guardado.
        """
        ckpt = torch.load(filepath, map_location=DEVICE)
        self.q_online.load_state_dict(ckpt["q_online_state_dict"])
        self.q_target.load_state_dict(ckpt["q_target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.total_steps = ckpt["total_steps"]
        self.epsilon     = ckpt["epsilon"]
        print(f"[INFO] Checkpoint cargado <- {filepath}")


# SECCION 6: CICLO DE ENTRENAMIENTO

def train_dqn(
    env_id                = "ALE/Pong-v5",
    n_episodes            = 2000,
    max_steps_per_episode = 10_000,
    save_every            = 100,
    print_every           = 20,
    checkpoint_path        = "/kaggle/working/dqn_pong_checkpoint.pth",
):
    """
    Bucle principal de entrenamiento del agente DQN.

    FLUJO POR EPISODIO:
        s_0 = env.reset()
        for t = 0, 1, 2, ...:
            a_t = pi_eps(s_t)                    # epsilon-greedy
            s_{t+1}, r_t, done = env.step(a_t)
            r_t = clip(r_t, -1, +1)              # reward clipping
            D.push(s_t, a_t, r_t, s_{t+1}, done) # almacenar
            (s,a,r,s') ~ D                       # muestrear batch
            theta <- theta - eta * grad_L(theta)  # actualizar online
            if t % C == 0: theta- <- theta        # sincronizar target
            if done: break

    REWARD CLIPPING: Normalizar a {-1, 0, +1} hace que la misma tasa
    de aprendizaje funcione de forma consistente entre juegos Atari.

    Args:
        env_id                : ID del entorno de Gymnasium
        n_episodes            : numero total de episodios a entrenar
        max_steps_per_episode : limite de pasos por episodio
        save_every            : frecuencia de checkpoints (episodios)
        print_every           : frecuencia de logging en consola
        checkpoint_path       : ruta para guardar checkpoints

    Returns:
        (episode_rewards, episode_losses, agent)
    """
    env       = make_env(env_id)
    n_actions = env.action_space.n

    print(f"\n{'='*65}")
    print(f"  ENTRENAMIENTO DQN — {env_id}")
    print(f"{'='*65}\n")
    print(f"  Forma del estado  : {env.observation_space.shape}")
    print(f"  Numero de acciones: {n_actions}")
    print(f"  Episodios totales : {n_episodes:,}")
    print(f"  Dispositivo       : {DEVICE}")
    print(f"{'='*65}\n")

    # Inicializacion del agente con hiperparametros justificados
    agent = DQNAgent(
        n_actions          = n_actions,
        epsilon_start      = 1.0,       # Exploracion total al inicio
        epsilon_end        = 0.1,       # 10% de exploracion residual
        epsilon_decay_steps= 1_000_000,   # ~250 episodios de 2000 pasos
        learning_rate      = 1e-4,      # Conservador y estable
        gamma              = 0.99,      # Horizonte largo de recompensa
        batch_size         = 32,        # Estandar DeepMind
        target_update_freq = 1_000,     # Actualizar target cada 1000 pasos
        buffer_capacity    = 50_000,   # ~3GB RAM con estados float32
    )

    episode_rewards = []
    episode_losses  = []
    best_avg_reward = -float("inf")
    start_time      = time.time()

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        ep_reward     = 0.0
        ep_loss_vals  = []

        for _ in range(max_steps_per_episode):
            # Seleccionar accion
            action = agent.select_action(state)
            # Ejecutar accion en el entorno
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Clipping de recompensa a {-1, 0, +1}
            reward_clipped = float(np.clip(reward, -1.0, 1.0))

            # Almacenar y aprender
            agent.store_transition(state, action, reward_clipped, next_state, done)
            loss = agent.update()
            if loss is not None:
                ep_loss_vals.append(loss)

            state      = next_state
            ep_reward += reward   # recompensa REAL (no clipeada) para metricas

            if done:
                break

        episode_rewards.append(ep_reward)
        episode_losses.append(
            float(np.mean(ep_loss_vals)) if ep_loss_vals else 0.0
        )

        # Logging
        if episode % print_every == 0:
            window_rews = episode_rewards[-print_every:]
            avg_reward  = float(np.mean(window_rews))
            elapsed     = time.time() - start_time

            print(
                f"Ep {episode:4d}/{n_episodes} | "
                f"Recomp: {ep_reward:6.1f} | "
                f"Prom({print_every}): {avg_reward:6.2f} | "
                f"eps: {agent.epsilon:.4f} | "
                f"Loss: {episode_losses[-1]:.5f} | "
                f"Pasos: {agent.total_steps:,} | "
                f"t: {elapsed/60:.1f}min"
            )

            # Guardar el mejor modelo
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_model("/kaggle/working/dqn_pong_best.pth")

        # Checkpoint periodico
        if episode % save_every == 0:
            agent.save_model(checkpoint_path)

    env.close()
    elapsed_total = time.time() - start_time
    print(f"\n[INFO] Entrenamiento completado en {elapsed_total/3600:.2f} horas")
    print(f"[INFO] Mejor recompensa promedio: {best_avg_reward:.2f}")

    return episode_rewards, episode_losses, agent


# SECCION 7: VISUALIZACION

def moving_average(data, window):
    """Calcula la media movil con convolucion rectangular de anchura window."""
    if len(data) < window:
        return np.array(data)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(data, kernel, mode="valid")


def plot_learning_curve(
    episode_rewards,
    episode_losses,
    window    = 50,
    save_path = "/kaggle/working/learning_curve.png",
):
    """
    Genera y guarda una figura con dos paneles:
      Panel 1: Recompensa por episodio + media movil + referencias humanas
      Panel 2: Huber Loss por episodio + media movil (escala logaritmica)

    La media movil con ventana `window` suaviza el ruido estocastico y
    revela la tendencia real del aprendizaje del agente.

    Args:
        episode_rewards : lista de recompensas totales por episodio
        episode_losses  : lista de perdidas promedio por episodio
        window          : tamaño de la ventana de la media movil
        save_path       : ruta donde guardar la figura (.png)
    """
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=120)
    fig.suptitle(
        "Curva de Aprendizaje — DQN en ALE/Pong-v5",
        fontsize=16, fontweight="bold", color="white", y=1.01,
    )

    rewards = np.array(episode_rewards, dtype=np.float64)
    losses  = np.array(episode_losses,  dtype=np.float64)
    eps     = np.arange(1, len(rewards) + 1)

    # Panel 1: Recompensa
    ax1.plot(eps, rewards, color="#4FC3F7", alpha=0.25, linewidth=0.7,
             label="Recompensa por episodio")
    if len(rewards) >= window:
        ma_r   = moving_average(rewards, window)
        ma_eps = np.arange(window, len(rewards) + 1)
        ax1.plot(ma_eps, ma_r, color="#FF7043", linewidth=2.5,
                 label=f"Media movil ({window} ep)", zorder=5)

    ax1.axhline(y= 15, color="#66BB6A", linestyle="--", linewidth=1.5,
                alpha=0.85, label="Nivel humano estimado (+15)")
    ax1.axhline(y=  0, color="white",   linestyle=":",  linewidth=0.8, alpha=0.4)
    ax1.axhline(y=-21, color="#EF5350", linestyle="--", linewidth=1.0,
                alpha=0.6, label="Peor caso (-21)")

    # Anotar maximo
    idx_max = int(np.argmax(rewards))
    ax1.annotate(
        f"Max: {rewards[idx_max]:.1f}",
        xy=(eps[idx_max], rewards[idx_max]),
        color="#FFD54F", fontsize=9,
        xytext=(8, 6), textcoords="offset points",
    )

    ax1.set_xlabel("Episodio",                     fontsize=12)
    ax1.set_ylabel("Recompensa total del episodio", fontsize=12)
    ax1.set_title("Evolucion de la Recompensa",    fontsize=13, color="#90CAF9")
    ax1.legend(fontsize=9, framealpha=0.3)
    ax1.grid(alpha=0.12, color="gray")
    ax1.set_facecolor("#12122a")

    # Panel 2: Loss
    mask      = losses > 0.0
    eps_loss  = eps[mask]
    loss_vals = losses[mask]

    ax2.plot(eps_loss, loss_vals, color="#CE93D8", alpha=0.3, linewidth=0.7,
             label="Huber Loss por episodio")
    if len(loss_vals) >= window:
        ma_l     = moving_average(loss_vals, window)
        ma_eps_l = eps_loss[window - 1:]
        ax2.plot(ma_eps_l, ma_l, color="#FFA726", linewidth=2.5,
                 label=f"Media movil ({window} ep)", zorder=5)

    ax2.set_xlabel("Episodio",           fontsize=12)
    ax2.set_ylabel("Huber Loss (prom.)", fontsize=12)
    ax2.set_title("Evolucion de la Pérdida (escala log)", fontsize=13, color="#90CAF9")
    ax2.set_yscale("log")
    ax2.legend(fontsize=9, framealpha=0.3)
    ax2.grid(alpha=0.12, color="gray", which="both")
    ax2.set_facecolor("#12122a")

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0d0d1a", edgecolor="none")
    plt.close(fig)
    print(f"[INFO] Grafica guardada -> {save_path}")


def print_training_summary(episode_rewards, episode_losses):
    """Imprime resumen estadístico del entrenamiento."""
    r = np.array(episode_rewards)
    print(f"\n{'='*65}")
    print(f"  RESUMEN ESTADÍSTICO DEL ENTRENAMIENTO")
    print(f"{'='*65}")
    print(f"  Episodios totales        : {len(r):,}")
    print(f"  Recompensa media (total) : {r.mean():.2f}")
    print(f"  Recompensa máxima        : {r.max():.1f}")
    print(f"  Recompensa mínima        : {r.min():.1f}")
    print(f"  Desviacion estandar      : {r.std():.2f}")
    last_n = r[-100:] if len(r) >= 100 else r
    print(f"  Prom. ultimos 100 ep     : {last_n.mean():.2f} +/- {last_n.std():.2f}")
    print(f"{'='*65}\n")


# SECCION 8: EVALUACION DEL AGENTE ENTRENADO

def evaluate_agent(
    agent,
    env_id          = "ALE/Pong-v5",
    n_eval_episodes = 10,
    render          = False,
):
    """
    Evalua el agente en modo GREEDY (epsilon=0.0) — solo explotacion.

    Args:
        agent           : DQNAgent con pesos entrenados
        env_id          : entorno de evaluacion
        n_eval_episodes : numero de episodios de evaluacion
        render          : si True, renderiza el entorno

    Returns:
        list[float]: recompensas de los n_eval_episodes episodios
    """
    env = make_env(env_id, render_mode="human" if render else None)
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0   # modo greedy puro

    eval_rewards = []
    for ep in range(1, n_eval_episodes + 1):
        state, _ = env.reset()
        total_r  = 0.0
        done     = False
        while not done:
            action = agent.select_action(state)
            state, r, term, trunc, _ = env.step(action)
            done    = term or trunc
            total_r += r
        eval_rewards.append(total_r)
        print(f"  Ep eval {ep:2d}/{n_eval_episodes}: Recompensa = {total_r:6.1f}")

    agent.epsilon = saved_epsilon
    env.close()

    mean_r = float(np.mean(eval_rewards))
    std_r  = float(np.std(eval_rewards))
    print(f"\n  Recompensa eval — Media: {mean_r:.2f}  Std: {std_r:.2f}")
    return eval_rewards


# SECCION 9: PUNTO DE ENTRADA

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  DEEP Q-NETWORK — ALE/Pong-v5")
    print("  Maestria en Inteligencia Artificial — Taller 2")
    print("=" * 65 + "\n")

    # 1. Verificar entorno
    print("[VERIFY] Pipeline de preprocesamiento...")
    env_test = make_env()
    obs, _   = env_test.reset()
    print(f"  Forma del estado   : {obs.shape}  (esperado: (4, 84, 84))")
    print(f"  Dtype              : {obs.dtype}   (esperado: float32)")
    print(f"  Rango de valores   : [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  Numero de acciones : {env_test.action_space.n}")
    try:
        print(f"  Nombres acciones   : {env_test.unwrapped.get_action_meanings()}")
    except AttributeError:
        pass
    env_test.close()
    print("[VERIFY] OK Entorno verificado\n")

    # 2. Verificar red neuronal
    print("[VERIFY] Arquitectura de la red DQN...")
    net_test  = DQNNetwork(n_actions=6).to(DEVICE)
    dummy_in  = torch.zeros(1, 4, 84, 84).to(DEVICE)
    dummy_out = net_test(dummy_in)
    print(f"  Input  shape : {tuple(dummy_in.shape)}")
    print(f"  Output shape : {tuple(dummy_out.shape)}  (esperado: (1, 6))")
    print(f"  Parametros   : {net_test.count_parameters():,}")
    del net_test, dummy_in, dummy_out
    print("[VERIFY] OK Red verificada\n")

    # 3. Entrenamiento
    # NOTA: Reducir n_episodes a ~100 para una prueba rapida.
    # El entrenamiento completo (2000 ep) requiere 6-12h en GPU.
    ep_rewards, ep_losses, trained_agent = train_dqn(
        env_id                = "ALE/Pong-v5",
        n_episodes            = 2000,
        max_steps_per_episode = 10_000,
        save_every            = 100,
        print_every           = 20,
        checkpoint_path       = "/kaggle/working/dqn_pong_checkpoint.pth",
    )

    # 4. Visualizacion

    # 4. Visualizacion y guardado
    plot_learning_curve(ep_rewards, ep_losses, window=50,
                        save_path="/kaggle/working/learning_curve.png")
    print_training_summary(ep_rewards, ep_losses)

    # Mostrar curva inline en el notebook (se guarda aunque crashee)
    try:
        from IPython.display import display, Image as IPImage
        display(IPImage("/kaggle/working/learning_curve.png"))
        print("[INFO] Curva mostrada inline en el notebook")
    except Exception:
        pass

    # Forzar guardado final del mejor modelo
    try:
        trained_agent.save_model("/kaggle/working/dqn_pong_best.pth")
        print("[INFO] Modelo final guardado -> /kaggle/working/dqn_pong_best.pth")
    except Exception as e:
        print(f"[WARN] No se pudo guardar modelo final: {e}")

    # Liberar memoria antes de evaluacion
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Evaluacion
    print("[INFO] Evaluando agente entrenado (modo greedy)...")
    try:
        evaluate_agent(trained_agent, n_eval_episodes=10, render=False)
    except Exception as e:
        print(f"[WARN] Error en evaluacion: {e}")

    print("\n[INFO] ¡LISTO! Descarga los archivos desde Output > /kaggle/working/")
    print("  - dqn_pong_best.pth")
    print("  - dqn_pong_checkpoint.pth")
    print("  - learning_curve.png")
