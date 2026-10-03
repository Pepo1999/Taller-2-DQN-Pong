# DQN — Entrenamiento de Agente en ALE/Pong-v5 🏓

Este repositorio contiene la implementación y los resultados del entrenamiento de un agente de Reinforcement Learning utilizando **Deep Q-Network (DQN)** para jugar Atari Pong (`ALE/Pong-v5`).

El proyecto es parte del Taller 2 de la Maestría en Inteligencia Artificial.

---

## 1. Descripción del Entorno (Acciones y Observaciones)

El entorno utilizado es `ALE/Pong-v5` de Gymnasium.

### Observaciones (Estado)
El entorno original devuelve imágenes RGB de `210x160` píxeles. Sin embargo, para que la red neuronal procese la información eficientemente (siguiendo el paper original de DeepMind), aplicamos un **pipeline de preprocesamiento**:
1. **Escala de grises**: Se elimina el color (irrelevante para Pong).
2. **Resizing**: La imagen se reduce a `84x84` píxeles.
3. **Normalización**: Los valores de los píxeles se escalan entre `[0, 1]` (float32).
4. **Frame Stacking**: Como un solo frame no contiene información sobre la velocidad o dirección de la pelota, **se apilan los últimos 4 frames**.
* **Estado Final:** Un tensor de dimensiones `(4, 84, 84)`.

### Acciones
El agente puede tomar 6 acciones discretas, aunque en la práctica para Pong se reducen a 3 movimientos útiles:
- `0`: NOOP (No hacer nada)
- `1`: FIRE (Disparar/Iniciar juego)
- `2`: RIGHT (Subir pala)
- `3`: LEFT (Bajar pala)
- `4`: RIGHTFIRE (Subir y disparar)
- `5`: LEFTFIRE (Bajar y disparar)

---

## 2. Flujo Lógico del Entrenamiento y Particularidades

El entrenamiento sigue el ciclo clásico de DQN con Experience Replay y una Red Target.

1. **Interacción y Recolección:** El agente observa el estado apilado `s` y selecciona una acción `a` usando una política $\epsilon$-greedy (explora aleatoriamente o explota la mejor acción según la red).
2. **Replay Buffer:** La transición `(s, a, r, s', done)` se guarda en un Replay Buffer. Para evitar problemas de memoria (OOM), los frames se almacenan como `uint8` en el buffer y se convierten a `float32` normalizado al momento de muestrear (reduciendo el uso de RAM un 75%).
3. **Muestreo (Mini-batch):** Se extrae un lote aleatorio de transiciones del buffer para romper la correlación temporal entre frames consecutivos, un problema crítico en los juegos de Atari.
4. **Actualización de la Red (Loss):** Se calcula el valor Q actual con la Red Online y el valor Q objetivo con la Red Target usando la ecuación de Bellman. Se minimiza la pérdida (Huber Loss para estabilidad) mediante descenso de gradiente (Adam).
5. **Sincronización:** Cada $N$ pasos, los pesos de la Red Online se copian a la Red Target.

**Particularidades en Pong:**
- Las recompensas en Pong son ralas (*sparse*). El agente recibe `+1` solo cuando anota un punto y `-1` cuando le anotan. Un episodio completo va a 21 puntos.
- Requiere un calentamiento largo (*warm-up*) del Replay Buffer antes de que la red empiece a aprender patrones reales.

---

## 3. Arquitectura de la Red Neuronal

La red sigue la arquitectura convolucional (CNN) diseñada por Mnih et al. (2015) para Atari:

* **Entrada:** Tensor `(Batch, 4, 84, 84)`
* **Conv1:** 32 filtros de $8\times8$, stride 4, activación ReLU $\rightarrow$ Salida: `(32, 20, 20)`
* **Conv2:** 64 filtros de $4\times4$, stride 2, activación ReLU $\rightarrow$ Salida: `(64, 9, 9)`
* **Conv3:** 64 filtros de $3\times3$, stride 1, activación ReLU $\rightarrow$ Salida: `(64, 7, 7)`
* **Flatten:** Aplanar el tensor espacial a un vector de tamaño `3136` ($64 \times 7 \times 7$).
* **Fully Connected 1:** Capa lineal de 512 neuronas con ReLU.
* **Fully Connected 2 (Salida):** Capa lineal con 6 neuronas de salida (sin activación final), representando el Valor $Q$ para cada una de las 6 acciones posibles.

**Total de parámetros:** 1,687,206.

---

## 4. Resultados del Entrenamiento

El agente fue entrenado en GPU (NVIDIA T4 x2 a través de Kaggle) durante **10.77 horas**, completando **2,000 episodios**.

| Métrica | Valor |
|---|---|
| Mejor Recompensa Promedio (Train) | +0.38 |
| Promedio últimos 100 episodios | +5.89 ± 6.99 |
| Recompensa Máxima Alcanzada | **+21.0 (Victoria Perfecta)** |
| Recompensa Media (Evaluación Greedy) | **+9.10** |
| Desviación Std (Evaluación) | 8.31 |

*(Se anexa en el repositorio el gráfico `learning_curve.png` con la evolución de la recompensa y la pérdida).*

---

## 5. Reflexión de Resultados

Los resultados demuestran que el algoritmo **DQN fue implementado correctamente y convergió**. 
- Durante los primeros 500-800 episodios, el agente oscilaba en recompensas de -21 a -19, perdiendo constantemente debido a la alta tasa de exploración ($\epsilon$).
- A medida que el Epsilon Decay hizo efecto y el Replay Buffer se nutrió de mejores experiencias, el agente descubrió que devolver la pelota aumentaba sus probabilidades de supervivencia.
- En la fase final (episodios 1500-2000), el agente logró victorias perfectas de `+21`, dominando el posicionamiento de la pala.
- La evaluación en modo *greedy* (sin exploración aleatoria) arrojó un promedio positivo de `+9.10`, confirmando que la política aprendida es superior a la del bot interno de Atari.

---

## 6. Reflexión sobre lo que más costó

El mayor reto de esta práctica fue la **gestión de memoria RAM y la infraestructura de entrenamiento**:
1. **Out of Memory (OOM):** Inicialmente, el Replay Buffer con capacidad para 100,000 estados (en `float32`) colapsaba la memoria del entorno de Kaggle/Colab (>20GB RAM). La solución técnica más importante fue refactorizar el código para almacenar los frames en memoria como `uint8` (1 byte por píxel) y realizar la conversión a `float32` y normalización (`/255.0`) *justo en el momento del muestreo*. Esto redujo el consumo de RAM en un 75%.
2. **Tiempos de Entrenamiento:** Correr este algoritmo en CPU resultó prohibitivo (estimado de ~30 horas para 2000 episodios). Migrar el código para ser ejecutado en la nube con aceleración por hardware (Kaggle GPUs) requirió adaptar rutas de archivos y lidiar con los límites de tiempo de sesión.
3. **Hiperparámetros:** Ajustar la tasa de caída de epsilon (`epsilon_decay_steps`) para equilibrar el tiempo limitado de entrenamiento con la necesidad de explorar suficientemente el entorno de Pong, el cual es fuertemente dependiente de encontrar las recompensas raras al inicio.

---

### Archivos en este repositorio
- `DQN_PONG_V2.py`: Script principal de entrenamiento y arquitectura.
- `ver_agente_jugar.py`: Script de inferencia para evaluar y visualizar visualmente al agente entrenado.
- `dqn_pong_best.pth`: Pesos de la mejor red neuronal obtenida (PyTorch).
- `dqn_pong_checkpoint.pth`: Punto de guardado del último estado del entrenamiento.
- `learning_curve.png`: Gráfica del rendimiento.
