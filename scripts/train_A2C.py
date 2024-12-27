from multiprocessing import freeze_support
import gymnasium
import highway_env
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import pandas as pd
import os

ENV_TYPE = "merge-v0"
N_TRAIN_STEPS = 40000

os.makedirs("../training_logs", exist_ok=True)

class TrainingAnalysisCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingAnalysisCallback, self).__init__(verbose)
        self.data = []

    def _on_step(self) -> bool:
        if "episode" in self.locals["infos"][0]:
            episode_info = self.locals["infos"][0]["episode"]
            self.data.append({
                "episode": len(self.data) + 1,
                "reward": episode_info["r"],
                "length": episode_info["l"],
                "time": episode_info["t"],
            })
            print(f"Episode {len(self.data)} finished. Reward: {episode_info['r']:.2f}, Length: {episode_info['l']}")

        return True

    def _on_training_end(self) -> None:
        df = pd.DataFrame(self.data)
        df.to_csv(f"../training_logs/train_a2c_{ENV_TYPE}.csv", index=False)
        print("Training data saved to training_logs/train_a2c-highway-v0.csv")

vec_env = make_vec_env(ENV_TYPE, n_envs=4)

model = A2C('MlpPolicy', vec_env,
            policy_kwargs=dict(net_arch=[256, 256]),  # Architettura della rete neurale
            learning_rate=5e-4,                       # Tasso di apprendimento
            n_steps=5,                                # Numero di passi prima di aggiornare il modello
            gamma=0.8,                                # Fattore di sconto
            gae_lambda=0.95,                          # Lambda per GAE (Generalized Advantage Estimation)
            verbose=0,                                # Livello di verbosità
            tensorboard_log="../tensorboard_logs/", # Percorso per i log di TensorBoard
            device="cpu"                             # Dispositivo di elaborazione delle tensori
            )

callback = TrainingAnalysisCallback(verbose=1)
model.learn(N_TRAIN_STEPS, progress_bar=True, callback=callback)

model.save(f"../models/{ENV_TYPE}_A2C_model")

