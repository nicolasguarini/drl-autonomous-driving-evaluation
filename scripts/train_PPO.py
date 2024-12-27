import gymnasium
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
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
        df.to_csv(f"../training_logs/train_ppo_{ENV_TYPE}.csv", index=False)
        print("Training data saved to training_logs/training_data.csv")

env = gymnasium.make(ENV_TYPE)

model = PPO('MlpPolicy', env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.8,
            n_epochs=10,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0,
            tensorboard_log="../tensorboard_logs/"
            )

callback = TrainingAnalysisCallback(verbose=1)
model.learn(N_TRAIN_STEPS, progress_bar=True, callback=callback)

model.save(f"../models/{ENV_TYPE}_PPO_model")