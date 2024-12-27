import gymnasium
import highway_env
from stable_baselines3 import PPO
import pandas as pd

ENV_TYPE = "intersection-v0"
MODEL_PATH = f"../models/{ENV_TYPE}_A2C_model"
EP_COUNT = 100

model = PPO.load(MODEL_PATH)
env = gymnasium.make(ENV_TYPE, render_mode="human")

data_collection = []

for i in range(EP_COUNT):
    done = truncated = False
    obs, info = env.reset()
    episode_reward = 0
    episode_length = 0
    crashes = 0
    collision_reward_sum = 0
    right_lane_reward_sum = 0
    high_speed_reward_sum = 0
    on_road_reward_sum = 0

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1

        crashes += 1 if info.get("crashed", False) else 0
        collision_reward_sum += info["rewards"].get("collision_reward", 0)
        right_lane_reward_sum += info["rewards"].get("right_lane_reward", 0)
        high_speed_reward_sum += info["rewards"].get("high_speed_reward", 0)
        on_road_reward_sum += info["rewards"].get("on_road_reward", 0)

        env.render()

    data_collection.append({
        "episode": i + 1,
        "total_reward": episode_reward,
        "episode_length": episode_length,
        "crashes": crashes,
        "avg_speed": info.get("speed", 0) / episode_length,
        "collision_reward_sum": collision_reward_sum,
        "right_lane_reward_sum": right_lane_reward_sum,
        "high_speed_reward_sum": high_speed_reward_sum,
        "on_road_reward_sum": on_road_reward_sum
    })
    print(f"Episode {i + 1} finished. Total reward: {episode_reward}, Length: {episode_length}, Crashes: {crashes}")


df = pd.DataFrame(data_collection)
df.to_csv(f"../performance_logs/{ENV_TYPE}_a2c_model_performance.csv", index=False)
print(f"Performance data saved to {ENV_TYPE}_a2c_model_performance.csv")
print(df.describe())