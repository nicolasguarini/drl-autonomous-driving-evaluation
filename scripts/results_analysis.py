import pandas as pd

df_highway_dqn = pd.read_csv("../performance_logs/highway-v0_dqn_model_performance.csv")
df_highway_ppo = pd.read_csv("../performance_logs/highway-v0_ppo_model_performance.csv")
df_highway_a2c = pd.read_csv("../performance_logs/highway-v0_a2c_model_performance.csv")

df_roundabout_dqn = pd.read_csv("../performance_logs/roundabout-v0_dqn_model_performance.csv")
df_roundabout_ppo = pd.read_csv("../performance_logs/roundabout-v0_ppo_model_performance.csv")
df_roundabout_a2c = pd.read_csv("../performance_logs/roundabout-v0_a2c_model_performance.csv")

df_merge_dqn = pd.read_csv("../performance_logs/merge-v0_dqn_model_performance.csv")
df_merge_ppo = pd.read_csv("../performance_logs/merge-v0_ppo_model_performance.csv")
df_merge_a2c = pd.read_csv("../performance_logs/merge-v0_a2c_model_performance.csv")

df_intersection_dqn = pd.read_csv("../performance_logs/intersection-v0_dqn_model_performance.csv")
df_intersection_ppo = pd.read_csv("../performance_logs/intersection-v0_ppo_model_performance.csv")
df_intersection_a2c = pd.read_csv("../performance_logs/intersection-v0_a2c_model_performance.csv")

print(" ------- Highway ------- ")
print("Max possible reward: " + str(df_highway_dqn["total_reward"].max()))
print("DQN")
print("Average " + str(df_highway_dqn["episode_length"].mean()) + " episodes")
print("Average " + str(df_highway_dqn["total_reward"].mean()) + " reward")
print("Average " + str(df_highway_dqn["high_speed_reward_sum"].mean()) + " high speed reward")
print("Average " + str(df_highway_dqn["right_lane_reward_sum"].mean()) + " right lane reward")
print("Percentage of crashes: " + str(df_highway_dqn["crashes"].sum()) + "%")

print("PPO")
print("Average " + str(df_highway_ppo["episode_length"].mean()) + " episodes")
print("Average " + str(df_highway_ppo["total_reward"].mean()) + " reward")
print("Average " + str(df_highway_ppo["high_speed_reward_sum"].mean()) + " high speed reward")
print("Average " + str(df_highway_ppo["right_lane_reward_sum"].mean()) + " right lane reward")
print("Percentage of crashes: " + str(df_highway_ppo["crashes"].mean() * 100) + "%")

print("A2C")
print("Average " + str(df_highway_a2c["episode_length"].mean()) + " episodes")
print("Average " + str(df_highway_a2c["total_reward"].mean()) + " reward")
print("Average " + str(df_highway_a2c["high_speed_reward_sum"].mean()) + " high speed reward")
print("Average " + str(df_highway_a2c["right_lane_reward_sum"].mean()) + " right lane reward")
print("Percentage of crashes: " + str(df_highway_a2c["crashes"].mean() * 100) + "%")

print(" ------- Roundabout ------- ")
print("Max possible reward: " + str(df_roundabout_dqn["total_reward"].max()))
print("DQN")
print("Average " + str(df_roundabout_dqn["episode_length"].mean()) + " episodes")
print("Average " + str(df_roundabout_dqn["total_reward"].mean()) + " reward")
print("Average " + str(df_roundabout_dqn["high_speed_reward_sum"].mean()) + " high speed reward")
print("Average " + str(df_roundabout_dqn["right_lane_reward_sum"].mean()) + " right lane reward")
print("Percentage of crashes: " + str(df_roundabout_dqn["crashes"].mean() * 100) + "%")

print("PPO")
print("Average " + str(df_roundabout_ppo["episode_length"].mean()) + " episodes")
print("Average " + str(df_roundabout_ppo["total_reward"].mean()) + " reward")
print("Average " + str(df_roundabout_ppo["high_speed_reward_sum"].mean()) + " high speed reward")
print("Average " + str(df_roundabout_ppo["right_lane_reward_sum"].mean()) + " right lane reward")
print("Percentage of crashes: " + str(df_roundabout_ppo["crashes"].mean() * 100) + "%")

print("A2C")
print("Average " + str(df_roundabout_a2c["episode_length"].mean()) + " episodes")
print("Average " + str(df_roundabout_a2c["total_reward"].mean()) + " reward")
print("Average " + str(df_roundabout_a2c["high_speed_reward_sum"].mean()) + " high speed reward")
print("Average " + str(df_roundabout_a2c["right_lane_reward_sum"].mean()) + " right lane reward")
print("Percentage of crashes: " + str(df_roundabout_a2c["crashes"].mean() * 100) + "%")

print(" ------- Merge ------- ")
print("Max possible reward: " + str(df_merge_dqn["total_reward"].max()))
print("DQN")
print("Average " + str(df_merge_dqn["episode_length"].mean()) + " episodes")
print("Average " + str(df_merge_dqn["total_reward"].mean()) + " reward")
print("Average " + str(df_merge_dqn["high_speed_reward_sum"].mean()) + " high speed reward")
print("Average " + str(df_merge_dqn["right_lane_reward_sum"].mean()) + " right lane reward")
print("Percentage of crashes: " + str(df_merge_dqn["crashes"].mean() * 100) + "%")

print("PPO")
print("Average " + str(df_merge_ppo["episode_length"].mean()) + " episodes")
print("Average " + str(df_merge_ppo["total_reward"].mean()) + " reward")
print("Average " + str(df_merge_ppo["high_speed_reward_sum"].mean()) + " high speed reward")
print("Average " + str(df_merge_ppo["right_lane_reward_sum"].mean()) + " right lane reward")
print("Percentage of crashes: " + str(df_merge_ppo["crashes"].mean() * 100) + "%")

print("A2C")
print("Average " + str(df_merge_a2c["episode_length"].mean()) + " episodes")
print("Average " + str(df_merge_a2c["total_reward"].mean()) + " reward")
print("Average " + str(df_merge_a2c["high_speed_reward_sum"].mean()) + " high speed reward")
print("Average " + str(df_merge_a2c["right_lane_reward_sum"].mean()) + " right lane reward")
print("Percentage of crashes: " + str(df_merge_a2c["crashes"].mean() * 100) + "%")

print(" ------- Intersection ------- ")
print("Max possible reward: " + str(df_intersection_dqn["total_reward"].max()))
print("DQN")
print("Average " + str(df_intersection_dqn["episode_length"].mean()) + " episodes")
print("Average " + str(df_intersection_dqn["total_reward"].mean()) + " reward")
print("Average " + str(df_intersection_dqn["high_speed_reward_sum"].mean()) + " high speed reward")
print("Average " + str(df_intersection_dqn["right_lane_reward_sum"].mean()) + " right lane reward")
print("Percentage of crashes: " + str(df_intersection_dqn["crashes"].mean() * 100) + "%")

print("PPO")
print("Average " + str(df_intersection_ppo["episode_length"].mean()) + " episodes")
print("Average " + str(df_intersection_ppo["total_reward"].mean()) + " reward")
print("Average " + str(df_intersection_ppo["high_speed_reward_sum"].mean()) + " high speed reward")
print("Average " + str(df_intersection_ppo["right_lane_reward_sum"].mean()) + " right lane reward")
print("Percentage of crashes: " + str(df_intersection_ppo["crashes"].mean() * 100) + "%")

print("A2C")
print("Average " + str(df_intersection_a2c["episode_length"].mean()) + " episodes")
print("Average " + str(df_intersection_a2c["total_reward"].mean()) + " reward")
print("Average " + str(df_intersection_a2c["high_speed_reward_sum"].mean()) + " high speed reward")
print("Average " + str(df_intersection_a2c["right_lane_reward_sum"].mean()) + " right lane reward")                 
print("Percentage of crashes: " + str(df_intersection_a2c["crashes"].mean() * 100) + "%")   