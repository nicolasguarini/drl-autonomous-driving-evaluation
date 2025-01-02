# Evaluation of Deep Reinforcement Learning methods in Autonomous Driving tasks

- [Literature Review: Modern Deep Reinforcement Learning approaches for Autonomous Driving](./pdf/Literature_Review_of_Reinforcement_Learning_for_Autonomous_Driving.pdf);

- [Project Report: Evaluation of Deep Reinforcement Learning methods in Autonomous Driving](./pdf/Project_Report___Reinforcement_Learning_for_Autonomous_Driving.pdf).

---

## Abstract
This project evaluates the effectiveness of three
Deep Reinforcement Learning (DRL) methods, Deep Q-Networks
(DQN), Proximal Policy Optimization (PPO), and Advantage
Actor-Critic (A2C), in addressing autonomous driving challenges.
Using a customizable simulation environment, the agents were
trained and tested across four diverse driving scenarios: highway,
roundabout, merge, and intersection. The analysis focused on both
the training process (e.g., reward progression) and the post-training performance of the models, evaluating metrics such
as total reward, collision rate, and driving behavior realism.
Results showed that PPO generally achieved the best overall
performance in terms of efficiency and realism. However, DQN
delivered results that were often comparable or only slightly
inferior to PPO, demonstrating robustness in various scenarios.
A2C, while effective in some cases, struggled with consistency
and adaptability

## Installation

```bash
pip install -r requirements.txt
```

## Agents' Training

### Highway-V0
Episode Length: <br>
<img src="./img/highway/ep_len_mean.png" width="500">

Reward: <br>
<img src="./img/highway/ep_rew_mean.png" width="500">

Fps:<br>
<img src="./img/highway/fps.png" width="500">

Entropy Loss:<br>
<img src="./img/highway/entropy_loss.png" width="500">

Explained Variance:<br>
<img src="./img/highway/explained_variance.png" width="500">

Loss:<br>
<img src="./img/highway/loss.png" width="500">

Value Loss:<br>
<img src="./img/highway/value_loss.png" width="500">

### Roundabout-V0

Episode Length: <br>
<img src="./img/roundabout/ep_len_mean.png" width="500">

Reward: <br>
<img src="./img/roundabout/ep_rew_mean.png" width="500">

Fps:<br>
<img src="./img/roundabout/fps.png" width="500">

Entropy Loss:<br>
<img src="./img/roundabout/entropy_loss.png" width="500">

Explained Variance:<br>
<img src="./img/roundabout/explained_variance.png" width="500">

Loss:<br>
<img src="./img/roundabout/loss.png" width="500">

Value Loss:<br>
<img src="./img/roundabout/value_loss.png" width="500">

### Merge-V0

Episode Length: <br>
<img src="./img/merge/ep_len_mean.png" width="500">

Reward: <br>
<img src="./img/merge/ep_rew_mean.png" width="500">

Fps:<br>
<img src="./img/merge/fps.png" width="500">

Entropy Loss:<br>
<img src="./img/merge/entropy_loss.png" width="500">

Explained Variance:<br>
<img src="./img/merge/explained_variance.png" width="500">

Loss:<br>
<img src="./img/merge/loss.png" width="500">

Value Loss:<br>
<img src="./img/merge/value_loss.png" width="500">

### Intersection-V0

Episode Length: <br>
<img src="./img/intersection/ep_len_mean.png" width="500">

Reward: <br>
<img src="./img/intersection/ep_rew_mean.png" width="500">

Fps:<br>
<img src="./img/intersection/fps.png" width="500">

Entropy Loss:<br>
<img src="./img/intersection/entropy_loss.png" width="500">

Explained Variance:<br>
<img src="./img/intersection/explained_variance.png" width="500">

Loss:<br>
<img src="./img/intersection/loss.png" width="500">

Value Loss:<br>
<img src="./img/intersection/value_loss.png" width="500">
