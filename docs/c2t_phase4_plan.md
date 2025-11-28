# C²T Phase 4 训练循环草案

## 组件对接

- **环境**：`CityFlowEnv` + `TrafficMetrics`。每步后获取 `env.get_c2t_state()` 与 `metrics.get_safety_metrics()`。
- **特征提取**：`utils.c2t_features.extract_c2t_features`，输出 `raw_features`；分别使用：
  - `feature_norm.json`（来自 Phase2 或可另行拟合）→ PPO 输入标准化。
  - `reward_norm.json`（Phase2 训练 reward model 时保存）→ intrinsic reward 标准化。
- **奖励模型**：`C2TRewardNet` 以 eval 模式加载，冻结参数。
- **PPOAgent**：复用 `models/ppo_agent.PPOAgent`，每个路口共享参数，action space=相位数。

## 循环流程

1. **采样**  
   - 对每个路口提取 `raw_features`，分别得到：
     - `state_features = apply_normalization(raw_features, ppo_feature_norm)` → 传入 agent。
     - `reward_features = apply_normalization(raw_features, reward_feature_norm)` → 输入 reward net。

2. **动作选择**  
   - `action, log_prob, value = agent.choose_action(state_features)`。
   - 环境执行：`next_state, env_rewards, done, info = env.step(action_dict)`。

3. **奖励混合**  
   - `r_env_norm` 与 `r_intrinsic_norm` 通过各自 running mean/std 标准化。
   - `lambda_t = schedule(episode_idx)`（线性或余弦）。  
   - `mask = 1` 若 `metrics["global"]["min_ttc"] > ttc_threshold`，否则 `0`。  
   - `r_total = r_env_norm + lambda_t * mask * r_intrinsic_norm`（若关闭 intrinsic 则仅 r_env_norm）。

4. **存储**  
   - `agent.store(state_features, action, r_total, log_prob, value, done)`。
   - Episode 结束或步数满足 `update_horizon` 时：
     - `agent.finish_path(last_value)` 并 `agent.update()`。

5. **日志**  
   - TensorBoard/WandB 记录：
     - `reward/env`, `reward/intrinsic`, `reward/total`。
     - `metric/ttc_global`, `metric/harsh_brakes_global`, `metric/red_violations_global`, `metric/queue_global`。
     - Mask 激活率、`lambda_t` 当前值。

## 额外注意

- **多路口**：`env.list_intersection` 中所有路口共享一套 PPO 参数；动作/奖励按 `junction_id` 单独处理。
- **Seeds**：脚本需支持 `--seed`，设置 numpy / torch / random / CityFlow。
- **Checkpoint**：定期保存 PPO 权重、归一化统计（obs、env reward、intrinsic reward）。
- **恢复训练**：预留加载模型与统计参数的入口，为 Phase4 长时间训练做准备。

