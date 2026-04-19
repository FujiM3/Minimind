# Mario PPO 数据采集任务报告

## 任务结论

已完成一个可直接使用的采集脚本：  
**`scripts/mario/collect_ppo_dataset.py`**

该脚本可读取 `PPO_trained_models` 下各关卡 PPO 权重，在 Mario 环境中进行**非贪心采样**并落盘轨迹数据（包含错误动作样本），同时自动给出 Decision Transformer 参数规模建议。

---

## 1. 已实现内容

### 1.1 PPO agent 数据采集脚本
- 输入：`PPO_trained_models/ppo_super_mario_bros_{world}_{stage}`
- 自动解析关卡编号并匹配环境 `SuperMarioBros-{world}-{stage}-v3`
- 重建 PPO 网络结构（与 checkpoint 键匹配）：
  - `conv1..conv4`
  - `linear`
  - `critic_linear`
  - `actor_linear`
- 支持多关卡批量采集与每关多 episode。

### 1.2 非贪心策略（避免过拟合）
脚本默认启用两层随机性：
1. **温度采样**（`temperature`，默认 `1.10`）  
2. **epsilon 随机动作注入**（`epsilon`，默认 `0.10`）

动作并非 argmax（贪心），可保留失败轨迹与次优行为，提升 DT 对分布外状态的鲁棒性。

### 1.3 轨迹数据格式
每个 episode 落盘为 `.npz`，字段包括：
- `obs`, `next_obs`（`[T,4,84,84]`）
- `actions`, `rewards`, `dones`
- `returns_to_go`
- `log_probs`, `values`
- `random_action`（是否 epsilon 随机动作）
- `action_entropy`
- `x_pos`, `flag_get`

并输出汇总文件：
- `collection_summary.json`

---

## 2. 已执行的实际测试（Smoke Test）

运行命令（已执行）：

```powershell
.\DT_env\Scripts\python.exe scripts\mario\collect_ppo_dataset.py --max_levels 2 --episodes_per_level 1 --max_steps 400 --output_dir dataset\ppo_mario_rollouts_smoke --epsilon 0.15 --temperature 1.2
```

结果：
- 采集 checkpoint 数：2
- 采集 episode 数：2
- 总步数：155
- 平均随机动作比例：`0.1707`
- 平均策略熵：`1.6339`
- 数据文件已生成：
  - `dataset\ppo_mario_rollouts_smoke\world_1_stage_1\ep_0000.npz`
  - `dataset\ppo_mario_rollouts_smoke\world_1_stage_2\ep_0000.npz`
  - `dataset\ppo_mario_rollouts_smoke\collection_summary.json`

---

## 3. Transformer 参数量适配评估

### 3.1 PPO 数据规模观察
- 单个 PPO checkpoint 约 `2.5MB`
- 共 `31` 个关卡模型
- 当前 smoke 数据总 token（按 DT 三元 token 估算）较小，因此推荐小模型起步。

### 3.2 当前建议
基于采集统计，脚本自动给出的建议为：
- **推荐档位：small**
- **推荐配置：**
  - `hidden_size=256`
  - `num_hidden_layers=4`
  - `num_attention_heads=8`

这与当前 `model_decision_transformer` 轻量配置方向一致，适合先用 PPO 采集数据做 BC/DT 训练，避免参数远大于数据规模造成过拟合。

### 3.3 何时升配
当你把采集扩展到全关卡、每关多 episode 后，若 `collection_summary.json` 的 `total_steps` 明显增大（例如 > 数十万 step），再升到：
- `hidden_size=384, layers=6`（中档）
或
- `hidden_size=512, layers=8`（基准档）

---

## 4. 推荐正式采集命令

```powershell
.\DT_env\Scripts\python.exe scripts\mario\collect_ppo_dataset.py --ckpt_dir PPO_trained_models --output_dir dataset\ppo_mario_rollouts --episodes_per_level 5 --max_steps 4000 --epsilon 0.10 --temperature 1.10 --gamma 0.99
```

---

## 5. 输出产物清单

1. `scripts/mario/collect_ppo_dataset.py`（新增）
2. `dataset/ppo_mario_rollouts_smoke/*`（测试数据）
3. `dataset/ppo_mario_rollouts_smoke/collection_summary.json`（统计与建议）
4. `mario_ppo_data_collection_report.md`（本报告）

---

## 6. 根因定位与修复（关键）

### 6.1 现象
- 在主工程采集/rollout 流程中，1-1 出现 `0/100` 通关。
- 但在 `legacy_ppo_env + 参考仓库原生 test 风格` 下，1-1 可 `100/100` 通关。

### 6.2 根因
- 主工程自定义推理网络 `CustomPPOPolicyNet.forward` 与原仓库 `src/model.py` 存在关键差异：
  - **错误实现**：`x = F.relu(self.linear(x))`
  - **正确实现**：`x = self.linear(x)`
- 这层额外 ReLU 改变了策略头输入分布，导致动作选择系统性偏移，最终表现为“稳定但错误”的固定失败轨迹。

### 6.3 修复
- 文件：`scripts/mario/collect_dt_dataset_from_ppo.py`
- 修复点：移除 `linear` 后额外 ReLU，恢复与原仓库推理路径一致。
- 相关提交：
  - `a226aeb`：修复采集脚本 PPO 推理不一致问题
  - `4288d8f`：同步提交 rollout 可视化脚本对齐版本

---

## 7. 修复后复现实验结果（贪心，epsilon=0）

### 7.1 1-1 复核
- `visualize_ppo_rollout.py`：`5/5` 通关
- `collect_dt_dataset_from_ppo.py`：`100/100` 通关

### 7.2 多关卡复现（每关 100 条）

| Level | Episodes | Clear (any) | Clear (terminal) | Clear Rate | Mean Return | Mean Length |
|---|---:|---:|---:|---:|---:|---:|
| 1-1 | 100 | 100 | 100 | 1.000 | 312.95 | 321.0 |
| 1-2 | 100 | 100 | 100 | 1.000 | 291.05 | 327.0 |
| 2-1 | 100 | 100 | 100 | 1.000 | 319.75 | 341.0 |

对应数据文件：
- `dataset/aligned_greedy/default_profile_100eps_after_fix.pkl`
- `dataset/aligned_greedy/default_profile_1_2_100eps_after_fix.pkl`
- `dataset/aligned_greedy/default_profile_2_1_100eps_after_fix.pkl`

---

## 8. 当前结论

- “官方权重不行”不是根因。
- 真实根因是**推理网络实现与原始 PPO 网络不一致**（多了 ReLU）。
- 修复后，主工程在 legacy 对齐配置下可稳定复现通关能力，并已在多关卡上验证。
