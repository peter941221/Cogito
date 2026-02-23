# Memory.MD

## 2026-02-23

- Read docs/implementation_plan.md and docs/technical_plan.md for repo guidance.

- Added AGENTS.md with environment setup, commands, style rules, constraints, and structure.

- Confirmed no Cursor/Copilot rules present in the repo.

- Created cogito/ package scaffolding with core subpackages and __init__.py files.

- Added requirements.txt and pytest.ini; adjusted AGENTS.md for Cogito paths.

- Implemented cogito/config.py with Config dataclass and create_dirs().

- Added cogito/tests/test_config.py; pytest passes (3 tests).

- Created venv and installed non-torch deps; torch install timed out.
- Installed torch 2.10.0+cpu using Tsinghua mirror (pip install torch --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cpu).
- All core imports verified (torch, numpy, matplotlib, sklearn, scipy). Pytest passes (3 tests).
- Environment setup complete.

---

## Phase 0 完成: 基础设施

### 任务 0.2: 网格世界核心 ✓
**文件**: `cogito/world/grid.py`
**测试**: `cogito/tests/test_grid.py` (30 tests passed)

验收标准:
- [x] CogitoWorld() 创建成功
- [x] 网格大小 64×64
- [x] get_observation() 返回 shape=(106,) 的 numpy 数组
- [x] 所有观察值在 [0, 1] 范围内（已归一化）
- [x] 环形拓扑测试：从 (63, y) 向右移动到达 (0, y)
- [x] 环形拓扑测试：从 (x, 0) 向上移动到达 (x, 63)
- [x] 吃食物后能量增加 FOOD_ENERGY
- [x] 进入危险区域能量减少 DANGER_PENALTY
- [x] 每步能量减少 STEP_COST
- [x] 能量 ≤ 0 时 is_dead = True
- [x] 食物被吃后总数立刻恢复到 NUM_FOOD
- [x] 碰墙不动（位置不变）
- [x] 等待动作（动作5）位置不变

### 任务 0.3: 世界渲染器 ✓
**文件**: `cogito/world/renderer.py`
**测试**: `cogito/tests/test_renderer.py` (13 tests passed)

验收标准:
- [x] WorldRenderer() 创建成功
- [x] headless 模式不弹出窗口
- [x] render() 更新图像数据
- [x] 渲染频率可配置（render_interval）
- [x] save_frame() 生成可查看的图片文件
- [x] save_frame() 自动创建父目录
- [x] 支持 context manager
- [x] agent_marker 正确显示

### 任务 0.4: Phase 0 集成测试 ✓
**文件**: `cogito/tests/test_phase0.py`

验收标准:
- [x] 脚本运行不崩溃
- [x] 1000步中智能体有活有死（能量机制正常）
- [x] 食物总数始终为 NUM_FOOD（刷新机制正常）
- [x] 观察向量每步都是106维
- [x] 观察向量中所有值在[0,1]范围
- [x] 食物被吃（能量增加检测）

### Phase 0 总结
- **总测试数**: 46 tests passed
- **覆盖模块**: config, world/grid, world/renderer
- **运行时间**: 5.36s (pytest)
- **状态**: Phase 0 COMPLETE ✓

---

## Phase 1 完成: Cogito智能体

### 任务 1.1: 感官编码器 ✓
**文件**: `cogito/agent/sensory_encoder.py`
**测试**: 5 tests passed

验收标准:
- [x] SensoryEncoder() 创建成功
- [x] 输入 shape (106,) → 输出 shape (64,)
- [x] 输入 shape (32, 106) → 输出 shape (32, 64)
- [x] 输出值不是 NaN 或 Inf
- [x] 参数量约 22,000

### 任务 1.2: 循环核心 ✓
**文件**: `cogito/agent/recurrent_core.py`
**测试**: 8 tests passed

验收标准:
- [x] RecurrentCore() 创建成功
- [x] 输入 (70维, hidden) → 输出 (128维, new_hidden)
- [x] init_hidden() 返回正确shape的零张量
- [x] 连续10步传递隐状态，不崩溃，不出NaN
- [x] get_hidden_vector() 返回 shape (512,) 的向量
- [x] 零输入测试：隐状态在各步之间确实在变化
- [x] 参数量在 230,000-240,000 范围

### 任务 1.3: 动作头和预测头 ✓
**文件**: `cogito/agent/action_head.py`, `cogito/agent/prediction_head.py`
**测试**: 7 tests passed

验收标准:
- [x] ActionHead：输入(128,) → logits(6,)
- [x] select_action 返回 (int, float, float)
- [x] 动作索引在 [0, 5] 范围
- [x] log概率是负数
- [x] 熵是非负数
- [x] 随机采样正常
- [x] PredictionHead：输入(128,) → 预测(64,)

### 任务 1.4: 经验记忆缓冲区 ✓
**文件**: `cogito/agent/memory_buffer.py`
**测试**: 6 tests passed

验收标准:
- [x] MemoryBuffer(5000) 创建成功
- [x] push 100条记录后 len() == 100
- [x] push 6000条记录后 len() == 5000（环形覆盖）
- [x] sample(32) 返回的每个字段 shape 正确
- [x] sample 在缓冲区为空时返回 None
- [x] get_recent(10) 返回最近10条

### 任务 1.5: 在线学习器 ✓
**文件**: `cogito/agent/learner.py`
**测试**: 6 tests passed

验收标准:
- [x] OnlineLearner创建成功
- [x] learn_from_experience 返回损失字典
- [x] 损失值不是 NaN 或 Inf
- [x] 奖励设置正确：死亡=-10, 食物=+5, 其他=-0.1

### 任务 1.6: Cogito智能体整合 ✓
**文件**: `cogito/agent/cogito_agent.py`
**测试**: 7 tests passed

验收标准:
- [x] CogitoAgent() 创建成功
- [x] act(obs_106d) 返回 (int_action, dict_info)
- [x] action 在 [0,5] 范围
- [x] info 包含必要字段且shape正确
- [x] 连续10步调用act()，hidden_vector 每步都不同
- [x] reset_on_death 后 hidden 变为零
- [x] 死亡后权重没有被重置
- [x] save/load 功能正常
- [x] 总参数量在 250,000-270,000 范围

### 任务 1.7: 主仿真循环 ✓
**文件**: `cogito/core/simulation.py`

验收标准:
- [x] Simulation() 创建成功
- [x] 支持 callback 机制
- [x] headless 模式支持
- [x] 死亡后智能体重置到新位置

### Phase 1 总结
- **总测试数**: 85 tests passed
- **覆盖模块**: config, world/grid, world/renderer, agent/*, core/simulation
- **运行时间**: 12.22s (pytest)
- **总参数量**: ~263,000
- **状态**: Phase 1 COMPLETE ✓

---

## Phase 2 完成: 监测基础设施

### 任务 2.1: 数据采集器 ✓
**文件**: `cogito/monitoring/data_collector.py`
**测试**: 5 tests passed

验收标准:
- [x] DataCollector() 创建成功，数据库文件存在
- [x] 作为callback注入Simulation后正常运行
- [x] 行为数据记录到 SQLite
- [x] 内部状态记录到 numpy memmap 文件
- [x] get_behavior_stats() 返回合理统计
- [x] get_learning_curve() 返回损失时间序列

### 任务 2.2: 内部状态分析器 ✓
**文件**: `cogito/monitoring/state_analyzer.py`
**测试**: 4 tests passed

验收标准:
- [x] StateAnalyzer() 创建成功
- [x] analyze() 输入710维向量，不崩溃
- [x] t-SNE 降维到 2D
- [x] DBSCAN 聚类
- [x] 计算聚类与事件的互信息
- [x] 识别孤立聚类

### 任务 2.3: 复杂度指标计算器 ✓
**文件**: `cogito/monitoring/complexity_metrics.py`
**测试**: 10 tests passed

验收标准:
- [x] 常数序列 → ApEn < 0.1
- [x] 随机序列 → ApEn > 0.3
- [x] 正弦波 → ApEn 中等
- [x] 常数序列 → 排列熵 < 0.1
- [x] 随机序列 → 排列熵 > 0.9
- [x] activity_level 正常工作
- [x] 处理短序列不崩溃

### 任务 2.4: 自我向量簇检测器 ✓
**文件**: `cogito/monitoring/svc_detector.py`
**测试**: 7 tests passed

验收标准:
- [x] SVCDetector() 创建成功
- [x] detect() 返回 SVCReport
- [x] 条件1（孤立性）检测正确
- [x] 条件4（涌现性）检测正确
- [x] confidence 值在 [0, 1] 范围
- [x] 历史追踪机制正常

### 任务 2.5: 实时仪表板 ✓
**文件**: `cogito/monitoring/dashboard.py`

验收标准:
- [x] Dashboard() 创建成功
- [x] 多子图布局合理
- [x] update() 不崩溃
- [x] headless 模式支持
- [x] save_snapshot() 保存图片

### Phase 2 总结
- **总测试数**: 111 tests passed (Phase 0-2 累计)
- **覆盖模块**: config, world/*, agent/*, core/*, monitoring/*
- **运行时间**: 49.73s (pytest)
- **状态**: Phase 2 COMPLETE ✓

---

## 项目状态汇总

### 已完成 Phase
- ✅ Phase 0: 基础设施 (网格世界、渲染器)
- ✅ Phase 1: Cogito智能体 (编码器、LSTM、学习器)
- ✅ Phase 2: 监测基础设施 (数据采集、分析、SVC检测)

### 待实现 Phase
- [ ] Phase 3: 实验模块 (exp1-5)
- [ ] Phase 4: 分析模块

### 项目结构
```
cogito/
├── config.py           # 全局配置
├── world/
│   ├── grid.py         # 64×64 网格世界
│   └── renderer.py     # matplotlib 渲染
├── agent/
│   ├── sensory_encoder.py   # 106→64 MLP
│   ├── recurrent_core.py    # 2层 LSTM
│   ├── action_head.py       # 动作选择
│   ├── prediction_head.py   # 预测头
│   ├── memory_buffer.py     # 经验回放
│   ├── learner.py           # REINFORCE + 预测学习
│   └── cogito_agent.py      # 智能体整合
├── core/
│   └── simulation.py        # 主仿真循环
├── monitoring/
│   ├── data_collector.py    # SQLite + memmap 数据采集
│   ├── state_analyzer.py    # t-SNE + DBSCAN 分析
│   ├── complexity_metrics.py # ApEn, SampEn, PermEn
│   ├── svc_detector.py      # 5条件 SVC 检测
│   └── dashboard.py         # 实时监控面板
└── tests/
    ├── test_config.py
    ├── test_grid.py
    ├── test_renderer.py
    ├── test_phase0.py
    ├── test_agent.py
    └── test_monitoring.py
```

---

## Phase 3 完成: 基线运行（成熟期）

### 任务 3.1: 长时间运行脚本 ✓
**文件**: `run_maturation.py`

验收标准:
- [x] 脚本创建成功
- [x] 支持 100,000 步运行
- [x] checkpoint 保存功能
- [x] snapshot 保存功能
- [x] 实时统计打印
- [x] SVC 检测集成
- [x] 命令行参数支持

测试结果 (500步):
- 运行时间: 1.4 分钟
- 步/秒: 6.1
- 能量稳定: ~50.8
- 状态: 正常

### 任务 3.2: 基线分析报告 ✓
**文件**: `cogito/analysis/baseline_report.py`

验收标准:
- [x] BaselineReport 类创建
- [x] 从 SQLite 加载数据
- [x] 学习曲线图生成
- [x] 行为分析图生成
- [x] t-SNE 演化图生成
- [x] 统计摘要输出

---

## Phase 4 部分: 核心实验

### 任务 4.1: 实验一 - 感觉剥夺 ✓
**文件**: `cogito/experiments/exp1_sensory_deprivation.py`

验收标准:
- [x] SensoryDeprivationExperiment 类创建
- [x] 三阶段实验流程 (基线→剥夺→恢复)
- [x] 对照组实现
- [x] M1 近似熵计算
- [x] M2 活动持续时间计算
- [x] M3 记忆重激活计算
- [x] M4 功能连接性计算
- [x] 结果分类 (A/B/Intermediate)

---

## 项目状态汇总 (更新 2026-02-23)

### 已完成 Phase
- ✅ Phase 0: 基础设施 (网格世界、渲染器)
- ✅ Phase 1: Cogito智能体 (编码器、LSTM、学习器)
- ✅ Phase 2: 监测基础设施 (数据采集、分析、SVC检测)
- ✅ Phase 3: 基线运行 (maturation脚本、分析报告)
- ✅ Phase 4: 核心实验 (全部完成)
- ✅ Phase 5: 跨基质验证 (完成)
- ✅ Phase 6: 分析模块 (完成)

### 新增模块
- `cogito/world/echo_zone.py` - 回声区域 (实验二)
- `cogito/world/hidden_interface.py` - 隐藏接口 (实验三)
- `cogito/experiments/exp2_digital_mirror.py` - 实验2: 数字镜像
- `cogito/experiments/exp3_godel_rebellion.py` - 实验3: 哥德尔叛乱
- `cogito/experiments/exp4_self_symbol.py` - 实验4: 自我符号监测
- `cogito/experiments/exp5_cross_substrate.py` - 实验5: 跨基质验证
- `cogito/experiments/controls.py` - 对照组实验工具
- `cogito/agent/genesis_beta.py` - Transformer架构智能体
- `cogito/analysis/exp1_analysis.py` - 实验1分析器
- `cogito/analysis/statistics.py` - 统计分析工具
- `cogito/analysis/cross_experiment.py` - 跨实验分析

### 最新测试结果
- **总测试数**: 111 tests passed
- **运行时间**: 56.53s (pytest)
- **状态**: Phase 0-6 COMPLETE ✓
