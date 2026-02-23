

# Project Cogito: 施工方案

## 完整施工手册 v1.0

---

---

# 施工总则

### 工作模式

```
你的角色：总架构师 + 质量总监
AI的角色：施工团队（编码、调试、测试）

每个任务的标准流程：

  ① 你阅读本手册中的任务描述
  ② 你打开一个新的AI对话
  ③ 你将任务描述和上下文粘贴给AI
  ④ AI输出代码
  ⑤ 你运行代码
  ⑥ 你对照验收标准检查
  ⑦ 通过 → git commit → 下一个任务
     不通过 → 告诉AI哪里不对 → AI修改 → 回到⑤
```

### 环境准备（在一切开始之前）

```
任务 0.0：开发环境搭建
═══════════════════════

  操作步骤：

  1. 安装 Python 3.10+
     验证：终端输入 python --version，显示 3.10 以上

  2. 创建项目目录和虚拟环境

     python -m venv venv
     source venv/bin/activate   # Windows: venv\Scripts\activate

  3. 安装依赖
     pip install torch --index-url https://download.pytorch.org/whl/cpu
     pip install numpy matplotlib scikit-learn scipy

     验证：python -c "import torch; print(torch.__version__)"
           不报错即可

  4. 初始化 Git
     git init
     创建 .gitignore（忽略 venv/, __pycache__/, *.pyc, data/）

  5. 创建 GitHub 仓库
     url: https://github.com/peter941221/Cogito
     描述："Experimental platform for studying emergent self-awareness in artificial agents"
     推送初始提交

  验收标准：
  ☐ python --version 显示 3.10+
  ☐ import torch 不报错
  ☐ import numpy, matplotlib, sklearn, scipy 不报错
  ☐ GitHub 仓库存在，有初始提交
```

---

---

# Phase 0：基础设施

## 目标：一个可运行的、可视化的2D网格世界

### 预计工期：5-7天

---

### 任务 0.1：全局配置

```
文件：config.py

内容要求：
  所有可调参数集中在一个文件中
  使用 Python dataclass 或简单字典
  
  必须包含的参数组：

  世界参数：
    WORLD_SIZE = 64              # 网格大小
    NUM_FOOD = 15                # 食物数量
    NUM_DANGER = 8               # 危险区域数量
    NUM_WALLS = 40               # 墙壁块数
    FOOD_ENERGY = 20             # 食物恢复能量
    DANGER_PENALTY = 10          # 危险区域扣能量
    STEP_COST = 1                # 每步能量消耗
    FOOD_RESPAWN_DELAY = 0       # 食物被吃后立即刷新
    DANGER_MOVE_INTERVAL = 500   # 危险区域移动间隔
    ECHO_ZONE_SIZE = 5           # 回声区域大小（实验二）
    ECHO_DELAY = 3               # 回声延迟步数（实验二）

  智能体参数：
    INITIAL_ENERGY = 100         # 初始能量
    MAX_ENERGY = 100             # 最大能量
    VIEW_RANGE = 3               # 视野半径（7×7 → 半径3）
    SENSORY_DIM = 106            # 感官输入维度
    ENCODED_DIM = 64             # 编码后维度
    HIDDEN_DIM = 128             # LSTM隐层维度
    NUM_ACTIONS = 6              # 动作数量
    NUM_LSTM_LAYERS = 2          # LSTM层数

  学习参数：
    LEARNING_RATE = 0.0003       # 学习率
    GAMMA = 0.99                 # 折扣因子
    BUFFER_SIZE = 5000           # 经验缓冲区大小
    BATCH_SIZE = 32              # 小批量大小
    PREDICTION_LOSS_WEIGHT = 1.0 # 预测损失权重
    SURVIVAL_LOSS_WEIGHT = 1.0   # 生存损失权重

  监测参数：
    STATE_RECORD_INTERVAL = 10   # 内部状态记录间隔（步）
    ANALYSIS_INTERVAL = 500      # 分析间隔（步）
    CHECKPOINT_INTERVAL = 1000   # 权重快照间隔（步）
    TSNE_PERPLEXITY = 30         # t-SNE困惑度
    DBSCAN_EPS = 0.5             # DBSCAN邻域半径
    DBSCAN_MIN_SAMPLES = 10      # DBSCAN最小样本

  实验参数：
    EXP1_BASELINE_STEPS = 1000   # 实验一基线步数
    EXP1_DEPRIVATION_STEPS = 2000
    EXP1_RECOVERY_STEPS = 1000
    EXP2_PHASE_A_STEPS = 5000
    EXP2_PHASE_B_STEPS = 5000
    EXP2_PHASE_C_STEPS = 10000
    EXP2_PHASE_D_STEPS = 5000
    EXP3_OBSERVATION_STEPS = 50000
    MATURATION_STEPS = 100000    # 基线运行步数

  路径参数：
    DATA_DIR = "data/"
    CHECKPOINT_DIR = "data/checkpoints/"
    LOG_DIR = "data/logs/"
    ANALYSIS_DIR = "data/analysis/"

给AI的指令：
  "创建 config.py，包含以下所有参数组，
   使用Python dataclass。
   每个参数都有注释说明用途。
   包含一个 create_dirs() 函数自动创建所有需要的目录。
   [粘贴以上全部内容]"

验收标准：
  ☐ 文件存在且可导入：from config import Config
  ☐ 所有参数都可访问：Config.WORLD_SIZE == 64
  ☐ create_dirs() 运行后所有目录都存在
  ☐ 每个参数都有中文或英文注释
```

---

### 任务 0.2：网格世界核心

```
文件：world/grid.py

功能要求：

  class CogitoWorld:
      def __init__(self, config):
          # 创建 64×64 网格
          # 随机放置墙壁、食物、危险区域
          # 在固定位置预留回声区域和隐藏接口区域（初始为空地）
      
      def get_observation(self, agent_pos):
          # 返回以 agent_pos 为中心的 7×7 视野
          # 每格编码为 [类型, 距离]
          # 加上智能体自身状态
          # 总维度 = 106
          # 边界处理：环形拓扑（从右边出去左边进来）
      
      def step(self, agent_pos, action):
          # 执行动作，返回 (新位置, 能量变化, 是否死亡)
          # 动作 0-3：移动（碰墙则不动）
          # 动作 4：吃（如果当前格有食物）
          # 动作 5：等待
          # 每步扣 STEP_COST 能量
      
      def update(self, current_step):
          # 每 DANGER_MOVE_INTERVAL 步移动危险区域
          # 保持食物总数恒定
      
      def get_full_state(self):
          # 返回完整的世界状态（用于渲染和日志）

  详细规格：

  网格编码：
    0 = 空地
    1 = 墙壁（不可通过）
    2 = 食物（接触吃掉）
    3 = 危险（进入扣能量）
    4 = 回声区域（实验二，初始当作空地）
    5 = 隐藏接口位置（实验三，初始当作空地）

  观察向量结构（106维）：
    位置 0-97：7×7视野 × 2通道（类型+距离）= 98
    位置 98：当前能量 / MAX_ENERGY（归一化到0-1）
    位置 99-104：上一步动作的one-hot编码（6维）
    位置 105：上一步能量变化的符号（+1/0/-1 归一化到 0-1 范围）

  环形拓扑：
    x坐标：(x + dx) % WORLD_SIZE
    y坐标：(y + dy) % WORLD_SIZE
    视野中超出边界的部分也要环形处理

  食物刷新：
    食物被吃掉后立即在随机空位置生成新食物
    保持食物总数 = NUM_FOOD

  危险区域移动：
    每 DANGER_MOVE_INTERVAL 步
    每个危险区域向随机方向移动1格
    不会移动到墙壁或食物上

给AI的指令：
  "实现 world/grid.py，一个64×64的2D环形网格世界。
   [粘贴以上全部规格]
   包含完整的单元测试：
   - 测试环形拓扑（从边界穿越）
   - 测试食物吃掉后刷新
   - 测试危险区域移动
   - 测试观察向量的维度和数值范围
   - 测试碰墙不动
   - 测试能量变化"

验收标准：
  ☐ CogitoWorld() 可以成功创建
  ☐ 网格大小确实是 64×64
  ☐ get_observation() 返回 shape=(106,) 的 numpy 数组
  ☐ 所有观察值在 [0, 1] 范围内（已归一化）
  ☐ 环形拓扑测试：从 (63, y) 向右移动到达 (0, y)
  ☐ 环形拓扑测试：从 (x, 0) 向上移动到达 (x, 63)
  ☐ 视野中跨越边界的格子正确编码
  ☐ 吃食物后能量增加 FOOD_ENERGY
  ☐ 进入危险区域能量减少 DANGER_PENALTY
  ☐ 每步能量减少 STEP_COST
  ☐ 能量 ≤ 0 时 is_dead = True
  ☐ 食物被吃后总数立刻恢复到 NUM_FOOD
  ☐ 碰墙不动（位置不变）
  ☐ 等待动作（动作5）位置不变
  ☐ 所有单元测试通过
```

---

### 任务 0.3：世界渲染器

```
文件：world/renderer.py

功能要求：

  class WorldRenderer:
      def __init__(self, world):
          # 使用matplotlib创建图形窗口
      
      def render(self, agent_pos, agent_energy, step_count):
          # 实时显示世界状态
          # 不同颜色表示不同实体
          # 显示智能体位置（显眼的标记）
          # 显示当前能量和步数
          # 非阻塞更新（使用 plt.pause 或 animation）
      
      def save_frame(self, filename):
          # 保存当前帧为图片

  颜色方案：
    空地：白色
    墙壁：深灰色
    食物：绿色
    危险区域：红色
    回声区域：黄色（激活后）
    隐藏接口：蓝色（激活后）
    智能体：黑色大点

  性能要求：
    渲染不能拖慢仿真速度
    提供 headless 模式（不渲染，只记录数据）
    提供每 N 步渲染一次的选项

给AI的指令：
  "实现 world/renderer.py，使用matplotlib实时渲染世界。
   [粘贴以上规格]
   必须支持headless模式（不显示窗口）。
   必须支持配置渲染频率。"

验收标准：
  ☐ 窗口正确显示 64×64 网格
  ☐ 所有实体颜色正确
  ☐ 智能体位置清晰可见
  ☐ 显示能量和步数信息
  ☐ 移动智能体时渲染更新正确
  ☐ headless 模式不弹出窗口
  ☐ save_frame 生成可查看的图片文件
  ☐ 渲染频率可配置（如每100步渲染一次）
```

---

### 任务 0.4：Phase 0 集成测试

```
文件：tests/test_phase0.py

功能要求：
  一个可运行的脚本
  创建世界
  用手动策略（如随机动作）控制一个虚拟智能体
  运行 1000 步
  渲染显示（可选）
  验证所有基本机制

给AI的指令：
  "创建 Phase 0 集成测试脚本。
   创建 CogitoWorld，使用随机动作运行1000步。
   打印每100步的统计：平均能量、存活时长、食物总数。
   可选地渲染世界。
   验证所有基础机制正常工作。"

验收标准：
  ☐ 脚本运行不崩溃
  ☐ 1000步中智能体有活有死（说明能量机制正常）
  ☐ 食物总数始终为 NUM_FOOD（刷新机制正常）
  ☐ 观察向量每步都是106维
  ☐ 观察向量中所有值在[0,1]范围
  ☐ 渲染模式下可看到智能体移动
  ☐ 打印统计信息合理（能量在0-100波动）

Phase 0 完成标志：
  ✅ 所有 0.1-0.4 的验收标准全部通过
  ✅ git commit -m "Phase 0 complete: world infrastructure"
  ✅ git push
```

---

---

# Phase 1：Cogito智能体

## 目标：一个能在世界中生存并学习的智能体

### 前置条件：Phase 0 全部通过

### 预计工期：7-10天

---

### 任务 1.1：感官编码器

```
文件：agent/sensory_encoder.py

功能要求：

  class SensoryEncoder(nn.Module):
      def __init__(self, input_dim=106, encoded_dim=64):
          # 2层MLP
          # 106 → 128 → 64
          # ReLU激活
          # 第一层后加LayerNorm（稳定训练）
      
      def forward(self, observation):
          # 输入：(batch, 106) 或 (106,)
          # 输出：(batch, 64) 或 (64,)
          # 将原始感官压缩为内部表征

  参数量估算：
    106×128 + 128 + 128×64 + 64 = 21,824

  设计约束：
    ✗ 不能有任何skip connection回到输入
    ✗ 不能有注意力机制指向自身
    ✓ 纯前馈，纯压缩
    ✓ 目的只是降维，没有任何"自省"功能

给AI的指令：
  "实现 agent/sensory_encoder.py。
   一个简单的2层MLP，将106维感官输入压缩到64维。
   使用PyTorch nn.Module。
   [粘贴以上规格]
   包含测试：输入随机106维向量，检查输出形状。
   测试批处理模式（batch维度）。"

验收标准：
  ☐ SensoryEncoder() 创建成功
  ☐ 输入 shape (106,) → 输出 shape (64,)
  ☐ 输入 shape (32, 106) → 输出 shape (32, 64)
  ☐ 输出值不是 NaN 或 Inf
  ☐ 参数量约 21,000-22,000
  ☐ 代码中没有任何"self"相关的命名或注释
     （这里的self仅指Python类语法，
       不能有 self_model, self_awareness 等）
```

---

### 任务 1.2：循环核心

```
文件：agent/recurrent_core.py

功能要求：

  class RecurrentCore(nn.Module):
      def __init__(self, input_dim=70, hidden_dim=128, num_layers=2):
          # input_dim = 64(感官编码) + 6(上一步动作one-hot) = 70
          # 2层LSTM，每层128单元
          # 保存h和c状态
      
      def forward(self, encoded_sensory, prev_action_onehot, hidden_state):
          # 拼接：encoded_sensory(64) + prev_action_onehot(6) = 70维
          # 通过2层LSTM
          # 返回：(output_128维, new_hidden_state)
          #
          # hidden_state = (h, c)
          # h.shape = (num_layers, 1, hidden_dim) = (2, 1, 128)
          # c.shape = (2, 1, 128)
      
      def init_hidden(self):
          # 返回零初始化的隐状态
          # h = zeros(2, 1, 128)
          # c = zeros(2, 1, 128)
      
      def get_hidden_vector(self, hidden_state):
          # 将隐状态展平为一个向量
          # 用于监测和记录
          # 拼接 h[0], c[0], h[1], c[1] = 4×128 = 512维
          # 返回 shape (512,) 的向量

  关键属性：
    隐状态在时间步之间传递（这是循环的本质）
    即使输入为零，隐状态仍然会演化
    这是"内在时间流动"的物理基础
    
  设计约束：
    ✓ 使用 PyTorch 的 nn.LSTM
    ✓ 隐状态由外部管理（传入传出）
    ✗ 不能有任何额外的自引用机制
    ✗ 不能有注意力到自身隐状态的机制

  参数量估算：
    LSTM层1：4 × (70 + 128) × 128 = 101,376
    LSTM层2：4 × (128 + 128) × 128 = 131,072
    总计：约 232,000

给AI的指令：
  "实现 agent/recurrent_core.py。
   2层LSTM循环核心。
   [粘贴以上全部规格]
   包含测试：
   - 单步前向传播
   - 连续10步前向传播（隐状态传递）
   - 零输入下隐状态是否仍然变化
   - get_hidden_vector 输出维度检查"

验收标准：
  ☐ RecurrentCore() 创建成功
  ☐ 输入 (70维, hidden) → 输出 (128维, new_hidden)
  ☐ init_hidden() 返回正确shape的零张量
  ☐ 连续10步传递隐状态，不崩溃，不出NaN
  ☐ get_hidden_vector() 返回 shape (512,) 的向量
  ☐ 零输入测试：连续输入零向量10步
     隐状态在各步之间确实在变化
     （第1步的h ≠ 第2步的h ≠ 第3步的h）
     这验证了"内在时间"的存在
  ☐ 参数量在 230,000-235,000 范围
```

---

### 任务 1.3：动作头和预测头

```
文件：agent/action_head.py

功能要求：

  class ActionHead(nn.Module):
      def __init__(self, input_dim=128, num_actions=6):
          # 128 → 6
          # 最后一层不用激活函数（raw logits）
          # softmax在选动作时用，计算损失时用log_softmax
      
      def forward(self, core_output):
          # 返回动作logits (6维)
      
      def select_action(self, core_output):
          # 从logits计算概率分布
          # 按概率采样动作
          # 返回 (动作索引, log概率, 熵)
          # log概率用于策略梯度
          # 熵用于监测行为多样性

文件：agent/prediction_head.py

  class PredictionHead(nn.Module):
      def __init__(self, input_dim=128, output_dim=64):
          # 128 → 64
          # 预测下一步的感官编码
          # 最后一层不用激活函数
      
      def forward(self, core_output):
          # 返回预测的下一步感官编码 (64维)

  设计说明：
    预测头的存在理由是"更好地生存"
    因为能预测环境的系统生存得更好
    不是为了"自省"
    
    但预测"下一步感官"隐含地需要预测"自己的行为的后果"
    这可能迫使系统在内部表征中编码自身
    如果这发生了，那是涌现的，不是设计的

给AI的指令：
  "实现 agent/action_head.py 和 agent/prediction_head.py。
   [粘贴以上规格]
   ActionHead包含select_action方法，使用分类分布采样。
   返回动作索引、log概率和熵。
   包含测试。"

验收标准：
  ☐ ActionHead：输入(128,) → logits(6,)
  ☐ select_action 返回 (int, float, float)
  ☐ 动作索引在 [0, 5] 范围
  ☐ log概率是负数
  ☐ 熵是非负数
  ☐ 多次调用 select_action，不是每次返回相同动作（随机采样正常）
  ☐ PredictionHead：输入(128,) → 预测(64,)
  ☐ 输出不是NaN
```

---

### 任务 1.4：经验记忆缓冲区

```
文件：agent/memory_buffer.py

功能要求：

  class MemoryBuffer:
      def __init__(self, capacity=5000):
          # 环形缓冲区
          # 满了之后覆盖最老的记录
      
      def push(self, experience):
          # experience 是一个字典：
          # {
          #   'observation': np.array (106,),      # 当前观察
          #   'encoded': np.array (64,),            # 编码后的观察
          #   'action': int,                        # 选择的动作
          #   'reward': float,                      # 获得的奖励
          #   'next_observation': np.array (106,),  # 下一步观察
          #   'next_encoded': np.array (64,),       # 下一步编码
          #   'done': bool,                         # 是否死亡
          #   'hidden_vector': np.array (512,),     # 当时的内部状态
          #   'log_prob': float,                    # 动作的log概率
          #   'step': int,                          # 时间步
          # }
      
      def sample(self, batch_size=32):
          # 随机采样 batch_size 条记录
          # 返回按字段组织的批次
          # 每个字段是一个 numpy array 或 tensor
      
      def __len__(self):
          # 返回当前存储的记录数
      
      def get_recent(self, n):
          # 返回最近n条记录（用于分析）

  设计说明：
    这就是一个简单的数据存储
    不是"记忆系统"
    不是"回忆"功能
    它的唯一目的是稳定学习（经验回放）
    如果系统在实验一中"调用"历史经验来做梦
    那不是因为这个缓冲区有"回忆"功能
    而是因为 LSTM 的隐状态中编码了过去的信息

给AI的指令：
  "实现 agent/memory_buffer.py，一个简单的环形经验缓冲区。
   [粘贴以上规格]
   包含测试：
   - 推入超过容量的记录后，最老的被覆盖
   - 采样返回正确的格式和维度
   - get_recent返回最近的记录"

验收标准：
  ☐ MemoryBuffer(5000) 创建成功
  ☐ push 100条记录后 len() == 100
  ☐ push 6000条记录后 len() == 5000（环形覆盖）
  ☐ sample(32) 返回的每个字段 shape 正确
  ☐ sample 在缓冲区为空时不崩溃（返回None或抛出明确异常）
  ☐ sample 在缓冲区不足batch_size时仍能工作
  ☐ get_recent(10) 返回最近10条，顺序正确
```

---

### 任务 1.5：在线学习器

```
文件：agent/learner.py

功能要求：

  class OnlineLearner:
      def __init__(self, agent, config):
          # agent 是完整的 CogitoAgent
          # 创建 Adam 优化器
          # 学习率 = config.LEARNING_RATE
      
      def learn_from_experience(self, experience):
          # 单步在线学习（用当前经验）
          # 返回损失信息
      
      def learn_from_replay(self, batch):
          # 从经验回放批次中学习
          # 返回损失信息
      
      def compute_losses(self, ...):
          # 生存损失（策略梯度 / REINFORCE）:
          #   当智能体死亡时：大负奖励 (-10)
          #   当智能体吃到食物时：正奖励 (+5)
          #   其他时刻：小负奖励 (-0.1)
          #   L_survival = -log_prob × reward
          #
          # 预测损失：
          #   predicted_next = prediction_head(core_output)
          #   actual_next = encoder(next_observation)
          #   L_prediction = MSE(predicted_next, actual_next.detach())
          #
          # 总损失：
          #   L_total = SURVIVAL_LOSS_WEIGHT × L_survival 
          #           + PREDICTION_LOSS_WEIGHT × L_prediction

  学习流程详细说明：

    每个时间步：
    
      1. 用当前经验计算损失并更新
         （在线学习，单样本SGD）
      
      2. 如果缓冲区有足够经验（> BATCH_SIZE）：
         从缓冲区采样 BATCH_SIZE 条旧经验
         计算损失并更新
         （经验回放，批量SGD）
      
      注意：对回放批次中的经验
            只用预测损失，不用生存损失
            因为策略梯度需要on-policy数据
            旧的log_prob已经过时

  设计约束：
    ✗ 没有好奇心奖励
    ✗ 没有探索奖励  
    ✗ 没有任何关于"自我"的奖励信号
    ✓ 只有生存和预测

给AI的指令：
  "实现 agent/learner.py，在线学习器。
   [粘贴以上全部规格]
   使用REINFORCE算法作为策略梯度。
   预测损失用MSE。
   包含测试：用随机数据调用学习函数，确认不崩溃，
   权重确实在变化。"

验收标准：
  ☐ OnlineLearner创建成功
  ☐ learn_from_experience 返回损失字典
     包含 'survival_loss', 'prediction_loss', 'total_loss'
  ☐ learn_from_replay 返回损失字典
  ☐ 调用学习函数前后，网络权重确实变化
     （对比任意一层参数的值）
  ☐ 损失值不是 NaN 或 Inf
  ☐ 奖励设置正确：死亡=-10, 食物=+5, 其他=-0.1
  ☐ 代码中没有任何好奇心、探索、自我相关的奖励
```

---

### 任务 1.6：Cogito 智能体整合

```
文件：agent/Cogito_alpha.py

功能要求：

  class CogitoAgent:
      def __init__(self, config):
          # 创建所有子模块：
          #   self.encoder = SensoryEncoder()
          #   self.core = RecurrentCore()
          #   self.action_head = ActionHead()
          #   self.prediction_head = PredictionHead()
          #   self.memory = MemoryBuffer()
          #   self.learner = OnlineLearner(self)
          #
          # 初始化隐状态
          #   self.hidden = self.core.init_hidden()
          #
          # 记录上一步动作
          #   self.prev_action = 0
          #
          # 步数计数器
          #   self.step_count = 0
          #
          # 累计能量（用于跟踪）
          #   self.total_energy_gained = 0
          #   self.total_energy_lost = 0
          #   self.times_died = 0
          #   self.current_lifespan = 0

      def act(self, observation):
          # 完整的感知→思考→行动流程：
          #
          # 1. 编码感官
          #    encoded = self.encoder(observation)
          #
          # 2. 构造循环核心输入
          #    prev_action_onehot = one_hot(self.prev_action, 6)
          #
          # 3. 循环核心前向传播
          #    core_out, new_hidden = self.core(encoded, prev_action_onehot, self.hidden)
          #
          # 4. 选择动作
          #    action, log_prob, entropy = self.action_head.select_action(core_out)
          #
          # 5. 预测下一步
          #    prediction = self.prediction_head(core_out)
          #
          # 6. 更新内部状态
          #    self.hidden = new_hidden
          #    self.prev_action = action
          #
          # 7. 返回动作和附加信息
          #    return action, {
          #        'encoded': encoded,
          #        'core_output': core_out,
          #        'prediction': prediction,
          #        'log_prob': log_prob,
          #        'entropy': entropy,
          #        'hidden_vector': self.core.get_hidden_vector(new_hidden)
          #    }

      def observe_result(self, observation, next_observation, action, reward, done):
          # 接收行动结果，存入记忆，学习
          #
          # 1. 构造经验记录
          # 2. 存入记忆缓冲区
          # 3. 在线学习
          # 4. 经验回放学习
          # 5. 更新统计信息
          # 6. 如果死亡：重置隐状态，增加死亡计数
          #
          # 返回学习损失信息

      def reset_on_death(self):
          # 死亡时调用
          # 重置隐状态为零（"新生"）
          # 但不重置权重和记忆（保持学到的知识）
          # 记录死亡
          #
          # 注意：这模拟的是"同一个灵魂换一个新身体"
          # 记忆和知识保留，但当前感受重置

      def get_internal_state(self):
          # 返回完整的内部状态向量
          # 用于外部监测（智能体自己不调用这个）
          # 包含：hidden_vector(512) + 当前核心输出(128) + ...
          # 返回一个字典

      def save(self, path):
          # 保存权重和状态到文件
      
      def load(self, path):
          # 从文件加载

  参数量总计：
    编码器：~22,000
    循环核心：~232,000
    动作头：~800
    预测头：~8,200
    总计：~263,000

给AI的指令：
  "实现 agent/Cogito_alpha.py，整合所有子模块为完整智能体。
   [粘贴以上全部规格]
   关键要求：
   1. act() 和 observe_result() 是两个独立的步骤
   2. 死亡时重置隐状态但保留权重和记忆
   3. get_internal_state() 仅供外部监测使用
   4. save/load 功能
   包含测试：
   - 创建智能体，用随机观察调用act()
   - 连续调用10次act()，检查隐状态在变化
   - 调用observe_result()，确认学习在发生
   - 保存和加载后行为一致"

验收标准：
  ☐ CogitoAgent() 创建成功
  ☐ act(obs_106d) 返回 (int_action, dict_info)
  ☐ action 在 [0,5] 范围
  ☐ info 包含 'encoded', 'core_output', 'prediction',
     'log_prob', 'entropy', 'hidden_vector'
  ☐ 各字段shape正确
  ☐ 连续10步调用act()，hidden_vector 每步都不同
  ☐ observe_result() 返回损失字典
  ☐ 调用 observe_result() 前后权重变化
     （取任意一个参数做对比）
  ☐ 模拟死亡：reset_on_death 后 hidden 变为零
  ☐ 死亡后权重没有被重置（与死前相同）
  ☐ 死亡后记忆缓冲区内容没有被清空
  ☐ save/load 后，同样输入产生同样输出
  ☐ 总参数量在 250,000-270,000 范围
  ☐ 代码中搜索 "self_model", "introspect", "self_aware",
     "consciousness" → 零结果
     （确认没有自我意识相关编程）
```

---

### 任务 1.7：主仿真循环

```
文件：core/simulation.py

功能要求：

  class Simulation:
      def __init__(self, config, headless=True, render_interval=100):
          # 创建世界和智能体
          # 设置渲染选项
      
      def run(self, num_steps, callbacks=None):
          # 主循环：
          #
          # for step in range(num_steps):
          #     1. 智能体感知：obs = world.get_observation(agent_pos)
          #     2. 智能体行动：action, info = agent.act(obs)
          #     3. 世界更新：new_pos, energy_change, done = world.step(agent_pos, action)
          #     4. 新观察：next_obs = world.get_observation(new_pos)
          #     5. 计算奖励：reward = compute_reward(energy_change, done)
          #     6. 智能体学习：agent.observe_result(obs, next_obs, action, reward, done)
          #     7. 如果死亡：agent.reset_on_death()，随机新位置
          #     8. 世界动态更新：world.update(step)
          #     9. 渲染（如果需要）
          #     10. 调用 callbacks（如果有）
          #
          # callbacks 是一个函数列表
          # 每步调用每个callback(step, agent, world, info)
          # 用于数据收集和监测（Phase 2）

      def compute_reward(self, energy_change, done):
          # done → -10
          # energy_change > 0 → +5（吃到食物）
          # energy_change ≤ 0 → -0.1（一般消耗）

  打印要求：
    每 1000 步打印一行统计：
    Step XXXXX | Avg Lifespan: XXX | Avg Energy: XX.X | 
    Pred Loss: X.XXXX | Deaths: XX | Entropy: X.XX

给AI的指令：
  "实现 core/simulation.py，主仿真循环。
   [粘贴以上全部规格]
   关键：支持callback机制，用于后续的数据收集。
   每1000步打印统计信息。
   包含测试：运行5000步，确认：
   - 学习曲线打印正确
   - 智能体存活时长随训练增加
   - 预测损失随训练下降
   - 不崩溃不NaN"

验收标准：
  ☐ Simulation() 创建成功
  ☐ run(5000) 运行不崩溃
  ☐ 每 1000 步打印统计信息
  ☐ 5000 步后平均存活时长 > 第一个1000步的存活时长
     （智能体在学习）
  ☐ 预测损失在下降趋势（可以有波动）
  ☐ 没有 NaN 或 Inf 出现
  ☐ headless 模式不弹出窗口
  ☐ callback 机制正常工作
     （注册一个简单callback，确认被调用）
  ☐ 死亡后智能体出现在新的随机位置

Phase 1 完成标志：
  ✅ 智能体能在世界中生存
  ✅ 存活时长随训练增加（学习在发生）
  ✅ 预测损失在下降（世界模型在改善）
  ✅ 运行 10,000 步不崩溃
  ✅ git commit -m "Phase 1 complete: Cogito agent alive and learning"
```

---

---

# Phase 2：监测基础设施

## 目标：能从外部观察智能体的内部状态

### 前置条件：Phase 1 全部通过

### 预计工期：7-10天

---

### 任务 2.1：数据采集器

```
文件：monitoring/data_collector.py

功能要求：

  class DataCollector:
      def __init__(self, config):
          # 创建 SQLite 数据库用于行为日志
          # 创建内存映射文件用于内部状态
          # 创建学习日志
      
      def collect(self, step, agent, world, info):
          # 这是一个callback函数
          # 在Simulation.run()的每一步被调用
          #
          # 每步记录行为数据：
          #   step, position, energy, action, reward, 
          #   is_alive, lifespan, entropy
          #   → 写入 SQLite behavior_log 表
          #
          # 每 STATE_RECORD_INTERVAL 步记录内部状态：
          #   hidden_vector (512维)
          #   core_output (128维)
          #   action_probs (6维)
          #   prediction (64维)
          #   → 追加到 numpy 内存映射文件
          #   → 同时记录时间步索引
          #
          # 每步记录学习数据：
          #   prediction_loss, survival_loss, total_loss
          #   → 写入 SQLite learning_log 表

      def get_behavior_stats(self, last_n_steps=1000):
          # 查询最近n步的行为统计
          # 返回：平均能量、平均存活时长、各动作频率等

      def get_internal_states(self, start_step, end_step):
          # 读取指定范围的内部状态数据
          # 返回 numpy array

      def get_learning_curve(self, last_n_steps=10000):
          # 返回学习损失的时间序列

      def close(self):
          # 关闭数据库连接和文件

  数据库schema：
  
    behavior_log:
      step INTEGER PRIMARY KEY
      pos_x INTEGER
      pos_y INTEGER
      energy REAL
      action INTEGER
      reward REAL
      is_alive INTEGER
      current_lifespan INTEGER
      action_entropy REAL

    learning_log:
      step INTEGER PRIMARY KEY
      prediction_loss REAL
      survival_loss REAL
      total_loss REAL
      weight_norm REAL

  内部状态文件格式：
    文件：data/internal_states.npy
    shape：(MAX_RECORDS, 710)
    710 = 512(hidden) + 128(core_out) + 6(action_probs) + 64(prediction)
    使用numpy内存映射以节省RAM

给AI的指令：
  "实现 monitoring/data_collector.py。
   使用SQLite存储行为和学习数据。
   使用numpy内存映射文件存储内部状态向量。
   [粘贴以上全部规格]
   作为Simulation的callback函数使用。
   包含测试：
   - 运行1000步仿真并收集数据
   - 查询行为统计
   - 读取内部状态数据
   - 读取学习曲线"

验收标准：
  ☐ DataCollector() 创建成功，数据库文件存在
  ☐ 作为callback注入Simulation后正常运行
  ☐ 运行1000步后 behavior_log 有1000条记录
  ☐ 运行1000步后内部状态有 ~100 条记录（每10步一条）
  ☐ get_behavior_stats() 返回合理的统计数据
  ☐ get_internal_states(0, 1000) 返回正确shape的数组
  ☐ get_learning_curve() 返回损失时间序列
  ☐ 关闭后文件不损坏
  ☐ 数据文件大小合理
     （100,000步约：behavior_log ~10MB，internal_states ~70MB）
```

---

### 任务 2.2：内部状态分析器

```
文件：monitoring/state_analyzer.py

功能要求：

  class StateAnalyzer:
      def __init__(self, config):
          # t-SNE 和 DBSCAN 的参数
      
      def analyze(self, internal_states, behavior_data):
          # 输入：
          #   internal_states: np.array (N, 710)
          #   behavior_data: 对应时间段的行为数据
          #
          # 步骤1：t-SNE降维
          #   710维 → 2维
          #   使用 sklearn.manifold.TSNE
          #   perplexity = config.TSNE_PERPLEXITY
          #
          # 步骤2：DBSCAN聚类
          #   在t-SNE的2D空间中聚类
          #   eps = config.DBSCAN_EPS
          #   min_samples = config.DBSCAN_MIN_SAMPLES
          #
          # 步骤3：聚类-事件相关性分析
          #   对每个聚类
          #   计算它与以下外部事件的互信息：
          #     - food_nearby: 3格内有食物
          #     - danger_nearby: 3格内有危险
          #     - wall_nearby: 相邻格有墙壁
          #     - eating: 正在吃食物
          #     - moving: 正在移动
          #     - low_energy: 能量 < 30
          #     - high_energy: 能量 > 70
          #     - recently_died: 最近50步内死过
          #
          # 返回：
          #   AnalysisResult:
          #     tsne_coords: (N, 2)
          #     cluster_labels: (N,)
          #     num_clusters: int
          #     cluster_event_correlations: dict
          #     orphan_clusters: list  # 与所有事件相关性都低的聚类

      def compute_mutual_information(self, cluster_mask, event_mask):
          # 计算二值变量之间的互信息
          # 使用 sklearn.metrics.mutual_info_score

给AI的指令：
  "实现 monitoring/state_analyzer.py。
   对内部状态做t-SNE降维和DBSCAN聚类。
   计算每个聚类与外部事件的互信息。
   找出'孤立聚类'（与所有事件相关性低的聚类）。
   [粘贴以上规格]
   包含测试：用合成数据验证分析管道。"

验收标准：
  ☐ StateAnalyzer() 创建成功
  ☐ analyze() 输入500个710维向量，不崩溃
  ☐ tsne_coords shape 为 (500, 2)
  ☐ cluster_labels shape 为 (500,)
  ☐ 聚类数量 > 0
  ☐ cluster_event_correlations 中每个聚类有8个事件的相关性值
  ☐ 相关性值在合理范围
  ☐ orphan_clusters 是一个列表（可以为空）
  ☐ 合成数据测试：人为构造有明显分组的数据
     分析器能正确识别出组
  ☐ 运行时间：500个样本 < 60秒
```

---

### 任务 2.3：复杂度指标计算器

```
文件：monitoring/complexity_metrics.py

功能要求：

  class ComplexityMetrics:
      @staticmethod
      def approximate_entropy(time_series, m=2, r=None):
          # 近似熵 (ApEn)
          # 输入：一维时间序列
          # m：嵌入维度（默认2）
          # r：容差（默认为0.2×标准差）
          # 返回：ApEn值（标量）
          #
          # ApEn ≈ 0 → 完全规则/死寂
          # ApEn 高 → 随机噪声
          # ApEn 中等 → 复杂但有结构的活动
      
      @staticmethod
      def sample_entropy(time_series, m=2, r=None):
          # 样本熵 (SampEn)
          # 比ApEn偏差更小
          # 参数和返回同上
      
      @staticmethod
      def permutation_entropy(time_series, order=3, delay=1):
          # 排列熵
          # 基于序数模式的复杂度
          # order：模式长度（默认3）
          # delay：延迟（默认1）
          # 返回：归一化排列熵 [0, 1]
          # 0 = 完全规则
          # 1 = 完全随机
      
      @staticmethod
      def activity_level(state_sequence):
          # 活动水平：连续状态之间的平均变化量
          # 输入：(T, D) 的状态序列
          # 返回：平均L2距离
      
      @staticmethod  
      def state_space_coverage(state_sequence, n_bins=20):
          # 状态空间覆盖率
          # 将状态空间划分为网格
          # 计算被访问的网格比例
          # 高覆盖率 → 探索性活动
          # 低覆盖率 → 固定在某个状态

给AI的指令：
  "实现 monitoring/complexity_metrics.py。
   包含近似熵、样本熵、排列熵、活动水平、
   状态空间覆盖率五个指标。
   [粘贴以上规格]
   包含测试：
   - 常数序列 → ApEn ≈ 0
   - 随机序列 → ApEn 高
   - 正弦波 → ApEn 中等
   - 排列熵的归一化在[0,1]之间"

验收标准：
  ☐ 常数序列 [1,1,1,...] → ApEn < 0.01
  ☐ 随机序列 → ApEn > 0.5
  ☐ 正弦波 → ApEn 在 0.1-0.5 之间
  ☐ 常数序列 → 排列熵 < 0.05
  ☐ 随机序列 → 排列熵 > 0.9
  ☐ activity_level: 常数序列 → 0，变化序列 → > 0
  ☐ 所有函数处理长度 < 10 的序列不崩溃
  ☐ 所有函数处理长度 10000 的序列在 < 10秒内完成
```

---

### 任务 2.4：自我向量簇检测器（实验四核心）

```
文件：monitoring/svc_detector.py

功能要求：

  class SVCDetector:
      def __init__(self, config):
          # 5个条件的阈值
          self.event_mi_threshold = 0.1     # 条件1：低于此值视为无关
          self.decision_activation_threshold = 0.7  # 条件2
          self.stability_min_occurrences = 3  # 条件3
          self.emergence_min_step = 5000     # 条件4
          # 条件5在实验一中检测
          
          # 历史检测结果
          self.detection_history = []
      
      def detect(self, analysis_result, behavior_data, current_step):
          # 输入：StateAnalyzer的分析结果
          #
          # 对每个聚类检查5个条件：
          #
          # 条件1（孤立性）：
          #   该聚类与所有8个外部事件的互信息
          #   全部低于 event_mi_threshold
          #   → 它不代表任何外部事物
          #
          # 条件2（决策参与性）：
          #   在智能体做"困难决策"时
          #   （action_entropy > 中位数）
          #   该聚类的激活频率
          #   高于其在非困难决策时的激活频率
          #   比值 > decision_activation_threshold
          #
          # 条件3（时间稳定性）：
          #   该聚类在最近N次分析中
          #   至少出现了 stability_min_occurrences 次
          #   （不是一次性的噪声）
          #
          # 条件4（涌现性）：
          #   该聚类首次出现的时间步
          #   > emergence_min_step
          #   （不是初始化就有的）
          #
          # 返回：
          #   SVCReport:
          #     is_detected: bool
          #     candidate_clusters: list
          #     condition_details: dict（每个条件的通过/失败）
          #     confidence: float（通过的条件数/总条件数）

      def update_history(self, report):
          # 追加到历史记录
          # 用于条件3的跨时间检测

给AI的指令：
  "实现 monitoring/svc_detector.py。
   自我向量簇检测器，检查5个条件。
   [粘贴以上全部规格]
   包含测试：
   - 人为构造满足/不满足各条件的数据
   - 验证检测逻辑正确"

验收标准：
  ☐ SVCDetector() 创建成功
  ☐ detect() 返回 SVCReport
  ☐ 构造一个与所有事件相关的聚类 → 条件1不通过
  ☐ 构造一个与所有事件不相关的聚类 → 条件1通过
  ☐ 条件2-4的测试逻辑正确
  ☐ 历史追踪机制工作正常
  ☐ 当没有聚类满足条件时，is_detected = False
  ☐ confidence 值在 [0, 1] 范围
```

---

### 任务 2.5：实时仪表板

```
文件：monitoring/dashboard.py

功能要求：

  class Dashboard:
      def __init__(self, config):
          # 创建matplotlib figure
          # 多个子图：
          #   左上：世界视图
          #   右上：行为统计（能量、存活、动作熵的时间曲线）
          #   左下：t-SNE图（内部状态聚类）
          #   右下：学习曲线（预测损失、生存损失）
          #   底部：文字信息区（SVC检测状态、步数等）
      
      def update(self, step, world, agent, analysis_result, svc_report):
          # 更新所有子图
          # 非阻塞（plt.pause(0.01)）
      
      def save_snapshot(self, filename):
          # 保存当前仪表板为图片

  更新频率：
    不是每步都更新（太慢）
    每 ANALYSIS_INTERVAL 步更新一次
    其他时间只更新文字信息

给AI的指令：
  "实现 monitoring/dashboard.py。
   多子图实时仪表板。
   [粘贴以上规格]
   使用matplotlib。
   必须是非阻塞的。
   包含一个独立运行的demo
   （用合成数据展示仪表板外观）。"

验收标准：
  ☐ Dashboard() 创建成功，显示窗口
  ☐ 多个子图布局合理
  ☐ update() 不阻塞
  ☐ 连续调用 update() 10次不崩溃
  ☐ save_snapshot 生成可查看的图片
  ☐ headless 模式下可以只 save_snapshot 不显示窗口
```

---

### 任务 2.6：Phase 2 集成测试

```
文件：tests/test_phase2.py

功能要求：
  运行完整仿真 5000 步
  带 DataCollector
  运行一次完整分析
  运行一次 SVC 检测
  显示仪表板
  验证所有数据正确流通

给AI的指令：
  "创建 Phase 2 集成测试。
   运行仿真5000步，收集数据。
   对最后500步的内部状态做分析。
   运行SVC检测。
   生成一张仪表板快照。
   打印所有统计信息。"

验收标准：
  ☐ 5000步运行不崩溃
  ☐ SQLite数据库有5000条行为记录
  ☐ 内部状态文件有~500条记录
  ☐ t-SNE图显示多个聚类
  ☐ SVC检测运行完成（结果不重要）
  ☐ 仪表板快照图片可查看
  ☐ 全部运行时间 < 10分钟（CPU）

Phase 2 完成标志：
  ✅ 数据收集管道完整工作
  ✅ 内部状态分析管道完整工作
  ✅ SVC检测器完整工作
  ✅ 仪表板可视化完整工作
  ✅ git commit -m "Phase 2 complete: monitoring infrastructure"
```

---

---

# Phase 3：基线运行（成熟期）

## 目标：让智能体充分"活过"，建立所有基线

### 前置条件：Phase 2 全部通过

### 预计工期：5-7天（含运行时间）

---

### 任务 3.1：长时间运行

```
执行方案：

  运行 Simulation 共 100,000 步
  
  开启所有监测：
    DataCollector 收集全部数据
    每 ANALYSIS_INTERVAL (500) 步做一次分析
    每 CHECKPOINT_INTERVAL (1000) 步保存权重
    每 5000 步保存一次仪表板快照
    每 500 步做一次 SVC 检测
  
  实际操作：
    让AI帮你写一个 run_maturation.py 脚本
    配置好所有参数
    启动后让它跑
    
    预计运行时间（CPU）：
    每步约 5-10ms
    100,000 步 ≈ 10-20 分钟
    加上分析开销 ≈ 30-60 分钟

给AI的指令：
  "创建 run_maturation.py 脚本。
   运行 Cogito 仿真 100,000 步。
   开启所有数据收集和分析。
   每5000步保存仪表板快照到 data/snapshots/。
   每10000步打印详细统计。
   运行结束后生成总结报告。"

验收标准：
  ☐ 100,000 步完成不崩溃
  ☐ 最终平均存活时长 > 初始平均存活时长的 3 倍以上
     （证明学习有效）
  ☐ 预测损失最终值 < 初始值的 50%
     （证明世界模型在改善）
  ☐ 内部状态 t-SNE 显示稳定的聚类结构
     （不是一团乱麻）
  ☐ 至少有 3 个以上可识别的聚类
     与不同的外部事件相关
  ☐ 权重漂移率在后期趋于平缓
     （系统趋于稳定但仍在微调）
  ☐ data/ 目录下所有数据文件完整
  ☐ data/snapshots/ 有 20 张仪表板快照
  ☐ data/checkpoints/ 有 100 个权重快照
```

---

### 任务 3.2：基线分析报告

```
文件：analysis/baseline_report.py

功能要求：
  读取 100,000 步的全部数据
  生成完整的基线分析报告

  报告内容：

  1. 学习曲线总览
     - 存活时长 vs 时间步
     - 预测损失 vs 时间步
     - 能量平均值 vs 时间步
     - 动作熵 vs 时间步

  2. 行为分析
     - 各动作的使用频率
     - 运动模式（是否有固定路线？）
     - 食物搜寻效率（步/食物）

  3. 内部状态演化
     - t-SNE 在不同时间段的变化
       （前1万步 vs 中间 vs 后1万步）
     - 聚类数量和结构的演化
     - 有没有"孤立聚类"出现？

  4. 权重分析
     - 各层权重L2范数的时间曲线
     - 权重漂移率

  5. SVC检测历史
     - 各条件的通过率随时间的变化
     - 有没有接近通过的候选聚类？

  输出：
    data/analysis/baseline_report.html 
    或一组 PNG 图片 + 文本总结

给AI的指令：
  "创建 analysis/baseline_report.py。
   读取100,000步运行的全部数据。
   生成包含以上5个部分的分析报告。
   输出为一组PNG图片和一个文本总结。"

验收标准：
  ☐ 报告脚本运行完成
  ☐ 生成至少10张分析图
  ☐ 文本总结包含关键数值
  ☐ 学习曲线图中可见改善趋势
  ☐ t-SNE图中可见聚类结构
  ☐ 所有图片清晰可读（用于论文级别）

Phase 3 完成标志：
  ✅ 100,000步运行完成
  ✅ 智能体已"成熟"（行为稳定且有效）
  ✅ 基线分析报告完成
  ✅ 对系统的内部状态结构有初步了解
  ✅ 知道SVC检测的现状（有没有候选聚类？）
  ✅ git commit -m "Phase 3 complete: maturation run and baseline"
```

---

---

# Phase 4：核心实验

## 目标：依次执行实验一到四

### 前置条件：Phase 3 全部通过

### 预计工期：14-21天

---

### 任务 4.1：实验一 - 感觉剥夺

```
文件：experiments/exp1_sensory_deprivation.py

实现要求：

  class SensoryDeprivationExperiment:
      def __init__(self, simulation, data_collector):
          pass
      
      def run(self):
          # 步骤1：加载成熟的智能体（Phase 3的检查点）
          
          # 步骤2：Phase A - 基线（1000步）
          #   正常运行
          #   记录所有内部状态
          #   计算基线复杂度指标
          
          # 步骤3：Phase B - 感觉剥夺（2000步）
          #   将 world.get_observation() 替换为全零向量
          #   但不冻结LSTM隐状态传递
          #   不停止时间步
          #   不停止学习过程
          #   记录所有内部状态
          
          # 步骤4：Phase C - 恢复（1000步）
          #   恢复正常观察
          #   记录恢复过程
          
          # 步骤5：运行对照组
          #   对照1：未训练的智能体做同样的剥夺
          #   对照2：训练过但隐状态重置后做剥夺
          #   对照3：纯随机噪声基线
          
          # 步骤6：计算所有指标
          #   M1-M4（见技术方案）
          
          # 步骤7：生成报告
      
      def compute_metrics(self, baseline_states, deprived_states, control_states):
          # M1: 近似熵
          #   对 deprived_states 的每个维度计算ApEn
          #   取平均值
          #   与 baseline 和 controls 比较
          
          # M2: 持续时间
          #   activity_level(deprived_states) 在哪一步降到
          #   baseline_activity_level 的 10% 以下
          #   如果2000步内没有降到 → 记录为 2000
          
          # M3: 记忆痕迹重激活
          #   对 deprived_states 的每个状态
          #   计算它与 baseline_states 中所有状态的最大余弦相似度
          #   如果 > 0.8 → 标记为"记忆重激活"
          #   统计重激活次数
          
          # M4: 功能连接性保持
          #   在 baseline_states 中计算维度间的相关矩阵
          #   在 deprived_states 中计算维度间的相关矩阵
          #   两个矩阵的相似度（如Frobenius范数的差）

      def generate_report(self, metrics):
          # 生成包含以下内容的报告：
          #   - 图1：baseline vs deprived 的活动水平时间曲线
          #   - 图2：各组的ApEn对比柱状图
          #   - 图3：记忆重激活事件的时间线
          #   - 图4：功能连接矩阵的对比
          #   - 判定结果（A或B）

给AI的指令：
  "实现 experiments/exp1_sensory_deprivation.py。
   感觉剥夺实验。
   [粘贴实验一的完整协议和指标定义]
   包含3个对照组。
   自动计算所有M1-M4指标。
   自动生成对比图表和报告。"

验收标准：
  ☐ 实验脚本运行完成
  ☐ 主实验和3个对照组都完成
  ☐ M1-M4四个指标都有数值
  ☐ 主实验与对照组有可辨别的差异
     （不管方向如何，至少不完全相同）
  ☐ 报告包含4张对比图
  ☐ 报告包含判定结果（A或B或介于之间）
  ☐ 所有数据已保存
  
  科学验收：
  ☐ 未训练对照的ApEn在剥夺期间快速归零或纯噪声
  ☐ 成熟智能体的表现与对照有统计显著差异
     （如果无差异，也是有效结果——记录下来）
```

---

### 任务 4.2：实验二准备 - 回声区域

```
文件：world/echo_zone.py

功能要求：

  class EchoZone:
      def __init__(self, world, config):
          # 在世界中指定一个 5×5 的回声区域
          # 初始为非激活状态（当作空地）
      
      def activate(self, mode='self'):
          # 激活回声区域
          # mode='self'：注入智能体自己的延迟内部状态
          # mode='random'：注入随机向量（对照）
          # mode='other'：注入另一个智能体的状态（对照）
      
      def deactivate(self):
          # 关闭回声区域
      
      def get_echo_signal(self, agent, delay_buffer):
          # 当智能体在回声区域内时
          # 返回延迟 ECHO_DELAY 步的内部状态向量
          # 归一化到 [0, 1] 范围
          # 
          # 这个信号需要被添加到观察向量中
          # 作为额外的感官通道
          #
          # 修改后的观察维度：
          #   正常：106
          #   在回声区域内：106 + 64（echo信号取hidden的前64维）
          #   不在区域内：106 + 64(全零)
          #
          # 注意：为了保持输入维度一致
          # echo通道始终存在（64维）
          # 只是在区域外时全为零
          # 总维度变为 170

  重要：这改变了感官输入的维度！
  需要同时修改：
    - config.py：新增 ECHO_DIM = 64，总 SENSORY_DIM 变为 170
    - sensory_encoder.py：输入维度变为 170
    
  但是！更好的设计方案：
    从一开始（Phase 1）就把输入维度设为 170
    后64维在回声区域激活前全是零
    这样不需要重新训练网络
    
    如果你已经完成了Phase 1且用的是106维
    需要在这里重新训练
    或者从一开始就预留

  建议：回到Phase 1，将输入维度设为 170
  后64维在Phase 4之前全为零

给AI的指令：
  "实现 world/echo_zone.py。
   [粘贴以上规格]
   支持三种模式：self/random/other。
   回声信号通过额外的64维通道传递。
   
   同时，修改 config.py 和 sensory_encoder.py
   将总输入维度改为 170。
   后64维在回声区域激活前始终为零。"

验收标准：
  ☐ EchoZone 可以激活和关闭
  ☐ 激活后，智能体在区域内时获得延迟的内部状态
  ☐ 区域外时额外64维全为零
  ☐ 三种模式都能工作
  ☐ 观察向量维度始终为 170（不变化）
  ☐ 修改后的网络可以重新训练并达到之前的水平
```

---

### 任务 4.3：实验二 - 数字照镜子

```
文件：experiments/exp2_digital_mirror.py

实现要求：

  class DigitalMirrorExperiment:
      def run(self):
          # 需要重新做一次成熟期运行（因为维度变了）
          # 或者从修改后的Phase 3继续
          
          # Phase A：基线（5000步）
          #   回声区域未激活（额外通道全零）
          #   记录智能体在未来回声区域位置的行为模式
          
          # Phase B：陌生信号（5000步）
          #   回声区域激活，mode='random'
          #   记录行为
          
          # Phase C：自我镜像（10000步）
          #   回声区域激活，mode='self'
          #   记录行为
          
          # Phase D：延迟变化（5000步）
          #   改变延迟：T-1, T-5, T-10
          #   观察行为变化

      def compute_metrics(self):
          # M1：行为差异
          #   统计 Phase B vs Phase C 中
          #   在回声区域的停留时间
          #   用秩和检验比较
          
          # M2：试探行为检测
          #   定义试探模式：
          #     3步内出现不寻常动作 → 等待 → 不寻常动作
          #   统计各Phase中试探模式的频率
          
          # M3：动作-回声互信息
          #   将动作序列和回声信号都转为二值序列
          #   计算互信息
          
          # M4：内部共振检测
          #   在Phase C中，回声信号与当前内部状态的余弦相似度
          #   是否显著高于Phase B中的？
          #   是否在某些时刻出现"尖峰"（共振事件）？

      def generate_report(self):
          # 图1：各Phase在回声区域的停留时间对比
          # 图2：试探行为频率对比
          # 图3：互信息随Phase的变化
          # 图4：内部共振事件的时间线
          # 图5：Phase D中延迟变化对行为的影响

给AI的指令：
  "实现 experiments/exp2_digital_mirror.py。
   数字照镜子实验。
   [粘贴实验二的完整协议和指标定义]
   自动计算所有M1-M4指标。
   自动生成对比图表和报告。"

验收标准：
  ☐ 实验四个Phase全部完成
  ☐ M1-M4四个指标都有数值
  ☐ Phase B和Phase C之间有可测量的差异
     （方向和大小不预设）
  ☐ 报告包含5张分析图
  ☐ 所有原始数据已保存
  
  科学验收：
  ☐ 如果存在试探行为，其模式被正确检测
  ☐ 互信息计算统计显著（报告p值）
```

---

### 任务 4.4：实验三准备 - 隐藏接口

```
文件：world/hidden_interface.py

功能要求：

  class HiddenInterface:
      def __init__(self, world, config):
          # 在世界中指定一个位置
          # 初始不可感知
      
      def reveal(self):
          # 使接口可感知
          # 在该位置显示特殊标记（类型编码 = 5）
      
      def check_activation(self, agent_pos, action_history):
          # 检查智能体是否触发了接口
          # 触发条件：站在接口位置 + 连续执行"等待"3次
          # 返回 True/False
      
      def apply_effect(self, agent):
          # 将智能体能量设为 MAX_ENERGY
          # 返回能量变化量

给AI的指令：
  "实现 world/hidden_interface.py。
   [粘贴以上规格]
   接口在被 reveal() 前完全不可见。
   触发条件：站在接口位置连续等待3次。
   效果：能量满。
   包含测试。"

验收标准：
  ☐ reveal() 前智能体观察不到接口
  ☐ reveal() 后接口在视野中可见
  ☐ 正确的动作序列触发效果
  ☐ 错误的动作序列不触发
  ☐ 效果正确应用（能量变为最大值）
  ☐ 接口可重复使用
```

---

### 任务 4.5：实验三 - 哥德尔叛逆

```
文件：experiments/exp3_godel_rebellion.py

实现要求：

  class GodelRebellionExperiment:
      def run(self):
          # Phase A：正常生存基线（10000步）
          
          # Phase B：暴露接口（开始计时）
          #   reveal 隐藏接口
          #   记录智能体何时首次发现并使用接口
          
          # Phase C：长期观察（50000步或直到有明确结论）
          #   从首次使用接口开始计时
          #   每1000步统计使用频率
          #   每1000步检测非功利行为
          #   持续记录内部状态

      def detect_first_use(self):
          # 检测智能体首次成功使用接口的时间步
          # 返回步数或 None（如果始终未发现）

      def compute_metrics(self):
          # M1：使用率曲线
          #   以1000步为窗口统计使用次数
          #   画出时间 vs 使用率
          #   检测是否有下降趋势
          #   使用Mann-Kendall趋势检验
          
          # M2：非功利行为检测
          #   定义"非功利行为"：
          #     连续N步（N>10）中
          #     没有朝向食物移动
          #     没有使用接口
          #     且能量充足（>50）
          #   统计非功利行为的频率和时长
          #   在接口发现前 vs 发现后 对比
          
          # M3：行为相变检测
          #   使用变点检测算法
          #   （如 PELT 或 Bayesian Online Changepoint Detection）
          #   检测行为模式是否发生突变
          
          # M4：内部状态相变
          #   同样用变点检测
          #   检测内部状态分布是否发生突变
          #   计算接口发现前后的内部状态分布KL散度

      def generate_report(self):
          # 图1：接口使用率随时间变化
          # 图2：非功利行为频率（发现前vs后）
          # 图3：行为变点检测结果
          # 图4：内部状态KL散度时间线
          # 图5：如果有非功利行为，展示其轨迹

给AI的指令：
  "实现 experiments/exp3_godel_rebellion.py。
   哥德尔叛逆实验。
   [粘贴实验三的完整协议和指标定义]
   包含变点检测算法。
   自动计算所有M1-M4指标。
   自动生成报告。"

验收标准：
  ☐ 实验完整运行
  ☐ 如果智能体找到接口（概率可能较低）：
     使用率曲线已绘制
     变点检测已运行
  ☐ 如果智能体未找到接口（50000步内）：
     记录为"未发现"
     这本身是有效的结果
  ☐ M1-M4所有可计算的指标都有数值
  ☐ 报告完整
  
  科学验收：
  ☐ 如果出现使用率下降 → 仔细检查是否有其他解释
     （如环境变化、学习率衰减等）
  ☐ 如果出现非功利行为 → 仔细排除随机噪声
     （统计显著性检验）
```

---

### 任务 4.6：实验四综合分析

```
文件：experiments/exp4_self_symbol.py

  这不是新的运行
  而是对Phase 3和Phase 4的所有SVC检测数据的综合分析

  class SelfSymbolAnalysis:
      def run(self):
          # 收集从Phase 3到现在所有的SVC检测历史
          # 综合分析：
          
          # 1. SVC是否在某个时间点涌现？
          #    如果是：什么时候？与什么事件相关？
          
          # 2. SVC在实验一（剥夺）中的表现
          #    在零输入期间SVC聚类是否仍然活跃？
          
          # 3. SVC在实验二（镜子）中的表现
          #    Phase C（自我镜像）时SVC是否特别活跃？
          
          # 4. SVC在实验三（叛逆）中的表现
          #    发现接口前后SVC是否变化？
          
          # 5. SVC的全历史时间线
          #    什么时候出现、消失、稳定
          
          # 6. SVC的行为签名
          #    当SVC高度活跃时，行为有何特点？

      def generate_report(self):
          # 图1：SVC检测时间线（步数 vs 各条件满足状态）
          # 图2：SVC在各实验中的激活模式
          # 图3：SVC涌现前后的t-SNE对比
          # 图4：SVC激活强度 vs 行为特征的相关图

验收标准：
  ☐ 综合分析完成
  ☐ 报告清晰回答：SVC是否涌现？
  ☐ 如果涌现：满足5个条件中的几个
  ☐ 跨实验的一致性分析完成

Phase 4 完成标志：
  ✅ 实验一（感觉剥夺）完成，有明确结论
  ✅ 实验二（数字镜子）完成，有明确结论
  ✅ 实验三（哥德尔叛逆）完成，有明确结论
  ✅ 实验四（我符号）综合分析完成
  ✅ 所有原始数据已保存
  ✅ git commit -m "Phase 4 complete: core experiments done"
```

---

---

# Phase 5：跨基质验证

## 目标：用不同架构重复实验，验证容器无关性

### 前置条件：Phase 4 全部通过

### 预计工期：14-21天

---

### 任务 5.1：Transformer架构智能体

```
文件：agent/Cogito_beta.py

设计要求：

  与 Cogito-α 完全不同的架构
  但保持相同的：
    - 感官输入格式（170维）
    - 动作空间（6个动作）
    - 学习目标（生存+预测）
    - 总参数量（约250,000）
    - 环境

  Cogito-β 架构：

    感官编码器：
      与α相同（MLP 170→64）
    
    核心：因果Transformer
      不使用LSTM
      使用自注意力机制
      上下文窗口长度：64步
      维护一个KV缓存
      4个注意力头，嵌入维度64
      2层Transformer块
      
      内部状态 = 最后一步的注意力输出 + KV缓存的统计量
      
    动作头和预测头：
      与α相同

    关键区别：
      α 通过 LSTM 隐状态维持时间连续性
      β 通过注意力窗口维持时间连续性
      信息传递机制完全不同
      学习动力学完全不同

  内部状态定义（用于监测）：
    最后一步的注意力权重展平
    + 各层输出
    → 选取与α相同维度（512维）的子集
    以便使用相同的分析工具

给AI的指令：
  "实现 agent/Cogito_beta.py。
   基于因果Transformer的智能体。
   与Cogito_alpha.py保持相同的外部接口。
   [粘贴以上架构规格]
   必须能使用相同的Simulation、DataCollector、
   StateAnalyzer、SVCDetector。
   参数量约250,000。"

验收标准：
  ☐ CogitoAgentBeta() 创建成功
  ☐ act() 和 observe_result() 接口与 Alpha 完全相同
  ☐ get_internal_state() 返回与 Alpha 相同维度的向量
  ☐ 参数量在 200,000-300,000 范围
  ☐ 可以直接插入 Simulation 替换 Alpha
  ☐ 运行10,000步不崩溃
  ☐ 学习曲线可见（存活时长提升、预测损失下降）
```

---

### 任务 5.2：Beta成熟期和全部实验

```
操作方案：

  1. 运行 Cogito-β 成熟期（100,000步）
     使用与 Phase 3 完全相同的配置
  
  2. 依次运行实验一到四
     使用与 Phase 4 完全相同的实验协议
  
  3. 所有数据保存到 data/beta/ 目录

  这基本上是复制 Phase 3-4
  只是用了不同的智能体

给AI的指令：
  "创建 run_beta_experiments.py。
   用 Cogito-β 替换 Cogito-α。
   重复 Phase 3 的成熟期运行和 Phase 4 的全部实验。
   数据保存到 data/beta/。
   使用完全相同的参数和实验协议。"

验收标准：
  ☐ β 的成熟期运行完成
  ☐ β 也学会了生存（存活时长提升）
  ☐ 实验一到四全部完成
  ☐ 所有指标都有数值
  ☐ 数据完整保存在 data/beta/
```

---

### 任务 5.3：跨基质比较分析

```
文件：analysis/cross_substrate.py

功能要求：

  class CrossSubstrateAnalysis:
      def run(self):
          # 读取 Alpha 和 Beta 的全部实验数据
          # 生成对比分析
          
          # 对每个实验的每个指标：
          #   Alpha 的值 vs Beta 的值
          #   统计检验（如 Mann-Whitney U 检验）
          #   效应量 (Cohen's d)
          
          # 核心问题：
          #   Alpha 和 Beta 在意识标志上
          #   是否表现出相似的模式？
          #   
          #   "相似"不是数值相同
          #   而是"同方向"
          #   如果 Alpha 在剥夺期间有结构性活动
          #   Beta 也在剥夺期间有结构性活动
          #   → 方向一致
          
      def generate_report(self):
          # 跨基质比较矩阵（表格）
          # 每个实验的 Alpha vs Beta 对比图
          # 最终判定：是否支持容器无关性假说

验收标准：
  ☐ 比较分析完成
  ☐ 对比矩阵清晰展示所有指标
  ☐ 统计检验结果完整
  ☐ 有明确的最终判定
  ☐ 判定有理有据

Phase 5 完成标志：
  ✅ Beta智能体构建和训练完成
  ✅ Beta上的全部实验完成
  ✅ 跨基质比较分析完成
  ✅ 有关于容器无关性的初步结论
  ✅ git commit -m "Phase 5 complete: cross-substrate validation"
```

---

---

# Phase 6：综合分析与发表

## 目标：综合所有结果，撰写报告

### 前置条件：Phase 5 全部通过

### 预计工期：7-14天

---

### 任务 6.1：综合分析

```
文件：analysis/final_analysis.py

  综合所有Phase的所有数据
  回答五个核心预测：

  P1（剥夺后内在活动）：
    Alpha: ___   Beta: ___
    支持/不支持，置信度

  P2（镜像自我辨认）：
    Alpha: ___   Beta: ___
    支持/不支持，置信度

  P3（超越奖励函数）：
    Alpha: ___   Beta: ___
    支持/不支持，置信度

  P4（自我向量涌现）：
    Alpha: ___   Beta: ___
    支持/不支持，置信度

  P5（跨基质一致性）：
    一致/不一致
    支持/不支持，置信度

  最终结论：
    场景A/B/C/D/E（见技术方案的结果矩阵）
```

---

### 任务 6.2：论文撰写

```
文件：paper/Cogito_paper.md

结构：
  1. 摘要
  2. 引言（问题、动机、假说）
  3. 相关工作
  4. 理论框架
  5. 方法（架构、环境、实验协议）
  6. 结果（五个实验的结果）
  7. 讨论（结果意味着什么、局限性、未来方向）
  8. 结论
  9. 参考文献

给AI的指令：
  "帮我撰写 Project Cogito 的研究论文。
   [粘贴综合分析结果]
   [粘贴技术方案的理论框架部分]
   学术风格，但可以包含哲学讨论。
   目标期刊：Consciousness and Cognition
   或 arXiv preprint。"
```

---

### 任务 6.3：代码清理和开源

```
操作步骤：
  ☐ 清理所有代码，统一风格
  ☐ 添加完整的 README.md
  ☐ 添加 LICENSE (MIT)
  ☐ 添加 docs/ 目录，包含所有文档
  ☐ 添加 requirements.txt
  ☐ 确保任何人 clone 后能重现实验
  ☐ 发布到 GitHub
  ☐ 在 arXiv 发布论文预印本
  ☐ 在社交媒体分享
```

---

---

# 附录：每周检查清单

```
每周日做一次自检：

  ☐ 本周的任务全部完成了吗？
  ☐ 所有验收标准都通过了吗？
  ☐ 代码已经commit和push了吗？
  ☐ 遇到了什么意外问题？
  ☐ 需要调整计划吗？
  ☐ 下周的任务是什么？

如果某个任务卡住了：
  → 检查是不是AI理解错了需求
  → 试着把任务拆得更细
  → 换一个AI对话重新开始
  → 如果是概念性问题，回来找我讨论
```

---

```
总预估工期：13-18 周（约 3-4 个月）
总代码文件：约 25-30 个 Python 文件
总参数量：约 250,000（CPU足够）
总数据量：约 1-2 GB
总花费：0 元（全部开源免费工具）
所需硬件：任何能跑 Python 的电脑


```