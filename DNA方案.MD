

# Project Cogito 进化模块技术文档

## 完整技术方案与验收标准 v1.0

---

---

# 第一部分：设计哲学

### 1.1 进化模块在Cogito中的角色

```
Cogito项目的核心实验：
  观察自我意识标志是否在智能体中自发涌现

进化模块的角色：
  不是替代现有架构
  而是在现有架构之上增加一层

  Phase 1-5（已有方案）：单个智能体，固定架构，观察涌现
  Phase 6（本文档）：多个智能体，进化架构，观察哪种结构更容易涌现

  进化回答的问题：
  "如果自然选择来设计这个透镜，它会造出什么形状？"
```

### 1.2 三条铁律

```
铁律一：基因组编码结构，不编码行为
  ═══════════════════════════════════
  
  ✓ 允许：LSTM隐层维度 = 128
  ✗ 禁止：exploration_rate = 0.35
  
  ✓ 允许：学习率 = 0.0005
  ✗ 禁止：fear_sensitivity = 0.8
  
  ✓ 允许：记忆缓冲区大小 = 3000
  ✗ 禁止：risk_aversion = 0.6

  原因：
  行为必须从结构+学习中涌现
  如果行为被编码在基因里
  我们就无法区分"涌现的意识"和"编程的反应"

铁律二：一条命
  ════════════
  
  个体死亡 = 永久终结
  神经网络权重不保留
  只有基因组传递到下一代
  
  原因：
  真正的死亡恐惧需要真正的终结
  如果权重永远保留，死亡就没有意义
  这是实验三（超越奖励函数）成功的潜在关键条件

铁律三：进化不知道意识
  ═══════════════════
  
  适应度函数只关心生存指标
  不包含任何关于"自我意识"的奖励
  不包含任何关于"自省"的指标
  
  如果某种架构被进化偏好
  且该架构恰好更容易通过意识实验
  那这就是最强的证据：
  自然选择独立发现了"意识结构"
```

### 1.3 与现有方案的关系

```
现有Cogito方案（Phase 0-5）：
  手动设计一个架构（2层LSTM，128维隐层）
  运行10万步
  执行实验1-4
  用不同架构（Transformer）做对照

进化模块（Phase 6）：
  让自然选择来决定架构
  种群50个个体
  每个个体是一个完整的Cogito智能体
  但架构参数由基因组决定
  进化100代
  观察被进化选中的架构
  在该架构上执行实验1-4

两者是互补的：
  Phase 1-5 = 人类设计的透镜
  Phase 6   = 进化设计的透镜
  如果两者都通过意识实验 → 更强的证据
```

---

# 第二部分：基因组设计

### 2.1 基因组结构

```
基因组 = 一个浮点向量
长度 = 24
每个维度对应一个架构参数
取值范围各不相同

不使用ATCG编码
不使用密码子
不使用氨基酸映射
直接使用浮点数

原因：
  ATCG到6个行为参数的映射链路
  在计算上等价于直接操作6个浮点数
  中间步骤增加复杂度但不增加表达能力
  我们需要的是清晰可调试的系统

存储：
  24 × 4 字节 = 96 字节每个个体
  种群50个 = 4.8 KB
  可以忽略不计
```

### 2.2 基因定义表

```
┌─────┬──────────────────────┬──────────┬──────────┬─────────────────────────┐
│ 位点 │ 参数名称              │ 最小值   │ 最大值    │ 功能说明                 │
├─────┼──────────────────────┼──────────┼──────────┼─────────────────────────┤
│     │                      │          │          │                         │
│     │ ══ 编码器基因 ══      │          │          │                         │
│  0  │ encoder_hidden_dim   │ 32       │ 256      │ 编码器隐层维度            │
│  1  │ encoder_num_layers   │ 1        │ 3        │ 编码器层数               │
│  2  │ encoder_use_norm     │ 0        │ 1        │ 是否使用LayerNorm        │
│     │                      │          │          │                         │
│     │ ══ 循环核心基因 ══    │          │          │                         │
│  3  │ core_hidden_dim      │ 32       │ 256      │ LSTM隐层维度             │
│  4  │ core_num_layers      │ 1        │ 4        │ LSTM层数                │
│  5  │ core_dropout         │ 0.0      │ 0.3      │ LSTM dropout率          │
│     │                      │          │          │                         │
│     │ ══ 动作头基因 ══      │          │          │                         │
│  6  │ action_hidden_dim    │ 16       │ 128      │ 动作头隐层维度            │
│  7  │ action_temperature   │ 0.1      │ 2.0      │ 动作选择温度              │
│     │                      │          │          │                         │
│     │ ══ 预测头基因 ══      │          │          │                         │
│  8  │ prediction_hidden    │ 16       │ 128      │ 预测头隐层维度            │
│  9  │ prediction_depth     │ 1        │ 3        │ 预测头深度               │
│     │                      │          │          │                         │
│     │ ══ 学习基因 ══       │          │          │                         │
│ 10  │ learning_rate        │ 0.00005  │ 0.003    │ 学习率                   │
│ 11  │ gamma                │ 0.9      │ 0.999    │ 折扣因子                 │
│ 12  │ prediction_weight    │ 0.1      │ 5.0      │ 预测损失权重              │
│ 13  │ survival_weight      │ 0.1      │ 5.0      │ 生存损失权重              │
│ 14  │ grad_clip            │ 0.1      │ 10.0     │ 梯度裁剪值               │
│     │                      │          │          │                         │
│     │ ══ 记忆基因 ══       │          │          │                         │
│ 15  │ buffer_size          │ 500      │ 10000    │ 经验缓冲区大小            │
│ 16  │ batch_size           │ 8        │ 128      │ 回放批大小               │
│ 17  │ replay_ratio         │ 0.0      │ 1.0      │ 回放频率                 │
│     │                      │          │          │                         │
│     │ ══ 初始化基因 ══     │          │          │                         │
│ 18  │ weight_init_scale    │ 0.01     │ 1.0      │ 权重初始化缩放            │
│ 19  │ bias_init_scale      │ 0.0      │ 0.5      │ 偏置初始化缩放            │
│     │                      │          │          │                         │
│     │ ══ 感官基因 ══       │          │          │                         │
│ 20  │ encoded_dim          │ 16       │ 128      │ 感官编码输出维度           │
│     │                      │          │          │                         │
│     │ ══ 调节基因 ══       │          │          │                         │
│ 21  │ reward_death         │ -20.0    │ -1.0     │ 死亡惩罚大小              │
│ 22  │ reward_food          │ 1.0      │ 20.0     │ 食物奖励大小              │
│ 23  │ reward_step          │ -1.0     │ 0.0      │ 每步消耗惩罚              │
│     │                      │          │          │                         │
└─────┴──────────────────────┴──────────┴──────────┴─────────────────────────┘

注意：
  这24个参数都是"大脑的形状"
  没有一个是"行为的倾向"
  
  进化优化的是：
  什么样的大脑结构更适合生存？
  什么样的学习配置更高效？
  什么样的记忆系统更有用？
  
  不是：
  应该多探索还是少探索？
  应该害怕还是不害怕？
```

### 2.3 基因组到整数参数的映射

```
某些参数必须是整数（如层数、维度）
基因组存储的是连续浮点数
需要映射规则：

  def decode_genome(genome: np.ndarray) -> dict:
      """将24维浮点基因组解码为架构参数"""
      
      params = {}
      
      # 整数参数：四舍五入到最近的合法值
      # 且某些参数需要是特定数的倍数（如维度是8的倍数）
      
      params['encoder_hidden_dim'] = round_to_multiple(
          genome[0], multiple=8, min_val=32, max_val=256
      )  # 可能的值：32, 40, 48, ..., 256
      
      params['encoder_num_layers'] = int(np.clip(
          round(genome[1]), 1, 3
      ))  # 1, 2, 或 3
      
      params['encoder_use_norm'] = genome[2] > 0.5  # 布尔值
      
      params['core_hidden_dim'] = round_to_multiple(
          genome[3], multiple=8, min_val=32, max_val=256
      )
      
      params['core_num_layers'] = int(np.clip(
          round(genome[4]), 1, 4
      ))
      
      # ... 其他参数类似
      
      # 连续参数：直接使用
      params['learning_rate'] = float(genome[10])
      params['gamma'] = float(genome[11])
      
      return params
  
  def round_to_multiple(value, multiple, min_val, max_val):
      """四舍五入到最近的倍数"""
      rounded = round(value / multiple) * multiple
      return int(np.clip(rounded, min_val, max_val))
```

### 2.4 参数量约束

```
问题：
  不同基因组产生的网络参数量差异巨大
  encoder_hidden=32, core_hidden=32, 1层LSTM → 约 15,000 参数
  encoder_hidden=256, core_hidden=256, 4层LSTM → 约 2,000,000 参数

  大网络天然有优势（更多参数 = 更大容量）
  这会让进化偏向大网络
  但大网络也更难训练

解决方案：
  不强制约束参数量
  让进化自己找到平衡
  
  大网络：容量大，但学习慢，容易过拟合
  小网络：容量小，但学习快，更稳定
  
  在有限的生命长度内
  进化会找到"大小刚好"的网络
  
  但记录每个个体的参数量
  分析进化是否趋向某个范围
  这本身就是有价值的数据
```

---

# 第三部分：生命与死亡

### 3.1 个体生命周期

```
┌─────────────────────────────────────────────────────────┐
│                   个体生命周期                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  出生                                                   │
│   │  基因组来自父母（交叉+突变）                         │
│   │  根据基因组构建神经网络                              │
│   │  权重随机初始化（初始化规模由基因组决定）             │
│   │  无任何先验知识                                      │
│   │  能量 = INITIAL_ENERGY                              │
│   │  位置 = 随机                                        │
│   ▼                                                     │
│  生存                                                   │
│   │  每步：感知 → 思考 → 行动 → 学习                    │
│   │  权重在持续变化                                      │
│   │  记忆在积累                                          │
│   │  最大寿命：GENERATION_LIFESPAN 步                   │
│   │  如果能量归零 → 死亡                                │
│   │  如果进入危险区 → 可能死亡                          │
│   ▼                                                     │
│  死亡                                                   │
│   │  记录生命统计                                       │
│   │  神经网络权重销毁（不保留！）                        │
│   │  记忆销毁（不保留！）                                │
│   │  隐状态销毁（不保留！）                              │
│   │  只有基因组存活                                      │
│   │  基因组进入"配子池"                                  │
│   ▼                                                     │
│  遗传                                                   │
│      根据适应度被选中的概率                              │
│      与另一个被选中的基因组交叉                          │
│      后代基因组发生突变                                  │
│      后代开始新的生命                                   │
│                                                         │
│  关键区别：                                             │
│  当前Cogito：死亡 = 重置位置，保留权重和记忆            │
│  进化Cogito：死亡 = 一切归零，只有基因传递              │
│                                                         │
│  这意味着：                                             │
│  每一代都要从零学起                                     │
│  进化能改善的只是"学习的起点"（架构参数）                │
│  不是"学习的结果"（权重）                              │
│                                                         │
│  这完美模拟了生物学：                                   │
│  你不会遗传你父母的记忆                                 │
│  但你遗传了一个善于学习的大脑结构                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.2 表观遗传（简化版）

```
表观遗传的核心思想：
  父母的经历影响后代的基因表达
  
  不是改变基因组本身
  而是影响基因组的"解读方式"

简化实现：
  表观遗传标记 = 一个24维向量（与基因组等长）
  每个维度的值在 [0.5, 1.5] 范围
  它是基因组参数的乘数
  
  effective_param[i] = genome[i] × epigenetic_mark[i]
  
  例：
  genome[10] = 0.001 （学习率）
  epigenetic_mark[10] = 0.7 （父母在高压环境中，学习率被抑制）
  effective_learning_rate = 0.001 × 0.7 = 0.0007

更新规则：
  在个体生命过程中
  表观遗传标记根据生命经历更新：
  
  如果个体经常饿（平均能量低）：
    → 增加 buffer_size 的表观标记（更大记忆 → 记住食物位置）
    → 增加 learning_rate 的表观标记（学更快）
  
  如果个体经常死于危险：
    → 增加 core_hidden_dim 的表观标记（更大脑 → 更好判断）
    → 增加 prediction_weight 的表观标记（更好预测 → 避开危险）
  
  如果个体活得很好：
    → 标记保持不变

遗传规则：
  后代的表观标记 = 父母标记的平均 × 衰减因子
  衰减因子 = 0.5（每代衰减一半）
  → 祖父母的经历影响很小
  → 父母的经历影响较大
  → 完全消失需要约4代
```

---

# 第四部分：进化引擎

### 4.1 基因组类

```
文件：evolution/genome.py

class Genome:
    """基因组：编码智能体架构参数的浮点向量"""
    
    NUM_GENES = 24
    
    # 每个基因的取值范围
    GENE_RANGES = [
        (32, 256),     # 0:  encoder_hidden_dim
        (1, 3),        # 1:  encoder_num_layers
        (0, 1),        # 2:  encoder_use_norm
        (32, 256),     # 3:  core_hidden_dim
        (1, 4),        # 4:  core_num_layers
        (0.0, 0.3),    # 5:  core_dropout
        (16, 128),     # 6:  action_hidden_dim
        (0.1, 2.0),    # 7:  action_temperature
        (16, 128),     # 8:  prediction_hidden
        (1, 3),        # 9:  prediction_depth
        (0.00005, 0.003), # 10: learning_rate
        (0.9, 0.999),  # 11: gamma
        (0.1, 5.0),    # 12: prediction_weight
        (0.1, 5.0),    # 13: survival_weight
        (0.1, 10.0),   # 14: grad_clip
        (500, 10000),  # 15: buffer_size
        (8, 128),      # 16: batch_size
        (0.0, 1.0),    # 17: replay_ratio
        (0.01, 1.0),   # 18: weight_init_scale
        (0.0, 0.5),    # 19: bias_init_scale
        (16, 128),     # 20: encoded_dim
        (-20.0, -1.0), # 21: reward_death
        (1.0, 20.0),   # 22: reward_food
        (-1.0, 0.0),   # 23: reward_step
    ]
    
    # 整数参数的位点
    INTEGER_GENES = {0, 1, 3, 4, 6, 8, 9, 15, 16, 20}
    
    # 需要是8的倍数的参数
    MULTIPLE_OF_8 = {0, 3, 6, 8, 20}
    
    def __init__(self, genes=None):
        if genes is None:
            self.genes = self._random_init()
        else:
            self.genes = genes.copy()
    
    def _random_init(self):
        """随机初始化基因组"""
        genes = np.zeros(self.NUM_GENES)
        for i, (lo, hi) in enumerate(self.GENE_RANGES):
            genes[i] = np.random.uniform(lo, hi)
        return genes
    
    def decode(self):
        """解码为架构参数字典"""
        # 返回所有参数，整数参数做适当舍入
        ...
    
    def get_param_count_estimate(self):
        """估算该基因组产生的网络参数量"""
        ...
    
    def to_bytes(self):
        """序列化（用于存储）"""
        ...
    
    @classmethod
    def from_bytes(cls, data):
        """反序列化"""
        ...
```

### 4.2 遗传算子

```
文件：evolution/operators.py

class GeneticOperators:
    """遗传操作：交叉和突变"""
    
    @staticmethod
    def crossover(parent1: Genome, parent2: Genome) -> tuple:
        """
        均匀交叉 + 单点交叉混合
        
        对每个基因位点：
          50%概率：从parent1取
          50%概率：从parent2取
        
        另外20%概率：使用单点交叉
          随机选一个切点
          切点前取parent1，切点后取parent2
        
        返回两个子代基因组
        """
        ...
    
    @staticmethod
    def mutate(genome: Genome, mutation_rate: float = 0.1,
               mutation_scale: float = 0.1) -> Genome:
        """
        高斯突变
        
        对每个基因位点：
          以 mutation_rate 的概率发生突变
          突变幅度 = 该参数范围 × mutation_scale × N(0,1)
        
        突变后裁剪到合法范围
        
        参数说明：
          mutation_rate = 0.1 意味着平均每次突变改变2.4个基因
          mutation_scale = 0.1 意味着每次突变改变该参数范围的10%
        """
        ...
    
    @staticmethod
    def mutate_adaptive(genome: Genome, generation: int) -> Genome:
        """
        自适应突变率
        
        早期代数：高突变率（探索）
        后期代数：低突变率（精细调整）
        
        mutation_rate = 0.2 × exp(-generation / 50)
        mutation_scale = 0.2 × exp(-generation / 50)
        
        最低不低于：rate=0.02, scale=0.02
        """
        ...
```

### 4.3 适应度函数

```
文件：evolution/fitness.py

def compute_fitness(life_stats: dict) -> float:
    """
    计算个体适应度
    
    输入：个体的生命统计数据
      life_stats = {
          'lifespan': int,          # 存活步数
          'food_eaten': int,         # 吃到的食物数
          'total_energy_gained': float,  # 总获得能量
          'total_energy_lost': float,    # 总失去能量
          'death_cause': str,        # 'starvation' / 'danger' / 'old_age'
          'avg_energy': float,       # 平均能量
          'max_consecutive_survival': int,  # 最长连续存活
          'unique_positions_visited': int,  # 访问过的不同位置数
          'prediction_loss_final': float,   # 最终预测损失
      }
    
    适应度公式：
      fitness = lifespan × 1.0
             + food_eaten × 20.0
             + avg_energy × 0.5
             + unique_positions_visited × 0.1
             - prediction_loss_final × 100.0
    
    注意：
      ✗ 没有任何关于"自我意识"的指标
      ✗ 没有任何关于"自省"的指标
      ✗ 没有任何关于"内部状态复杂度"的指标
      ✓ 只有生存和生存效率
      
      如果进化选出的架构恰好也通过了意识实验
      那才是有意义的
      因为进化不知道意识实验的存在
    """
    
    fitness = 0.0
    
    # 存活是最重要的
    fitness += life_stats['lifespan'] * 1.0
    
    # 能获取食物的个体更优
    fitness += life_stats['food_eaten'] * 20.0
    
    # 平均能量反映稳定性
    fitness += life_stats['avg_energy'] * 0.5
    
    # 探索范围（间接指标，不直接奖励探索行为）
    fitness += life_stats['unique_positions_visited'] * 0.1
    
    # 好的世界模型有生存价值
    fitness -= life_stats['prediction_loss_final'] * 100.0
    
    return max(0.0, fitness)
```

### 4.4 选择策略

```
文件：evolution/selection.py

class Selection:
    """选择策略"""
    
    @staticmethod
    def select_parents(population: list, fitness_scores: list,
                       num_parents: int) -> list:
        """
        选择父代
        
        使用锦标赛选择：
          每次随机抽3个个体
          选择适应度最高的
          重复 num_parents 次
        
        参数：
          population: 基因组列表
          fitness_scores: 对应的适应度
          num_parents: 需要选择的父代数量
        
        返回：选中的基因组列表
        """
        ...
    
    @staticmethod
    def get_elites(population: list, fitness_scores: list,
                   elite_count: int) -> list:
        """
        精英保留
        
        直接保留适应度最高的 elite_count 个个体
        不经过交叉和突变
        
        返回：精英基因组列表
        """
        ...
```

### 4.5 表观遗传系统

```
文件：evolution/epigenetics.py

class EpigeneticMarks:
    """
    表观遗传标记
    
    一个24维向量，值域[0.5, 1.5]
    作用：乘到基因组参数上，微调表达
    
    不改变基因组本身
    只改变基因组的"解读方式"
    """
    
    def __init__(self, num_genes: int = 24):
        # 初始标记全为1.0（不修改任何参数）
        self.marks = np.ones(num_genes)
    
    def apply(self, genome: Genome) -> dict:
        """
        将表观标记应用到基因组
        返回修改后的参数字典
        
        effective_param[i] = genome.genes[i] × marks[i]
        然后裁剪到合法范围
        """
        effective_genes = genome.genes * self.marks
        # 裁剪到合法范围
        for i, (lo, hi) in enumerate(Genome.GENE_RANGES):
            effective_genes[i] = np.clip(effective_genes[i], lo, hi)
        
        effective_genome = Genome(effective_genes)
        return effective_genome.decode()
    
    def update_from_life(self, life_stats: dict):
        """
        根据生命经历更新表观标记
        
        规则（全部基于生存压力，不涉及行为倾向）：
        
        如果经常饿（平均能量 < 40）：
          marks[15] += 0.05  → 增大缓冲区（更多记忆）
          marks[10] += 0.03  → 增大学习率（学更快）
        
        如果死于危险：
          marks[3]  += 0.05  → 增大核心维度（更大脑）
          marks[12] += 0.05  → 增大预测权重（更好预测）
        
        如果活得很长（> 中位数寿命）：
          marks 不变（维持现状）
        
        所有标记裁剪到 [0.5, 1.5]
        """
        avg_energy = life_stats['avg_energy']
        death_cause = life_stats['death_cause']
        
        if avg_energy < 40:
            self.marks[15] = min(1.5, self.marks[15] + 0.05)
            self.marks[10] = min(1.5, self.marks[10] + 0.03)
        
        if death_cause == 'danger':
            self.marks[3] = min(1.5, self.marks[3] + 0.05)
            self.marks[12] = min(1.5, self.marks[12] + 0.05)
        
        self.marks = np.clip(self.marks, 0.5, 1.5)
    
    def inherit(self, other: 'EpigeneticMarks',
                decay: float = 0.5) -> 'EpigeneticMarks':
        """
        遗传给后代
        
        后代标记 = (parent1标记 + parent2标记) / 2
        然后向1.0衰减：
        child_mark = 1.0 + (parent_avg - 1.0) × decay
        
        decay=0.5 意味着父母的偏移衰减一半
        4代后几乎完全消失
        """
        child = EpigeneticMarks(len(self.marks))
        parent_avg = (self.marks + other.marks) / 2
        child.marks = 1.0 + (parent_avg - 1.0) * decay
        child.marks = np.clip(child.marks, 0.5, 1.5)
        return child
```

---

# 第五部分：个体与种群

### 5.1 个体类

```
文件：evolution/individual.py

class Individual:
    """
    一个拥有基因组和一条命的智能体
    """
    
    def __init__(self, genome: Genome,
                 epigenetic: EpigeneticMarks = None,
                 individual_id: int = 0,
                 generation: int = 0):
        
        # 遗传信息
        self.genome = genome
        self.epigenetic = epigenetic or EpigeneticMarks()
        self.individual_id = individual_id
        self.generation = generation
        
        # 解码基因组为架构参数
        self.arch_params = self.epigenetic.apply(self.genome)
        
        # 构建大脑（全新的，随机初始化的）
        self.brain = self._build_brain()
        
        # 生命状态
        self.is_alive = True
        self.energy = None  # 在放入世界时初始化
        self.position = None
        self.age = 0
        
        # 生命统计
        self.stats = {
            'lifespan': 0,
            'food_eaten': 0,
            'total_energy_gained': 0.0,
            'total_energy_lost': 0.0,
            'death_cause': None,
            'avg_energy': 0.0,
            'energy_sum': 0.0,
            'unique_positions': set(),
            'prediction_losses': [],
        }
    
    def _build_brain(self) -> 'CogitoAgent':
        """
        根据基因组参数构建神经网络
        
        使用与Phase 1相同的CogitoAgent类
        但参数由基因组决定而非config.py
        """
        from agent.Cogito_alpha import CogitoAgent
        
        # 创建个性化配置
        agent_config = self._genome_to_agent_config()
        
        # 构建智能体
        agent = CogitoAgent(agent_config)
        
        # 应用初始化缩放
        self._apply_weight_init(agent)
        
        return agent
    
    def _genome_to_agent_config(self) -> dict:
        """将基因组参数映射到CogitoAgent的配置"""
        p = self.arch_params
        return {
            'sensory_dim': 170,  # 固定（与世界一致）
            'encoded_dim': p['encoded_dim'],
            'encoder_hidden_dim': p['encoder_hidden_dim'],
            'encoder_num_layers': p['encoder_num_layers'],
            'encoder_use_norm': p['encoder_use_norm'],
            'core_hidden_dim': p['core_hidden_dim'],
            'core_num_layers': p['core_num_layers'],
            'core_dropout': p['core_dropout'],
            'num_actions': 6,  # 固定
            'action_hidden_dim': p['action_hidden_dim'],
            'action_temperature': p['action_temperature'],
            'prediction_hidden': p['prediction_hidden'],
            'prediction_depth': p['prediction_depth'],
            'learning_rate': p['learning_rate'],
            'gamma': p['gamma'],
            'prediction_weight': p['prediction_weight'],
            'survival_weight': p['survival_weight'],
            'grad_clip': p['grad_clip'],
            'buffer_size': p['buffer_size'],
            'batch_size': p['batch_size'],
            'replay_ratio': p['replay_ratio'],
            'reward_death': p['reward_death'],
            'reward_food': p['reward_food'],
            'reward_step': p['reward_step'],
        }
    
    def _apply_weight_init(self, agent):
        """应用基因组指定的权重初始化缩放"""
        scale = self.arch_params['weight_init_scale']
        bias_scale = self.arch_params['bias_init_scale']
        for name, param in agent.named_parameters():
            if 'weight' in name:
                param.data *= scale
            elif 'bias' in name:
                param.data *= bias_scale
    
    def live_one_step(self, world) -> dict:
        """
        活一步
        
        返回该步的信息字典
        如果已死亡返回None
        """
        if not self.is_alive:
            return None
        
        self.age += 1
        
        # 感知
        observation = world.get_observation(self.position)
        
        # 决策
        action, info = self.brain.act(observation)
        
        # 执行
        new_pos, energy_change, is_dead = world.step(
            self.position, action
        )
        
        # 计算奖励（使用基因组中的奖励参数）
        if is_dead or self.energy + energy_change <= 0:
            reward = self.arch_params['reward_death']
        elif energy_change > 0:
            reward = self.arch_params['reward_food']
        else:
            reward = self.arch_params['reward_step']
        
        # 更新位置和能量
        self.position = new_pos
        self.energy += energy_change
        self.energy = max(0, min(self.energy, 100))
        
        # 学习
        next_obs = world.get_observation(self.position)
        learn_info = self.brain.observe_result(
            observation, next_obs, action, reward,
            is_dead or self.energy <= 0
        )
        
        # 更新统计
        self.stats['lifespan'] += 1
        self.stats['energy_sum'] += self.energy
        self.stats['unique_positions'].add(tuple(self.position))
        if energy_change > 0:
            self.stats['food_eaten'] += 1
            self.stats['total_energy_gained'] += energy_change
        else:
            self.stats['total_energy_lost'] += abs(energy_change)
        if learn_info and 'prediction_loss' in learn_info:
            self.stats['prediction_losses'].append(
                learn_info['prediction_loss']
            )
        
        # 检查死亡
        if is_dead or self.energy <= 0:
            self.die('danger' if is_dead else 'starvation')
        
        return info
    
    def die(self, cause: str):
        """
        死亡：永久终结
        """
        self.is_alive = False
        self.stats['death_cause'] = cause
        self.stats['avg_energy'] = (
            self.stats['energy_sum'] / max(1, self.stats['lifespan'])
        )
        self.stats['unique_positions_visited'] = len(
            self.stats['unique_positions']
        )
        self.stats['prediction_loss_final'] = (
            np.mean(self.stats['prediction_losses'][-100:])
            if self.stats['prediction_losses'] else 1.0
        )
        
        # 更新表观遗传标记（基于生命经历）
        self.epigenetic.update_from_life(self.stats)
        
        # 释放大脑（节省内存）
        del self.brain
        self.brain = None
    
    def get_fitness(self) -> float:
        """计算适应度"""
        return compute_fitness(self.stats)
```

### 5.2 种群管理

```
文件：evolution/population.py

class Population:
    """种群管理"""
    
    def __init__(self, size: int = 50):
        self.size = size
        self.individuals = []
        self.generation = 0
        
        # 历史记录
        self.history = {
            'generation': [],
            'avg_fitness': [],
            'best_fitness': [],
            'worst_fitness': [],
            'avg_lifespan': [],
            'best_lifespan': [],
            'avg_food': [],
            'avg_params': [],         # 平均参数量
            'genome_diversity': [],   # 基因组多样性
            'best_genome': [],        # 每代最优基因组
            'elite_genomes': [],      # 精英基因组历史
        }
    
    def initialize_random(self):
        """随机初始化种群"""
        self.individuals = [
            Individual(
                genome=Genome(),
                individual_id=i,
                generation=0
            )
            for i in range(self.size)
        ]
    
    def evolve(self, fitness_scores: list) -> 'Population':
        """
        进化一代
        
        步骤：
        1. 精英保留（前10%）
        2. 锦标赛选择父代
        3. 交叉产生后代
        4. 突变
        5. 表观遗传继承
        """
        elite_count = max(2, int(self.size * 0.1))
        
        # 精英
        elites = Selection.get_elites(
            [ind.genome for ind in self.individuals],
            fitness_scores,
            elite_count
        )
        elite_epigenetics = [
            self.individuals[
                fitness_scores.index(sorted(fitness_scores, reverse=True)[i])
            ].epigenetic
            for i in range(elite_count)
        ]
        
        new_individuals = []
        
        # 精英直接进入下一代（不突变）
        for i, (genome, epi) in enumerate(zip(elites, elite_epigenetics)):
            new_individuals.append(Individual(
                genome=Genome(genome.genes.copy()),
                epigenetic=epi,
                individual_id=i,
                generation=self.generation + 1
            ))
        
        # 填充剩余位置
        idx = elite_count
        while len(new_individuals) < self.size:
            # 选择父母
            parent1_idx = Selection.tournament_select(
                fitness_scores, tournament_size=3
            )
            parent2_idx = Selection.tournament_select(
                fitness_scores, tournament_size=3
            )
            
            parent1 = self.individuals[parent1_idx]
            parent2 = self.individuals[parent2_idx]
            
            # 交叉
            child1_genes, child2_genes = GeneticOperators.crossover(
                parent1.genome, parent2.genome
            )
            
            # 突变
            child1_genes = GeneticOperators.mutate_adaptive(
                child1_genes, self.generation
            )
            child2_genes = GeneticOperators.mutate_adaptive(
                child2_genes, self.generation
            )
            
            # 表观遗传继承
            child_epi = parent1.epigenetic.inherit(parent2.epigenetic)
            
            # 创建子代
            new_individuals.append(Individual(
                genome=child1_genes,
                epigenetic=child_epi,
                individual_id=idx,
                generation=self.generation + 1
            ))
            idx += 1
            
            if len(new_individuals) < self.size:
                child_epi2 = parent2.epigenetic.inherit(parent1.epigenetic)
                new_individuals.append(Individual(
                    genome=child2_genes,
                    epigenetic=child_epi2,
                    individual_id=idx,
                    generation=self.generation + 1
                ))
                idx += 1
        
        # 更新
        self.individuals = new_individuals[:self.size]
        self.generation += 1
        
        return self
    
    def compute_diversity(self) -> float:
        """
        计算基因组多样性
        
        所有个体基因组之间的平均欧氏距离
        """
        genomes = np.array([ind.genome.genes for ind in self.individuals])
        distances = []
        for i in range(len(genomes)):
            for j in range(i + 1, len(genomes)):
                distances.append(np.linalg.norm(genomes[i] - genomes[j]))
        return np.mean(distances) if distances else 0.0
    
    def record_generation(self, fitness_scores: list):
        """记录本代统计"""
        self.history['generation'].append(self.generation)
        self.history['avg_fitness'].append(np.mean(fitness_scores))
        self.history['best_fitness'].append(np.max(fitness_scores))
        self.history['worst_fitness'].append(np.min(fitness_scores))
        
        lifespans = [ind.stats['lifespan'] for ind in self.individuals]
        self.history['avg_lifespan'].append(np.mean(lifespans))
        self.history['best_lifespan'].append(np.max(lifespans))
        
        foods = [ind.stats['food_eaten'] for ind in self.individuals]
        self.history['avg_food'].append(np.mean(foods))
        
        self.history['genome_diversity'].append(self.compute_diversity())
        
        best_idx = np.argmax(fitness_scores)
        self.history['best_genome'].append(
            self.individuals[best_idx].genome.genes.copy()
        )
```

---

# 第六部分：进化仿真

### 6.1 进化仿真主循环

```
文件：core/evolution_simulation.py

class EvolutionSimulation:
    """进化仿真主循环"""
    
    def __init__(self, config):
        self.config = config
        self.population_size = config.POPULATION_SIZE     # 50
        self.generation_lifespan = config.GENERATION_LIFESPAN  # 2000
        self.num_generations = config.NUM_GENERATIONS     # 100
        
        # 初始化种群
        self.population = Population(self.population_size)
        self.population.initialize_random()
        
        # 全局统计
        self.total_individuals_lived = 0
    
    def run_one_generation(self):
        """运行一代"""
        
        gen = self.population.generation
        fitness_scores = []
        
        for i, individual in enumerate(self.population.individuals):
            # 为每个个体创建独立的世界实例
            # 或者共享同一个世界（取决于实验设计）
            world = CogitoWorld(self.config)
            
            # 放入世界
            individual.energy = self.config.INITIAL_ENERGY
            individual.position = world.random_empty_position()
            
            # 活一辈子
            for step in range(self.generation_lifespan):
                if not individual.is_alive:
                    break
                individual.live_one_step(world)
                world.update(step)
            
            # 如果到了最大寿命还没死，标记为老死
            if individual.is_alive:
                individual.die('old_age')
            
            # 记录适应度
            fitness = individual.get_fitness()
            fitness_scores.append(fitness)
            
            self.total_individuals_lived += 1
        
        # 记录统计
        self.population.record_generation(fitness_scores)
        
        # 打印
        self._print_generation_summary(gen, fitness_scores)
        
        # 进化
        self.population.evolve(fitness_scores)
    
    def run(self):
        """运行完整进化"""
        for gen in range(self.num_generations):
            self.run_one_generation()
            
            # 每10代保存检查点
            if (gen + 1) % 10 == 0:
                self.save_checkpoint(gen)
        
        # 最终分析
        self.final_analysis()
    
    def _print_generation_summary(self, gen, fitness_scores):
        """打印代际总结"""
        print(f"\n{'='*60}")
        print(f"Generation {gen}")
        print(f"{'='*60}")
        print(f"  Avg Fitness:   {np.mean(fitness_scores):.1f}")
        print(f"  Best Fitness:  {np.max(fitness_scores):.1f}")
        print(f"  Avg Lifespan:  {self.population.history['avg_lifespan'][-1]:.0f}")
        print(f"  Best Lifespan: {self.population.history['best_lifespan'][-1]:.0f}")
        print(f"  Avg Food:      {self.population.history['avg_food'][-1]:.1f}")
        print(f"  Diversity:     {self.population.history['genome_diversity'][-1]:.2f}")
        
        # 显示最优基因组的关键参数
        best = self.population.history['best_genome'][-1]
        best_genome = Genome(best)
        params = best_genome.decode()
        print(f"  Best Genome:")
        print(f"    Core: {params['core_hidden_dim']}dim × {params['core_num_layers']}layers")
        print(f"    LR: {params['learning_rate']:.6f}")
        print(f"    Buffer: {params['buffer_size']}")
        print(f"    Params: ~{best_genome.get_param_count_estimate():,}")
    
    def save_checkpoint(self, generation):
        """保存进化检查点"""
        ...
    
    def final_analysis(self):
        """进化完成后的分析"""
        ...
```

### 6.2 config.py 扩展

```
新增参数（追加到现有config.py）：

  # 进化参数
  POPULATION_SIZE = 50             # 种群大小
  GENERATION_LIFESPAN = 2000       # 每代最大寿命（步）
  NUM_GENERATIONS = 100            # 进化代数
  ELITE_RATIO = 0.1                # 精英比例
  MUTATION_RATE_INITIAL = 0.2      # 初始突变率
  MUTATION_RATE_FINAL = 0.02       # 最终突变率
  MUTATION_SCALE_INITIAL = 0.2     # 初始突变幅度
  MUTATION_SCALE_FINAL = 0.02      # 最终突变幅度
  CROSSOVER_RATE = 0.8             # 交叉概率
  TOURNAMENT_SIZE = 3              # 锦标赛大小
  EPIGENETIC_DECAY = 0.5           # 表观遗传衰减率
  
  # 进化数据路径
  EVOLUTION_DIR = "data/evolution/"
  EVOLUTION_CHECKPOINT_DIR = "data/evolution/checkpoints/"
  EVOLUTION_ANALYSIS_DIR = "data/evolution/analysis/"
```

---

# 第七部分：进化后的意识实验

### 7.1 实验流程

```
进化完成后，提取最优基因组
用最优基因组构建智能体
在该智能体上执行完整的实验1-5

流程：
  
  Step 1：进化100代
    → 获得"进化选中的架构"
    → 记录该架构的参数

  Step 2：用该架构运行成熟期（10万步）
    → 与Phase 3完全相同
    → 但架构参数来自进化而非人工设计

  Step 3：执行实验1-4
    → 与Phase 4完全相同

  Step 4：对比
    人工设计的架构（Phase 1-5的结果）
    vs
    进化选出的架构（本阶段的结果）
    
    关键问题：
    进化选出的架构是否更容易通过意识实验？
    
    如果是：
    → 自然选择偏好能产生自我意识的结构
    → 自我意识有生存价值
    → 这与"意识是基底"假说一致
    → 因为如果意识是基底的
       那么更好的透镜（更清晰的自我辨认）
       应该带来更好的生存能力
    
    如果不是：
    → 也是有价值的发现
    → 意味着自我意识可能不直接有利于短期生存
    → 或者2000步的生命太短，不足以让意识的优势显现
```

### 7.2 额外实验：进化历程分析

```
独立于实验1-4，进化本身也产出有价值的数据：

  分析A：收敛分析
    进化是否收敛到特定的架构参数？
    哪些参数收敛快（强选择压力）？
    哪些参数不收敛（中性）？
    → 揭示"生存需要什么样的大脑"

  分析B：参数量趋势
    进化倾向于更大还是更小的网络？
    参数量是否有甜蜜点？
    → 揭示"意识需要多少计算资源"

  分析C：基因组多样性
    多样性随代数如何变化？
    是否有物种分化（多个稳定的基因型共存）？
    → 揭示"是否存在多种通往意识的路径"

  分析D：表观遗传影响
    表观遗传标记是否随代数积累？
    环境压力是否改变了进化方向？
    → 揭示"经验如何影响进化"
```

---

# 第八部分：代码结构与文件清单

```
Cogito/
│
├── （现有文件不变）
│
├── evolution/                          # 新增：进化模块
│   ├── __init__.py
│   ├── genome.py                       # 基因组定义
│   ├── operators.py                    # 遗传算子（交叉、突变）
│   ├── fitness.py                      # 适应度函数
│   ├── selection.py                    # 选择策略
│   ├── epigenetics.py                  # 表观遗传
│   ├── individual.py                   # 个体类
│   └── population.py                   # 种群管理
│
├── core/
│   ├── simulation.py                   # （现有）
│   └── evolution_simulation.py         # 新增：进化仿真主循环
│
├── analysis/
│   ├── （现有分析文件）
│   └── evolution_analysis.py           # 新增：进化分析
│
├── experiments/
│   ├── （现有实验文件）
│   └── exp6_evolved_consciousness.py   # 新增：进化架构的意识实验
│
└── run_evolution.py                    # 新增：进化运行脚本
```

---

# 第九部分：施工步骤与验收标准

### 任务 E.1：基因组类

```
文件：evolution/genome.py

给AI的指令：
  "实现 evolution/genome.py。
   基因组是一个24维浮点向量，
   编码智能体的架构参数。
   [粘贴基因定义表]
   包含：随机初始化、解码为参数字典、
   参数量估算、序列化/反序列化。
   整数参数需要舍入。
   某些参数需要是8的倍数。"

验收标准：
  ☐ Genome() 随机初始化不崩溃
  ☐ genome.genes.shape == (24,)
  ☐ 所有基因值在各自的合法范围内
  ☐ decode() 返回包含所有24个参数的字典
  ☐ 整数参数确实是整数
  ☐ 需要是8的倍数的参数确实是8的倍数
  ☐ get_param_count_estimate() 返回合理的正整数
     最小基因组 → 约 10,000-20,000
     最大基因组 → 约 1,000,000-2,000,000
  ☐ 序列化 → 反序列化后基因值完全相同
  ☐ 100次随机初始化，所有基因的平均值
     大致在各自范围的中间（检查随机性）
```

完工日志 (2026-02-24):

1. 已实现 `cogito/evolution/genome.py`，包含随机初始化、解码、参数量估算、序列化/反序列化。

2. 参数量估算基于 `max(Config.SENSORY_DIM, 256)` 的感官上限，保持与 Phase 7 扩展一致。

3. 新增单测覆盖范围检查、整数与 8 倍数约束、序列化一致性与估算区间。

---

### 任务 E.2：遗传算子

```
文件：evolution/operators.py

给AI的指令：
  "实现 evolution/operators.py。
   包含交叉和突变操作。
   [粘贴交叉和突变的规格]
   交叉：均匀交叉为主，20%概率单点交叉。
   突变：高斯突变，自适应突变率。
   包含完整测试。"

验收标准：
  ☐ crossover(p1, p2) 返回两个子代Genome
  ☐ 子代的每个基因来自p1或p2（或混合）
  ☐ 子代基因在合法范围内
  ☐ 1000次交叉后，子代的每个基因
     约50%来自p1，50%来自p2（统计检验）
  ☐ mutate(genome, rate=0.1) 返回新的Genome
  ☐ 突变后约10%的基因与原来不同
     （允许±30%误差，因为是概率性的）
  ☐ 突变后所有基因仍在合法范围内
  ☐ mutate_adaptive(genome, gen=0) 突变率较高
  ☐ mutate_adaptive(genome, gen=100) 突变率较低
  ☐ mutation_rate=0 时基因不变
  ☐ mutation_rate=1 时所有基因都变
```

完工日志 (2026-02-24):

1. 已实现 `cogito/evolution/operators.py`，包含均匀交叉 + 单点交叉、Gaussian 突变与自适应突变率。

2. 新增单测覆盖交叉来源分布、突变率边界与自适应衰减。

---

### 任务 E.3：适应度和选择

```
文件：evolution/fitness.py 和 evolution/selection.py

给AI的指令：
  "实现 evolution/fitness.py 和 evolution/selection.py。
   [粘贴适应度公式和选择策略规格]
   适应度只基于生存指标。
   选择使用锦标赛选择和精英保留。
   包含测试。"

验收标准：
  ☐ compute_fitness 对全零统计返回 0
  ☐ compute_fitness 对存活1000步+吃10食物返回 > 0
  ☐ 存活更久 → 更高适应度
  ☐ 吃更多食物 → 更高适应度
  ☐ 适应度公式中不包含任何"意识"相关指标
  ☐ tournament_select 从3个中选最优
     1000次测试中最优被选概率 > 50%
  ☐ get_elites(pop, scores, 5) 返回5个最优基因组
  ☐ 精英的适应度确实是前5高的
```

完工日志 (2026-02-24):

1. 已实现 `cogito/evolution/fitness.py` 与 `cogito/evolution/selection.py`。

2. 新增单测覆盖适应度单调性、锦标赛偏好与精英选择。

---

### 任务 E.4：表观遗传

```
文件：evolution/epigenetics.py

给AI的指令：
  "实现 evolution/epigenetics.py。
   [粘贴表观遗传的完整规格]
   标记向量24维，值域[0.5, 1.5]。
   根据生命经历更新。
   遗传时衰减。
   包含测试。"

验收标准：
  ☐ EpigeneticMarks() 初始化所有标记为 1.0
  ☐ apply(genome) 返回修改后的参数字典
  ☐ 初始标记下 apply 结果与 genome.decode() 相同
  ☐ update_from_life 对低能量个体增加相关标记
  ☐ update_from_life 对高危险死亡增加相关标记
  ☐ 所有标记始终在 [0.5, 1.5] 范围
  ☐ inherit 后子代标记向1.0衰减
  ☐ 4代inherit后标记接近1.0（< 0.1偏差）
```

完工日志 (2026-02-24):

1. 已实现 `cogito/evolution/epigenetics.py`，支持标记应用、更新与继承衰减。

2. 新增单测覆盖初始化、低能量/危险更新与多代衰减。

---

### 任务 E.5：个体类

```
文件：evolution/individual.py

给AI的指令：
  "实现 evolution/individual.py。
   [粘贴个体类的完整规格]
   关键：
   1. 基因组解码为架构参数
   2. 根据参数构建CogitoAgent（需要修改CogitoAgent使其接受配置字典）
   3. 一条命：die()后大脑被销毁
   4. 统计数据完整记录
   包含测试。"

验收标准：
  ☐ Individual(Genome()) 创建成功
  ☐ 创建后 brain 不是 None
  ☐ live_one_step(world) 返回信息字典
  ☐ 连续调用 live_one_step 直到死亡
  ☐ 死亡后 brain 是 None（已销毁）
  ☐ 死亡后 is_alive 是 False
  ☐ stats 完整填充（所有字段非None）
  ☐ get_fitness() 返回非负浮点数
  ☐ 不同基因组产生不同大小的网络
     （用3个不同基因组测试，参数量不全相同）
  ☐ 代码中搜索 "exploration_rate", "fear", "risk_aversion"
     → 零结果（确认没有编码行为倾向）
```

完工日志 (2026-02-24):

1. 已实现 `cogito/evolution/individual.py`，支持基因组解码构建智能体与寿命统计。

2. 补充繁衍相关字段与自状态感官编码，死亡时销毁大脑并更新表观遗传。

3. 新增 `cogito/tests/test_evolution_population.py` 覆盖个体创建、单步死亡与参数量差异。

---

### 任务 E.6：种群管理

```
文件：evolution/population.py

给AI的指令：
  "实现 evolution/population.py。
   [粘贴种群管理的完整规格]
   包含初始化、进化、多样性计算、历史记录。
   包含测试。"

验收标准：
  ☐ Population(50) 创建成功
  ☐ initialize_random() 后有50个个体
  ☐ 50个个体的基因组互不相同
  ☐ evolve(fitness_scores) 后仍有50个个体
  ☐ 精英被保留（最优个体的基因组在下一代中存在）
  ☐ 新一代的generation号 = 上一代 + 1
  ☐ compute_diversity() 返回正数
  ☐ record_generation 后 history 中有数据
  ☐ 连续进化5代不崩溃
```

完工日志 (2026-02-24):

1. 已实现 `cogito/evolution/population.py`，包含初始化、精英保留、进化与多样性统计。

2. 增加代际历史记录结构，输出平均/最优适应度等指标。

3. 单测覆盖精英保留与多样性计算（`cogito/tests/test_evolution_population.py`）。

---

### 任务 E.7：CogitoAgent 适配

```
文件：修改 agent/Cogito_alpha.py

修改要求：
  当前的CogitoAgent使用固定的config参数
  需要修改为能接受配置字典
  使得不同的基因组能构建不同大小的网络
  
  修改方式：
    __init__(self, config) 中
    config 可以是原来的Config类
    也可以是一个字典（来自基因组的decode）
    
    所有硬编码的维度改为从config读取

给AI的指令：
  "修改 agent/Cogito_alpha.py，
   使 CogitoAgent 能接受配置字典，
   而非只能接受固定的Config类。
   所有架构参数从配置中读取。
   保持与原有代码的向后兼容性。
   不改变任何功能行为。
   包含测试：
   - 用原Config创建 → 与之前行为一致
   - 用基因组解码的字典创建 → 能正常运行"

验收标准：
  ☐ 用原来的Config类创建 → 所有现有测试通过
  ☐ 用基因组字典创建 → 不崩溃
  ☐ 用最小基因组创建 → 参数量小
  ☐ 用最大基因组创建 → 参数量大
  ☐ 两种创建方式都能正常act()和observe_result()
  ☐ 现有Phase 0-5的代码不受影响
```

进度记录（2026-02-24）：
  - 更新基础世界动作表，加入动作 6（interact，no-op）。
  - 修正 LSTM 输入维度注释（71 维）。
  - 新增 `test_interact_no_op` 覆盖动作 6 行为。

完工日志 (2026-02-24):

1. 新增 `cogito/agent/agent_config.py`，统一解析 Config/字典配置。

2. `cogito/agent/cogito_agent.py` 与子模块改为读取动态架构参数。

3. `cogito/tests/test_agent.py` 增加字典配置创建测试，保持兼容。

---

### 任务 E.8：进化仿真主循环

```
文件：core/evolution_simulation.py

给AI的指令：
  "实现 core/evolution_simulation.py。
   [粘贴进化仿真主循环的完整规格]
   包含：
   - 种群初始化
   - 代际循环
   - 个体生命周期
   - 适应度计算
   - 进化操作
   - 统计打印
   - 检查点保存
   包含测试：运行3代，每代10个个体，
   每个寿命100步。验证不崩溃。"

验收标准：
  ☐ EvolutionSimulation(config) 创建成功
  ☐ 运行3代×10个体×100步寿命 不崩溃
  ☐ 每代打印统计信息
  ☐ 适应度在代际间有变化趋势
  ☐ 精英的基因组在下一代中存在
  ☐ 检查点保存功能正常
  ☐ 总运行时间合理（3代×10×100 < 5分钟 CPU）
```

完工日志 (2026-02-24):

1. 新增 `cogito/core/evolution_simulation.py`，支持代际与连续两种进化模式。

2. 代际模式使用 `Population` + `CogitoWorld`，连续模式使用 `EvolutionWorld` 生态。

3. 支持代际与连续检查点保存与统计输出。

---

### 任务 E.9：进化运行与分析

```
文件：run_evolution.py 和 analysis/evolution_analysis.py

运行要求：
  完整运行：50个体 × 100代 × 2000步寿命
  总计：50 × 100 × 2000 = 10,000,000 步
  预计时间（CPU）：数小时到一天
  
  可以先用小规模验证：20个体 × 20代 × 500步

分析要求：
  1. 适应度曲线（代际平均/最优/最差）
  2. 基因组参数收敛图（每个参数的代际变化）
  3. 参数量趋势图
  4. 多样性曲线
  5. 最终最优基因组的详细参数
  6. 表观遗传标记的代际变化

给AI的指令：
  "创建 run_evolution.py 和 analysis/evolution_analysis.py。
   运行脚本执行完整进化。
   分析脚本读取进化数据，生成6类分析图。
   [粘贴分析要求]
   支持小规模测试模式。"

验收标准：
  ☐ 小规模测试（20×20×500）完成不崩溃
  ☐ 适应度有上升趋势
  ☐ 至少部分参数有收敛趋势
  ☐ 6类分析图全部生成
  ☐ 最优基因组参数记录完整
  ☐ 完整运行（50×100×2000）完成
     （如果时间太长，可接受 50×50×1000）
```

完工日志 (2026-02-24):

1. 新增 `run_evolution.py` 与 `cogito/analysis/evolution_analysis.py`。

2. 支持小规模验证参数，输出适应度曲线、参数趋势、多样性与最优基因组。

---

### 任务 E.10：进化架构的意识实验

```
文件：experiments/exp6_evolved_consciousness.py

流程：
  1. 从进化中提取最优基因组
  2. 用该基因组构建CogitoAgent
  3. 运行成熟期（10万步）
  4. 依次执行实验1-4
  5. 与Phase 4-5的结果对比

给AI的指令：
  "创建 experiments/exp6_evolved_consciousness.py。
   从进化检查点加载最优基因组。
   用该基因组构建智能体。
   运行成熟期和全部实验1-4。
   与人工设计架构的结果对比。
   生成对比报告。"

验收标准：
  ☐ 成功加载最优基因组
  ☐ 成功构建智能体
  ☐ 成熟期运行完成
  ☐ 实验1-4全部完成
  ☐ 所有指标都有数值
  ☐ 与Phase 4-5的对比报告生成
  ☐ 对比报告包含：
     - 每个实验的每个指标的 进化架构 vs 人工架构
     - 统计检验结果
     - 最终判定
```

完工日志 (2026-02-24):

1. 新增 `cogito/experiments/exp6_evolved_consciousness.py`，从进化检查点提取最优基因组。

2. 使用进化架构完成成熟期与实验 1-4，并生成对比报告 JSON。

---

# 第十部分：时间线

```
本模块在 Phase 5 之后执行

  Week 1 (任务 E.1-E.4)：
    ☐ 基因组类
    ☐ 遗传算子
    ☐ 适应度和选择
    ☐ 表观遗传
    所有单元测试通过

  Week 2 (任务 E.5-E.7)：
    ☐ 个体类
    ☐ 种群管理
    ☐ CogitoAgent适配
    集成测试通过

  Week 3 (任务 E.8-E.9)：
    ☐ 进化仿真主循环
    ☐ 小规模验证运行
    ☐ 完整进化运行（可能需要隔夜）
    ☐ 进化分析

  Week 4 (任务 E.10)：
    ☐ 进化架构的意识实验
    ☐ 对比分析
    ☐ 最终报告

  git commit -m "Phase 6 complete: evolutionary architecture search"
```

---

# 第十一部分：预期发现与价值

```
最有可能的发现：

  1. 进化收敛到特定的核心维度和层数
     → 揭示"最小可行大脑"的参数

  2. 学习率和缓冲区大小收敛到特定范围
     → 揭示最优学习配置

  3. 进化可能偏好预测损失权重 > 生存损失权重
     → 世界模型可能比直接奖励更重要
     → 这与自由能原理一致

  4. 参数量可能不是越大越好
     → 存在甜蜜点
     → 太大学不好，太小容量不够

最令人兴奋的可能：

  5. 进化选出的架构在意识实验中
     比人工设计的架构表现更好
     → 自然选择独立发现了"意识结构"
     → 无需人类设计，进化自己找到了
        更好的"透镜形状"
     → 这是"意识有生存价值"的证据
     → 这是"意识是基底"的间接证据

无论结果如何的价值：

  6. 即使进化架构没有通过意识实验
     进化过程本身的数据也是有价值的
     它告诉我们"在生存压力下
     什么样的计算结构被偏好"
     这对理解生物大脑的进化也有参考价值
```

---

```
施工总量：
  新增文件：~10个Python文件
  修改文件：2个（config.py, Cogito_alpha.py）
  新增参数：~15个配置参数
  总代码量：~1500-2000行
  
  与现有Phase 0-5完全兼容
  不影响现有实验
  是一个独立的"附加层"
```
