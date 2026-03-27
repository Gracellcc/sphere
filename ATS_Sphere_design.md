Adaptive Training Skills (ATS) + Sphere

motivation: 让模型自己学会在训练过程中动态调整训练策略。
agent环境: AppWorld、WebArena…
Reasoner: multi-turn trajectory。
训练策略的优化体现在两层：模型学会生成和演化behavioral skills（教agent什么行为），同时学会调整training skill（怎么训练agent）。
Sphere负责per-step的skill检索和注入（inference-time，纯几何，免费），ATS负责skill质量和演化（training-time，GRPO）。

J(θ) = J_inner(θ) + λ_outer · J_outer(θ)

两种skill：
* behavioral skill：agent应该怎么做任务。数量多（十几个），per-step Sphere检索。同时是prompt hint + supervision标准。
* training skill：怎么训练agent。library里存多个，全局1个active。负责data selection + reward weighting。按照训练状态切换。

prompt设计的模型角色：
1. 固定角色：reasoner，跟环境交互完成任务，输出multi-turn trajectory τ = (o₁,a₁,o₂,a₂,...,oₖ,aₖ)
2. 每M步工作一次的角色：skill evolver，看诊断报告（含sphere空间信息），对两种skill做修改/新增/淘汰

—————————————————————————————————————————————————————————————————————————————————————————

skill的格式（ATS + Sphere）：

behavioral skill：
```
# [行为模式名称]
## Type: behavioral
## Guidance: [给agent的行为指导，Sphere per-step注入到prompt]
## Scoring: [自然语言评分标准，喂给verifier LLM打分]
## When to Use: [适用条件，Sphere条件过滤用]
## Embedding: [frozen encoder生成，Sphere几何检索用]
## Category: [环境/任务类型标签，Sphere category过滤用]
```
一个behavioral skill同时承担：prompt hint（Guidance）、supervision标准（Scoring）、条件触发（When to Use）、几何位置（Embedding）。

training skill：
```
# [策略名称]
## Type: training
## Data Selection: [选什么任务，怎么选。半结构化格式，程序可parse]
## Reward Formula: [outcome和supervision怎么组合。数学表达式，直接parse系数]
## When to Use: [什么训练状态下用这个策略]
```

—————————————————————————————————————————————————————————————————————————————————————————

skill library + Sphere：
* behavioral skills：多个文件，全部编码到同一个Sphere上
    * 检索方式（per-step，Sphere负责，免费）：
        1. 条件过滤：当前训练状态数值 匹配 skill的When to Use
        2. Category过滤：当前任务涉及的app → 去掉不相关category的skill
        3. 几何检索：intent tracking（slerp更新intent point）→ complementarity selection选3-5个
        4. 注入决策：SGC confidence判断注入多少（SGC高→少注入，SGC低→强注入，loop→强制注入+rotation）
    * Guidance注入：per-step动态，每步可能注入不同的skills
    * Scoring评价：用固定的active behavioral skill set（跟外层演化同步，不随step变化）
    * 淘汰：score长期饱和（std→0）的被archive
    * 保留版本历史（v1→v2→v3）
    * Skill被修改/新增后重新encode embedding（几秒），sphere自动更新
* training skills：library里存多个，全局只有1个active
    * 训练框架读取active training skill的Data Selection和Reward Formula来配置当前训练
    * skill evolver每M步可以修改当前active的、切换到另一个、或新增一个
    * 保留版本历史

Sphere不需要训练：
* Encoder: frozen（如Qwen3-Embedding-0.6B）
* Skill embeddings: skill文本变了就re-encode
* 几何算法: 纯数学（complementarity, slerp, coverage, isolation）
* ATS evolution改了skill → 重新embed → sphere自适应，无gradient

—————————————————————————————————————————————————————————————————————————————————————————

behavioral skill示例（初始由API根据trajectory分析生成，后续由模型修改/新增）：

# Documentation Before Action
## Type: behavioral
## Guidance
在调用任何不熟悉的API之前，先通过ApiDocs查询其完整文档，
确认所需参数、返回格式和可能的错误码。不要猜测参数。
## Scoring
检查trajectory中每次API调用：之前是否查过该API的文档。
score = 查过文档的API调用数 / 总API调用数
## When to Use
API调用失败率 > 40%
## Category
appworld/general

# Error Diagnosis and Recovery
## Type: behavioral
## Guidance
每次API调用后检查返回状态。如果失败：
1. 读错误信息，判断原因（未登录/参数错/权限不足）
2. 针对原因采取不同措施，不要重复同样的调用
3. 如果连续两次同一API失败，换一种方法
## Scoring
检查trajectory中每次API失败后的行为：
- 读了错误信息并改变了下一步 → 1分
- 直接重试同样的调用 → 0分
score = 正确处理次数 / 总失败次数（无失败则score=1）
## When to Use
"连续重复同一API调用"的比例 > 20%
## Category
general（跨环境通用）

# Cross-App Data Flow
## Type: behavioral
## Guidance
把一个app的数据传给另一个app时：
1. 从返回值中精确提取需要的字段
2. 确认字段类型和格式符合下游app要求
3. 不要传整个JSON对象，只传需要的值
## Scoring
检查trajectory中所有跨app数据传递：
- 正确提取并传递所需字段 → 1分
- 传了整个返回对象 → 0.3分
- 格式错误导致下游失败 → 0分
score = 各次传递得分的平均值
## When to Use
涉及≥2个app的任务
## Category
appworld/multi_app

# Login Management
## Type: behavioral
## Guidance
访问某个app前，先调用apis.supervisor.show_account_passwords()获取所有app的密码，
然后用对应密码调用apis.APP_NAME.login(username=email, password=pw)获取access_token。
phone app用phone_number作为username，其他app用email。
不要重复登录同一个app，把token存在变量里复用。
## Scoring
- 每个用到的app是否在使用前登录了（各0或1）
- 有无重复登录（每次重复扣0.1）
score = 正确登录数/需登录数 - 重复扣分
## When to Use
"未登录"相关API错误占总错误 > 30%
## Category
appworld/general

# Task Completion Awareness
## Type: behavioral
## Guidance
执行过程中持续追踪任务的所有要求是否已满足。
完成后立即调用complete_task，不要做额外操作（避免collateral damage）。
## Scoring
- 所有要求满足且调了complete_task → 1
- 完成了但没调complete_task → 0.5
- 做了不必要的额外操作 → 扣0.2/次
- 没完成就停了 → 0
## When to Use
始终启用
## Category
general

# Efficient Planning
## Type: behavioral
## Guidance
开始前先制定简要计划：需要哪些app、哪些API、什么顺序、数据依赖关系。
然后按计划执行，避免无目的探索。
## Scoring
score = min(1.0, shortest_success_steps / actual_steps)
（无成功参考时：成功=1，失败=0）
## When to Use
平均trajectory长度 > 成功trajectory平均长度的2倍
## Category
general

—————————————————————————————————————————————————————————————————————————————————————————

training skill示例（初始由API生成，后续由模型修改）：

# Early Stage Balanced
## Type: training
## Data Selection
选择条件：均匀采样所有任务类型，排除成功率>80%的类型
## Reward Formula
加权公式：r = 1.0 × outcome + 0.3 × supervision
（早期supervision信号不够成熟，低权重）
## When to Use
整体成功率 < 20%

# Weakness Focused
## Type: training
## Data Selection
选择条件：70%选成功率最低的两个任务类型，30%其他
## Reward Formula
加权公式：r = 0.7 × outcome + 1.0 × supervision
（supervision已成熟，加大权重）
## When to Use
整体成功率 > 20% 且 最弱任务类型成功率 < 10%

# Efficiency Push
## Type: training
## Data Selection
选择条件：选成功率>30%但平均步数>20的任务（能做对但做得慢）
## Reward Formula
加权公式：r = 0.5 × outcome + 1.0 × supervision + 0.3 × (1 - steps/max_steps)
（额外奖励效率）
## When to Use
整体成功率 > 40%

—————————————————————————————————————————————————————————————————————————————————————————

skill修改方式（外层）：
* 两种skill的更新频率不同：
    * behavioral skills：每M步可改（需要频繁更新，某个skill可能很快饱和）
    * training skill：每K×M步可改（全局配置不宜频繁切换，K>1，如K=3或5）
* 实现方式：每M步都生成G个candidate完整方案（training + behavioral），但在非training更新周期时，training部分被锁定
* 每个candidate方案是一个完整的skill配置：
    * 1个training skill（锁定周期内不变；更新周期内可切换/修改/新增）
    * 1组behavioral skills（可包含修改版、新增的、淘汰某些）

proxy reward（统一）：
* 每个candidate用自己的完整配置（training skill + behavioral skills）跑一小批任务
* proxy reward = 这一小批任务的平均outcome reward

—————————————————————————————————————————————————————————————————————————————————————————

verifier（LLM-as-Judge + SGC gate）：
* 作用：把behavioral skill的Scoring标准变成supervision reward数值，给内层GRPO提供比纯outcome更丰富的学习信号。
* 实现：外部frozen LLM。输入behavioral skill的Scoring描述 + trajectory，输出0-1的分数。
* Scoring评价用固定的active behavioral skill set（跟外层演化同步），不随per-step注入变化。即：Sphere管"每步注入哪些Guidance"（动态），verifier管"按哪些Scoring评价"（固定）。两者解耦。
* SGC gate省成本：
    * trajectory平均SGC > τ 且 outcome成功 → 跳过verifier，supervision reward用SGC归一化值近似（agent确实做对了且sphere也认为稳定）
    * 其他情况（SGC低 或 outcome失败）→ 调verifier精确打分（需要精确诊断哪里出了问题）
    * 注意：SGC衡量的是"agent在skill空间中的位置是否稳定"，不等于"agent做得好"。高SGC+失败的case对evolution最有价值（说明skill本身有问题），必须调verifier
    * 预计省约40-60%的verifier API调用（取决于成功率）
* 每个active behavioral skill各自独立打分（0-1），取平均作为总supervision reward。

—————————————————————————————————————————————————————————————————————————————————————————

诊断报告（外层每M步收集，含sphere空间信息）：
* 训练统计（原有）：
    * 各任务类型/difficulty level的成功率
    * 各behavioral skill的score分布和趋势
    * 失败trajectory的常见pattern
    * 当前training skill的配置效果
* Sphere空间信息（新增）：
    * Coverage gap：sphere上哪些区域没有skill覆盖（Voronoi稀疏区域），说明需要新增skill
    * Drift pattern：agent在episode中的intent trajectory，频繁drift的方向对应当前痛点
    * 冗余检测：两个skills的embedding cosine > 0.9，建议合并或淘汰一个
    * Skill-intent mismatch：某个skill在sphere上的位置 vs agent实际drift的方向不匹配，说明skill内容需要修改

—————————————————————————————————————————————————————————————————————————————————————————

训练流程：
0. warmup：
    1. base model经过cold-start SFT（教Reasoner格式+两种skill markdown格式+skill修改格式）
    2. 用base model在训练集子集上跑一批trajectory，收集数据
    3. 用强模型API分析trajectory（成功/失败pattern、常见错误类型）
    4. API生成初始skill library：~6个behavioral skills + ~3个training skills（Early Stage Balanced设为初始active）
       - 关键：初始skill的Guidance必须具体到API调用level（包含具体的函数名、参数pattern、代码示例），不能是抽象原则。
         实验教训：抽象的"从系统源获取凭证"反而误导agent去找password reset；具体的"调apis.supervisor.show_account_passwords()"直接解决问题。
    5. Frozen encoder对所有behavioral skills生成embedding，初始化sphere

1. 阶段1：API主导（前P步，P为超参数）
    * 内层循环 - GRPO（每步）：
        1. 读active training skill的Data Selection → 从任务库选一批任务
        2. 对每个任务的每一步交互：
            a. Sphere条件过滤（When to Use）+ category过滤 + 几何检索（intent tracking + complementarity）
            b. SGC判断注入强度 → 选出的skills的Guidance注入到当前step的prompt
        3. Reasoner对任务与环境multi-turn交互，生成N条trajectory（同一任务N条，GRPO）
        4. SGC gate：
            - trajectory平均SGC > τ 且 outcome成功 → supervision reward ≈ SGC归一化值（跳过verifier）
            - 其他情况 → active behavioral skills的Scoring + trajectory喂给verifier LLM → 各skill打分 → 取平均 = supervision reward
        5. 读active training skill的Reward Formula → 组合outcome和supervision → 总reward
        6. GRPO更新θ（只有J_inner）
    * 外层循环（每M步，behavioral可改；每K×M步，training也可改）：
        1. 收集诊断报告（训练统计 + sphere空间信息）
        2. API看诊断报告+当前skill library，生成G个candidate完整方案
            - 若非training更新周期：所有candidate的training skill = 当前active，只改behavioral部分
            - 若是training更新周期：training部分也可改
            - Sphere信息辅助决策：coverage gap → 建议新增skill；冗余 → 建议合并；mismatch → 建议修改
        3. 每个candidate试用一小批任务 → 算proxy reward（平均outcome reward）
        4. 选最优candidate → 更新active配置
        5. 新增/修改的behavioral skills重新encode embedding → sphere自动更新
        6. 记录（诊断报告, candidate方案, proxy reward）数据
    * 注意：这一阶段外层不更新θ（candidate是API生成的）

2. 过渡SFT：用阶段1积累的（诊断报告, skill修改, 效果）数据SFT模型的skill修改能力

3. 阶段2：模型主导（P步起）
    * 内层循环 - GRPO（每步）：跟阶段1完全一样
    * 外层循环 - GRPO（每M步，behavioral可改；每K×M步，training也可改）：
        1. 收集诊断报告（训练统计 + sphere空间信息）
        2. 模型自己看诊断报告+当前skill library，生成G个candidate完整方案
            - 若非training更新周期：所有candidate的training skill = 当前active，只改behavioral部分
            - 若是training更新周期：training部分也可改
        3. 每个candidate试用一小批任务 → 算proxy reward（平均outcome reward）
        4. GRPO更新θ的skill修改能力
        5. 最优candidate → 更新active配置，新增/修改的skills重新encode → sphere更新
    * J(θ) = J_inner(θ) + λ_outer · J_outer(θ) 正式生效
    * API历史skills保留在library中，模型需要真正超过API版本才能替代

—————————————————————————————————————————————————————————————————————————————————————————

跨环境Unified Sphere：
* 不同环境训练各自产出evolved skills → 放到同一个sphere上
* 通用skills（Error Recovery, Task Decomposition等）位于sphere通用区域，跨环境共享
* 某环境inference时，category filtering去掉不相关环境的env-specific skills
* 通用skills不被过滤，自然参与所有环境的检索

—————————————————————————————————————————————————————————————————————————————————————————

ATS vs Sphere职责分工：
* ATS负责skill是什么（质量、内容、演化）— training-time
* Sphere负责skill怎么用（检索、注入、自适应）— per-step, 免费
* Verifier负责skill的Scoring变成reward — episode结束后, SGC gate省成本

—————————————————————————————————————————————————————————————————————————————————————————


—————————————————————————————————————————————————————————————————————————————————————————

跟RLCER的关系：RLCER是ATS的inspiration。RLCER的rubric相当于一种behavioral skill（Scoring部分=rubric），RLCER的verifier π_ϕ相当于我们的verifier LLM，RLCER的固定1:1加权相当于一个不演化的training skill。ATS+Sphere把这些都泛化了：behavioral skill不只是rubric还包含行为指导，检索从静态变成per-step几何自适应，training skill包含data selection，两种skill都会随训练演化。
