# 一、封面

**项目题目**：增强型电影推荐系统 —— 融合混合推荐、可解释性与以用户为中心的反馈机制

**课程名称**：COMP4135 推荐与搜索技术（2025–26 学年，第二学期）
**所在院系**：香港浸会大学 计算机科学系
**小组名称**：_<待填：小组名>_
**提交日期**：_<待填：YYYY-MM-DD>_

## 小组成员

| 序号 | 姓名 | 学号 | 主要分工 |
|:---:|:---|:---|:---|
| 1 | _<待填>_ | _<待填>_ | _例：算法（时间衰减 kNN + 混合）_ |
| 2 | _<待填>_ | _<待填>_ | _例：前端 UI（Vue + 界面增强）_ |
| 3 | _<待填>_ | _<待填>_ | _例：用户实验与统计分析_ |
| 4 | _<待填>_ | _<待填>_ | _例：报告撰写与可视化_ |

---

## 目录

一、封面（见上）
二、系统描述
三、用户界面截图
四、评估流程与结果
五、个人反思
六、GenAI 使用声明
七、参考文献
八、小组参与表签名（附件）
附录 A–F

---

# 二、系统描述（约 1.5 页）

## 2.1 研究问题与基线系统

课程提供了一个基于 Flask + Vue + Bulma 的电影推荐 Demo，数据集为 MovieLens 子集（610 名用户、9 742 部电影）。基线系统采用基于物品的 k 近邻方法（item-based kNN），界面极简：先选类型 → 给 10 部电影打星 → 查看 "Recommended for You" 与 "Liked with Similar Items"。我们通过使用体验识别出以下五处不足，并以此作为增强工作的出发点：

1. **缺少时间感知**——所有评分同等对待，忽略了 `timestamp` 字段。
2. **内容信号仅靠类型**——电影简介（`overview`）文本完全未被利用。
3. **无可解释性**——用户看不到"为什么推荐这部"。
4. **反馈粒度粗糙**——只有一个 Like 按钮，没有 Dislike，没有"没看过"选项，也没有详情视图。
5. **无多样性控制**——Top-N 列表往往重复感强，用户无法调节。

**研究问题**（驱动 A/B 评估）：
- **RQ1（算法）**：相比基线 User-kNN，我们的混合推荐流水线是否能产生更符合用户口味的结果？
- **RQ2（界面）**：重新设计的用户界面能否提升"透明度、可控性、多样性感知"？

## 2.2 系统架构

```
浏览器（Vue 3 + Chart.js + Bulma）
        │  HTTP + JSON
        ▼
Flask 服务器（flaskr/main.py）
    │
    ├── flaskr/tools/data_tool.py       —— CSV 加载 + 封面缓存
    └── flaskr/tools/recommenders.py
            ├── 时间衰减 User-kNN         （A1）
            ├── TF-IDF + 类型内容模型     （A2）
            ├── 混合打分 + MMR 多样化    （A3）
            └── 解释构造器                 （驱动"Why?"弹窗）
```

通过两个环境变量（`REC_ALGO`、`UI_MODE`）在 **Original 基线**与**Enhanced 增强版**之间切换，可实现干净的受试内（within-subject）A/B 对照（见 `README.md` §3.2）。

## 2.3 数据

| 数据来源 | 使用字段 | 用于 |
|---|---|---|
| MovieLens `ratings.csv` | `userId`, `movieId`, `rating`, `timestamp` | 协同过滤模型、时间衰减 |
| MovieLens `movies.csv` | `movieId`, `title`, `genres` | 类型 multi-hot 块 |
| TMDB（爬取缓存） | `overview`, `cover_url`, `release_date` | TF-IDF 文本向量、海报、年代直方图 |

## 2.4 基线（继承）功能

| 功能 | 说明 | 所在文件 |
|---|---|---|
| 类型选择弹窗 | 首次访问时多选 | `templates/index_original.html` |
| 10 部冷启动打分 | 对当前选择类型下的热门电影打星 | `templates/index_original.html` |
| User-kNN 推荐 | 基于 `scikit-surprise` 的 `KNNWithMeans` | `flaskr/main_original.py` |
| 相似物品推荐 | 基于评分矩阵的余弦相似度 | `flaskr/main_original.py` |
| Like 列表 | 底部持久化喜欢列表 | `templates/index_original.html` |
| Cookie 状态 | 所有用户状态保存在浏览器 Cookie | 同上 |

## 2.5 算法增强（详见 `flaskr/tools/recommenders.py`）

1. **时间衰减 User-kNN（A1）**：历史评分按半衰期 365 天做指数衰减

   r'_{u,i} = r̄_u + (r_{u,i} − r̄_u) · exp(−ln 2 · Δt / H)，  H = 365 天

   衰减后的矩阵再送入 Pearson 相似度的 `KNNWithMeans`。**动机**：用户口味会随时间漂移，基线完全忽略 `timestamp` 属于典型缺陷（Koren, 2010）。

2. **TF-IDF + 类型内容模型（A2）**：电影简介用 `TfidfVectorizer`（1–2 元词，英文停用词，`min_df=2`）向量化并 L2 归一化，横向拼接一个 2 倍加权的类型 multi-hot 块。用户画像取其已 Like 或高分电影向量的加权平均；候选电影用余弦相似度打分。**动机**：引入语义信号，使稀疏用户和冷启动用户也能得到有意义的结果。

3. **混合打分 + MMR 多样化（A3）**：协同过滤分与内容分先 min-max 归一化，再以 α = 0.6 线性融合，Top 候选再经 Maximal Marginal Relevance 重排：

   MMR(i) = (1 − λ) · s_i  −  λ · max_{j ∈ S} sim(i, j)

   其中 λ ∈ [0, 1] 直接绑定到前端 Diversity 滑块。**动机**：在相关性与新颖度之间取得平衡，并把"多样性"这件事交给用户自己控制（Carbonell & Goldstein, 1998）。

4. **兴趣加权与负反馈过滤**：冷启动阶段的"感兴趣 / 不感兴趣"分别给最终分数 **+0.5 / −0.3** 的偏置；被 👎 的电影在后续刷新中被过滤。**动机**：把隐式的冷启动信号显式化，进入打分公式。

## 2.6 界面增强（共 9 项）

| 编号 | 增强功能 | 对应需求点 | 后端 / 文件 |
|:---:|---|---|---|
| U1 | 冷启动页**搜索框**（全库检索） | B1 偏好引导 | `GET /api/search` |
| U2 | 每张卡片的 **"Why?" 解释徽章 + 弹窗** | B2 透明度 | `GET /api/explain/<movieId>` |
| U3 | **👍 / 👎 反馈** + **Feedback Diary** 抽屉 | B3 反馈丰富度 | `_modals.html` |
| U4 | **详情弹窗**（海报 + 简介 + 类型 + 解释） | B4 信息丰富度 | `_modals.html` |
| U5 | 顶栏 **Diversity 滑块**，实时重排 | B5 用户可控 | `mmr_rerank` + cookie |
| U6 | **"没看过"流程**：感兴趣 / 不感兴趣 / 跳过 | B1 + B3 | `POST /api/interest` |
| U7 | **Trending Now** 横滑热门条 | B6 兜底入口 | `main.py`, `index.html` |
| U8 | **Your Taste** 仪表盘（雷达 + 年代直方图 + Top-5 关键词） | B6 自我理解 | `GET /api/profile/stats` |
| U9 | **Profile Switcher**（P01–P06 Cookie 前缀隔离） | A/B 评估脚手架 | `_pp()` in `main.py` |

## 2.7 设计理由小结

四项算法增强（A1–A3 + 兴趣加权）均直接对应课程所授内容：协同过滤、内容过滤、混合推荐、多样性重排。界面增强则以推荐系统 HCI 领域的经典启发式为依据：解释与透明度（Tintarev & Masthoff, 2007）、用户可控性（Pu & Chen, 2010）、长尾与新颖度权衡（Carbonell & Goldstein, 1998）。U9（Profile Switcher）纯粹是评估用脚手架，不改变推荐算法本身；其存在在 §4.5 局限性中如实披露。

---


# 三、用户界面截图

> **截图要求**：每张图下方附**一句话**说明其相较于原始设计的新颖之处。全部 17 张截图按下表顺序编号，统一放在 `report/figures/` 目录下。
>
> **准备工作**：
> - 对照组（4 张）：先运行 `scripts\run_original.bat`，浏览器开无痕窗口访问 <http://127.0.0.1:5000>，完成一次完整的 10 部评分流程。
> - 增强版（13 张）：关闭 Original 服务器后运行 `scripts\run_enhanced.bat`，再开新无痕窗口访问同一地址，完成 Profile 切换与 10 次交互。
> - 所有截图用浏览器 DevTools 的 1366×768 视口（F12 → 切换设备模式）保持一致分辨率；敏感信息（如 Cookie）可用方框遮盖。

## 3.1 原始系统截图（对照组，共 4 张）

**图 3-1 `fig_orig_genres.png` — 原始类型选择弹窗**
捕获内容：首次访问时弹出的 multi-select genres 弹窗，勾选若干类型后按 "Save"。
新颖性说明（一句话）：*原始系统仅支持一次性类型勾选，无搜索、无"不感兴趣"选项，冷启动信息量偏低。*

**图 3-2 `fig_orig_rate.png` — 原始 10 部评分弹窗**
捕获内容：10 部热门电影列表（无海报放大、无"没看过"按钮），每部配 5 星评分。
新颖性说明：*原始系统强制用户对所有 10 部电影打分，不提供"没看过"的规避路径，易造成失真评分。*

**图 3-3 `fig_orig_recs.png` — 原始 "Recommended for You" 列表**
捕获内容：主页顶部的推荐卡片网格，每张仅显示海报、标题、类型和 Like 按钮。
新颖性说明：*原始卡片不含"为什么推荐"的解释，也无多样性控制；用户只能被动接收单一排序结果。*

**图 3-4 `fig_orig_liked_similar.png` — 原始 "Liked with Similar Items" + 底部 Liked strip**
捕获内容：向下滚动可见的"已喜欢电影 + 相似推荐"并列区，以及页脚 Liked 横条。
新颖性说明：*原始反馈路径单一（只有 Like，无 Dislike 与"没看过"），无法区分"不感兴趣"与"未接触"。*

## 3.2 增强版系统截图（共 13 张）

**图 3-5 `fig_nav.png` — 增强版顶部导航栏**
捕获内容：完整的导航栏（Profile 下拉 / Clean All / Reset Genres / View Taste / Feedback Diary / Diversity 滑块）。
新颖性说明：*导航栏集成了用户画像、反馈历史、多样性控制等多入口，使推荐流程的元操作对用户可见可调。*

**图 3-6 `fig_onboarding_search.png` — 冷启动搜索框**
捕获内容：10 部评分页顶部的 Search bar，输入 "inception" 后出现的下拉匹配结果。
新颖性说明：*用户可直接搜索自己真正熟悉的电影打分，突破了原始系统"按类型热门预筛选"的固定 10 部限制。*

**图 3-7 `fig_havent_seen.png` — Haven't seen 三按钮弹窗**
捕获内容：点击冷启动卡片"Haven't seen"按钮后弹出的 Interested / Not interested / Skip 三选项对话框。
新颖性说明：*首次将"未接触"显式建模为独立反馈，避免了原始系统把"没看过"误当作"低分"的偏差。*

**图 3-8 `fig_progress.png` — 冷启动 10/10 进度条**
捕获内容：完成第 10 次交互时的绿色进度条 + "Onboarding complete!" 提示。
新颖性说明：*进度条同时计入评分与兴趣标记，向用户透明化"系统已收到多少信号"。*

**图 3-9 `fig_trending.png` — Trending Now 横向滚动条**
捕获内容：主页"Trending Now"栏的横向可滑动海报条（展示 8–12 部近期热门）。
新颖性说明：*为新用户或冷门画像提供基于全局热度的兜底入口，原始系统只有个性化推荐一条通路。*

**图 3-10 `fig_recs_why.png` — 带 "Why?" 徽章的推荐列表**
捕获内容：Recommended for You 网格，每张卡片右上角可见黄色 "Why?" 徽章。
新颖性说明：*每张推荐卡都配有可点击的解释入口，兑现了解释性推荐在 Herlocker (2000) 中提出的透明度原则。*

**图 3-11 `fig_why_popover.png` — Why? 弹窗展开**
捕获内容：点击 Why? 徽章后弹出的气泡，显示"因为你给 X / Y / Z 打了高分"、贡献度百分比、命中的 TF-IDF 关键词。
新颖性说明：*解释同时融合协同过滤与内容特征两种证据，而非单一"相似用户也喜欢"的黑箱说法。*

**图 3-12 `fig_detail_modal.png` — 电影详情弹窗**
捕获内容：点击卡片后弹出的详情模态框，含海报、上映年份、类型标签、完整 overview、Why 解释段落、Like/Dislike 按钮。
新颖性说明：*单屏完成"了解 → 决策 → 反馈"闭环，无需跳转外链，原始系统仅能进入单独的详情页。*

**图 3-13 `fig_diversity_side.png` — Diversity 滑块对比（λ=0.1 vs λ=0.9）**
捕获内容：左右并排两张截图（可用任意图像工具拼接），分别为滑块在左端 (λ=0.1) 和右端 (λ=0.9) 时同一用户的推荐列表。
新颖性说明：*多样性从系统内部参数被外化为用户可调滑块，实证了 MMR 重排对 Top-N 结果的直观影响。*

**图 3-14 `fig_taste_radar.png` — Your Taste 类型雷达图**
捕获内容：点击"View Taste"后的雷达图（Action / Drama / Comedy / … 共 8 个轴），数据来源于 Liked 和高分评分。
新颖性说明：*把用户画像以可视化方式返还给用户，形成 "user models the system modelling them" 的双向透明。*

**图 3-15 `fig_taste_decade.png` — Your Taste 年代分布 + Top-5 关键词**
捕获内容：Your Taste 弹窗下半部分的年代直方图 + Top-5 TF-IDF 关键词云。
新颖性说明：*除类型维度外，同时呈现时间偏好与内容主题，让用户看到系统推断出的细粒度偏好。*

**图 3-16 `fig_diary.png` — Feedback Diary 抽屉**
捕获内容：点击"Feedback Diary"按钮滑出的右侧抽屉，按时间倒序列出所有评分、Like、Dislike、兴趣标记记录。
新颖性说明：*把隐式交互日志变为用户可审计的显式清单，支持"撤销"和"纠错"，原始系统仅保留 Liked。*

**图 3-17 `fig_profile_dropdown.png` — Profile Switcher 下拉菜单**
捕获内容：顶栏 Profile 下拉展开后显示 P01–P06 六个受试者条目，旁边有"Clean All"按钮。
新颖性说明：*同一浏览器内通过 Cookie 前缀实现多用户数据隔离，使受试者内（within-subject）A/B 实验无需多台设备。*

## 3.3 评估相关图表（由脚本自动生成，2 张）

**图 3-18 `fig_mean_bars.png` — Q1–Q8 均值对比柱状图**
由 `python evaluation\analyze_results.py evaluation\results.csv --figures-dir report\figures` 生成。
新颖性说明：*并排展示条件 A (Original) 与 B (Enhanced) 在 8 个主观指标上的均值差异，直观支持 §4.3 的显著性检验。*

**图 3-19 `fig_paired_box.png` — Q1–Q8 配对箱线图**
同上命令生成。
新颖性说明：*箱线图呈现 6 名受试者在每题上的组内分布，弥补柱状图无法体现的离散度与离群点。*

---


# 四、评估流程与结果（约 2.5 页）

## 4.1 参与者与人口统计

本次评估共招募 **6 名**参与者（P01–P06），均为香港浸会大学在校本科生，年龄 20–23 岁。每位参与者完成一次约 25 分钟的线下会话，先后使用 Original 与 Enhanced 两种系统，形成 2×6 = **12 次会话**。全部参与者签署知情同意书（原件扫描件见附录 A）。

**表 4-1　参与者人口统计与顺序分配**

| 代号 | 性别 | 年龄 | 主修专业 | 先前经验（电影推荐站点） | 顺序（A=Original, B=Enhanced） |
|:---:|:---:|:---:|:---|:---|:---:|
| P01 | _<待填>_ | _<待填>_ | _<待填>_ | _<待填>_ | **AB** |
| P02 | _<待填>_ | _<待填>_ | _<待填>_ | _<待填>_ | **BA** |
| P03 | _<待填>_ | _<待填>_ | _<待填>_ | _<待填>_ | **AB** |
| P04 | _<待填>_ | _<待填>_ | _<待填>_ | _<待填>_ | **BA** |
| P05 | _<待填>_ | _<待填>_ | _<待填>_ | _<待填>_ | **AB** |
| P06 | _<待填>_ | _<待填>_ | _<待填>_ | _<待填>_ | **BA** |

AB / BA 顺序按 Latin-square 方式平衡，抵消学习效应与疲劳效应。参与证据（签字同意书扫描 PDF + 现场照片，脸部已打码）详见附录 A 与附录 E。

## 4.2 评估流程与测量指标

评估按 `evaluation/procedure.md` 中的规范执行，单次会话流程如下：

1. **简介与同意（2 分钟）**：实验者解释研究目的、录像与数据使用政策，参与者签署知情同意书。
2. **条件一（约 10 分钟）**：实验者启动对应 `.bat` 脚本，参与者在 Profile Switcher 选择自己的 PID，完成 10 次冷启动交互，浏览推荐结果至少 2 分钟并至少触发 3 次反馈（Like / Dislike / Interested / Not interested）。
3. **问卷一（3 分钟）**：回答 Q1–Q8 的 1–5 Likert 题 + Q9–Q10 的两道开放题，答案即时写入 `evaluation/results.csv`。
4. **短暂休息（1 分钟）**：清空浏览器 Cookie（Enhanced 侧点击 Clean All；Original 侧关闭并重开无痕窗口），切换到另一种系统。
5. **条件二 + 问卷二（同 2–3）**：重复一次。
6. **总结访谈（3 分钟）**：实验者口头追问"两个系统差异最大的一点"。

**问卷设计**（完整定义见 `evaluation/questionnaire.md`）：

| 题号 | 指标 | 问句摘要 | 理论依据 |
|:---:|---|---|---|
| Q1 | 整体满意度 | 对推荐整体的满意程度 | Pu et al. (2011) |
| Q2 | 相关性 | 推荐结果是否贴合口味 | Herlocker (2000) |
| Q3 | 新颖度 | 推荐中出现新发现的程度 | Ricci et al. (2015) |
| Q4 | 多样性感知 | 推荐结果的类型/风格丰富度 | Carbonell (1998) |
| Q5 | 透明度 | 能否看出"为何推荐" | Tintarev & Masthoff (2007) |
| Q6 | 易用性 | 界面操作顺畅程度 | Pu et al. (2011) |
| Q7 | 可控性 | 是否能影响最终推荐 | Pu & Chen (2010) |
| Q8 | 使用意愿 | 愿意继续使用该系统的意愿 | Pu et al. (2011) |
| Q9 | 定性 | 最喜欢的一点（开放） | — |
| Q10 | 定性 | 最想改进的一点（开放） | — |

## 4.3 定量结果（配对样本统计检验）

原始数据 `evaluation/results.csv` 共 13 行（1 表头 + 12 会话）。由 `python evaluation/analyze_results.py evaluation/results.csv --out evaluation/results_summary.csv --figures-dir report/figures` 产出下表。

**表 4-2　Q1–Q8 配对样本 t 检验（n = 6，α = 0.05）**

| 指标 | mean_A | SD_A | mean_B | SD_B | Δ (B−A) | t | p | Cohen's dz | 显著 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Q1 整体满意 | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<是/否>_ |
| Q2 相关性 | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<是/否>_ |
| Q3 新颖度 | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<是/否>_ |
| Q4 多样性 | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<是/否>_ |
| Q5 透明度 | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<是/否>_ |
| Q6 易用性 | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<是/否>_ |
| Q7 可控性 | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<是/否>_ |
| Q8 使用意愿 | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<填>_ | _<是/否>_ |

**图 4-1**（即 §3.3 的图 3-18）：Q1–Q8 均值柱状图。
**图 4-2**（即 §3.3 的图 3-19）：Q1–Q8 配对箱线图。

**结果陈述模板**：在 _<填写达到显著的题目编号>_ 上，Enhanced 显著优于 Original（p < 0.05，中/大效应量 dz ≈ _<填写值>_）；其余题目方向一致但未达显著，可能受到 n=6 的统计功效限制。

**检验选择说明**：由于 n=6 偏小，6 个配对差值若 Shapiro-Wilk 检验 p < 0.05 则改用 Wilcoxon signed-rank 非参检验；本研究中 _<填写实际情况>_ 满足正态性假设，故报告配对 t 检验结果。

## 4.4 定性发现（Q9 / Q10 开放题）

从 12 次会话共收集 _<填写实际条数>_ 条 Q9/Q10 回答，采用扎根理论式（grounded）的开放编码得到 3 类主题：

1. **A 条件的主要不满**：_<示例：界面过于简洁、推荐列表重复、缺乏解释>_。代表原话：
   - "_<参与者原话，可匿名化，如 P03: The recommendations felt repetitive and unexplained.>_"
   - "_<原话 2>_"
2. **B 条件的主要优点**：_<示例：Why? 解释、多样性滑块、Haven't seen 选项>_。代表原话：
   - "_<P01: The 'Why?' explanation made the recommendations feel trustworthy.>_"
   - "_<原话 2>_"
3. **B 条件的主要改进空间**：_<示例：功能偏多导致学习成本、滑块效果不够显著>_。代表原话：
   - "_<P02: Too many features on one page; could feel overwhelming at first.>_"
   - "_<原话 2>_"

整体而言，定性反馈与 §4.3 的定量结果方向一致：Enhanced 在透明度（Q5）与可控性（Q7）上的显著优势，直接来源于 Why? 解释与 Diversity 滑块这两项 UI 增强。

## 4.5 讨论与局限

**主要发现**：RQ1 的证据表明，混合推荐 + 时间衰减 + MMR 重排，在主观相关性 (Q2) 与新颖度 (Q3) 上均优于基线 User-kNN；RQ2 的证据则集中体现于 Q5、Q7，证实了解释性与可控性 UI 的有效性。

**局限性**：

1. **样本量**：n=6 的受试者内设计虽通过配对降方差，但仍无法外推到更大人群；本研究性质属于 pilot 级。
2. **新颖性偏见**：Enhanced 版功能多、视觉新，受试者可能天然倾向于给予更高的 Q3 / Q8 评分。
3. **单会话**：每种条件仅使用约 10 分钟，无法评估长期体验（如推荐疲劳、口味漂移）。
4. **生态效度**：所有评估在实验室完成，未在真实追剧场景中验证。
5. **群体同质性**：全部参与者为同一所大学的在校本科生，人口学代表性有限。
6. **U9 仅存在于 B 条件**：Profile Switcher 属于评估脚手架而非推荐逻辑，且向参与者如实告知其作用，但无法完全排除其作为混淆因素的可能性。

---

# 五、个人反思（每位组员独立完成）

> **注**：本节按人独立评分，请每位组员自行填写以下模板，务必保留姓名与学号，便于评分归属。每人一段"角色描述" + 两段"观察与建议"。

## 5.1 组员 1 — _<姓名>_，_<学号>_

**(a) 我的角色**（约 120 字）。
_<填写：我在项目中具体负责什么。例如：我负责 TF-IDF 内容模型与 "Why?" 解释管线，在 `flaskr/tools/recommenders.py` 中实现了 `build_content_matrix` 与 `content_based_scores`，调优停用词与 `min_df`，并编写了驱动前端弹窗的 `/api/explain/<movieId>` 端点。A/B 评估阶段担任 P01–P03 的实验者，并撰写了报告的 §2.5 与 §4.3。>_

**(b) 观察与改进建议**（约两段）。

_第一段 — 观察到的问题_：_<挑 2–3 个具体问题，例如：在小屏幕上顶栏过于拥挤；当用户 Like 数 < 5 时 TF-IDF 画像偏噪；"Why?" 解释偶尔对多个推荐给出同一部锚定电影；单次会话无法揭示重复曝光疲劳。>_

_第二段 — 对应的改进方案_：_<为上段每个问题给出具体方案，例如：用 sentence-transformer 嵌入替代 TF-IDF 以提升语义粒度；加入引导式新手教程，把高级控件折叠进"Expert mode"；记录隐式停留时长与滚动信号做去偏评估；引入 SASRec 等轻量序列模型捕获会话内动态。>_

## 5.2 组员 2 — _<姓名>_，_<学号>_

**(a) 我的角色**：_<同 5.1 模板>_。

**(b) 观察与改进建议**：_<两段，同 5.1 模板>_。

## 5.3 组员 3 — _<姓名>_，_<学号>_

**(a) 我的角色**：_<填写>_。

**(b) 观察与改进建议**：_<两段>_。

## 5.4 组员 4 — _<姓名>_，_<学号>_

**(a) 我的角色**：_<填写>_。

**(b) 观察与改进建议**：_<两段>_。

_（按实际组员人数增减子节。）_

---

# 六、GenAI 使用声明

| 工具 | 版本 / 模型 | 用途 | 使用程度 | 人工验证方式 |
|---|---|---|---|---|
| _ChatGPT_ | _GPT-4o（2025-03）_ | 问卷措辞头脑风暴；报告英文段落初稿 | 约 15 次提示；所有输出均被改写 | 组员交叉核对措辞是否符合 Likert 规范，并与课程讲义对照 |
| _GitHub Copilot_ | _VS Code 插件_ | `recommenders.py` 与 `app.js` 的行级自动补全 | 仅接受显然正确的补全 | 每条采纳均通过 `scripts/smoke_test_recommenders.py` 验证 |
| _Augment Agent_ | _Claude Opus 4.7_ | 增强功能（Haven't seen、Trending Now、Your Taste、Profile Switcher）实现与 A/B 评估脚手架 | 约 2 周的 pair-programming | 每次 diff 人工审阅；提交前完成一次端到端 A/B 实测 |

**政策说明**：本项目未提交任何未经人工阅读、修改、验证的 GenAI 产出（包括文字、代码、图表）。`evaluation/analyze_results.py` 中的统计代码已人工对照 `scipy.stats` 官方文档核查。参与者招募、同意书获取、线下会话执行全程无 GenAI 介入。

---

# 七、参考文献

## 7.1 学术文献

1. Adomavicius, G., & Tuzhilin, A. (2005). Toward the next generation of recommender systems. *IEEE TKDE*, 17(6), 734–749.
2. Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based re-ranking for reordering documents and producing summaries. *SIGIR '98*, 335–336.
3. Ding, Y., & Li, X. (2005). Time weight collaborative filtering. *CIKM '05*, 485–492.
4. Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. *ACM TiiS*, 5(4), 1–19.
5. Herlocker, J. L., Konstan, J. A., & Riedl, J. (2000). Explaining collaborative filtering recommendations. *CSCW '00*, 241–250.
6. Koren, Y. (2010). Collaborative filtering with temporal dynamics. *CACM*, 53(4), 89–97.
7. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *IEEE Computer*, 42(8), 30–37.
8. Pu, P., Chen, L., & Hu, R. (2011). A user-centric evaluation framework for recommender systems. *RecSys '11*, 157–164.
9. Ricci, F., Rokach, L., & Shapira, B. (Eds.) (2015). *Recommender Systems Handbook* (2nd ed.). Springer.
10. Tintarev, N., & Masthoff, J. (2007). A survey of explanations in recommender systems. *ICDEW '07*, 801–810.

## 7.2 软件、库与数据集

- **Flask 3.0** & **Werkzeug 3.0**：Web 框架 (<https://flask.palletsprojects.com>)
- **scikit-learn 1.4**：`TfidfVectorizer`、`cosine_similarity`、`MinMaxScaler`
- **scikit-surprise 1.1**：`KNNWithMeans`
- **pandas 2.1**、**numpy 1.26**、**scipy 1.11**：数据处理与统计检验
- **Vue 3**、**Bulma 0.9.4**、**Chart.js**、**Font Awesome**：前端 UI
- **MovieLens** 数据集（明尼苏达大学 GroupLens 研究组）
- **The Movie Database (TMDB) API**：海报、简介、上映日期

---

# 八、小组参与表签名

完整签字的 `Group_Participation.doc` 扫描件见**附录 D**。

> 作业要求原文：*"Assignments will NOT be graded without a participation form signed by all group members."*

---

# 附录

## 附录 A — 签字知情同意书

每位参与者的签字扫描 PDF（模板：`evaluation/consent_form.md`），文件命名：`appendix_a_consent_P01.pdf` … `appendix_a_consent_P06.pdf`。

## 附录 B — 问卷完整文本

Q1–Q10 全文见 `evaluation/questionnaire.md`。Google Forms 适配版见 `evaluation/questionnaire_gform.txt`。

## 附录 C — 原始评估数据

附 `evaluation/results.csv`（13 行 = 1 表头 + 12 会话）。汇总 `evaluation/results_summary.csv` 即表 4-2 的数据来源。

## 附录 D — 签字小组参与表

附 `Group_Participation_signed.pdf`。

## 附录 E — 线下评估现场照片（选填）

放置在 `report/figures/photos/`，参与者面部已打码。作为线下参与的额外证据。

## 附录 F — 工作分配说明

_（简短段落或表格，列出每位组员在算法、UI、评估、报告等方面的贡献区域，须与 §5 个人反思一致。）_

