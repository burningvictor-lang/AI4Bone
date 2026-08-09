# AI4Bone

AI 驱动的骨缺损修复智能材料设计与抗感染方案平台（GOAI 世界人工智能开源大赛 · 前沿探索赛道）

## 模块
- `src/mcts_structure_design.py` — 模块 A：基于蒙特卡洛树搜索（MCTS）的个性化可降解金属骨科植入物设计（对应专利《一种基于蒙特卡洛树搜索的可降解金属骨科植入物设计方法》）
- `src/mg_alloy_baseline.py` — 模块 A 扩展：镁合金成分-性能预测 + 逆向设计基线
- `src/phage_baseline.py` — 模块 B：感染性骨缺损噬菌体内溶素候选排序（骨架）

## 运行
```
pip install -r requirements.txt
python src/mcts_structure_design.py
python src/mg_alloy_baseline.py
python src/phage_baseline.py
```

## 数据
- 正式镁合金数据集：Zenodo `records/17672235`（DatasetMg_imputed.csv，410 样本）
  下载后放到 `data/DatasetMg_imputed.csv`（.gitignore 已排除）。
- `data/mg_alloy_sample.csv` / `data/lysin_candidates.txt` 为占位演示数据。

## 可复现
固定随机种子（SEED=42）、公开数据、Dockerfile、requirements 锁定。
