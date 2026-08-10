# AI4Bone

面向骨缺损修复的可降解金属植入物设计与抗感染辅助工具（GOAI 世界人工智能开源大赛 · 前沿探索赛道）。

## 模块
- `src/structure_design.py` —— 模块 A：面向可降解金属植入物的结构设计与多目标优化
- `src/mg_alloy_baseline.py` —— 模块 A 扩展：镁合金成分-性能预测与逆向设计基线
- `src/phage_baseline.py` —— 模块 B：噬菌体内溶素候选排序（骨架）

## 运行
```
pip install -r requirements.txt
python src/structure_design.py
python src/mg_alloy_baseline.py
python src/phage_baseline.py
```

## 数据
- 正式镁合金数据集：Zenodo `records/17672235`（DatasetMg_imputed.csv，600 个样本）
  下载后放到 `data/DatasetMg_imputed.csv`（已在 .gitignore 中排除）。
- `data/mg_alloy_sample.csv` 与 `data/lysin_candidates.txt` 为占位演示数据，用于验证流程。

## 模型
- 成分-性能预测优先使用随机森林（scikit-learn，R²≈0.80）；无 sklearn 时自动回退线性 ridge。
- 真实数据集清洗：特征空值按中位数填充；缺目标值的 28 行剔除（600→572 行）。

## 可复现
固定随机种子（SEED=42）、公开数据、Dockerfile、依赖锁定（requirements.txt）。

## Git / GitHub 推送
本地仓库已完成 `git init` 与首次提交（main 分支）。
推到 GitHub：
1. 在 GitHub 新建空仓库（例如 `AI4Bone`，不要勾选 README/LICENSE 初始化）。
2. 在仓库根目录执行：
   ```
   git remote add origin https://github.com/<你的用户名>/AI4Bone.git
   git push -u origin main
   ```
3. 日常提交：
   ```
   git add -A
   git commit -m "描述本次改动"
   git push
   ```
首次推送会提示登录，按窗口用你的 GitHub 账号登录即可。


## 双合金场景（镁 / 锌）
- 结构设计管线为「合金无关」：同一套树搜索流程可适配镁（WE43）与锌（纯 Zn、Zn-1Mg），
  材料参数（E、屈服强度、降解时间尺度）来自 `data/materials_lib.json`（文献参考值，正式版以公开数据/实测为准）。
- 镁成分-性能回归使用镁数据集（Zenodo records/17672235）；锌成分建模需单独数据集，留待复赛文献挖掘补充。

### 运行示例
```
python src/structure_design.py                        # we43_mg / cancellous
python src/structure_design.py --material zn_1mg --site cortical
python src/mg_alloy_baseline.py --alloy zn_1mg
python src/mg_alloy_baseline.py --list-alloys
```

## Web Demo
选合金 + 骨部位 → 实时生成候选结构（模量/强度/降解周期 + 3×3×3 孔隙率矩阵 + LPBF 参数）。

本地运行：
```
pip install -r requirements.txt
python -m uvicorn web.main:app --host 127.0.0.1 --port 8765
# 浏览器打开 http://127.0.0.1:8765
```
Docker 运行：
```
docker build -t ai4bone .
docker run -p 8765:8765 ai4bone
```
API：
- `GET /api/meta` — 合金/部位/工艺参数
- `POST /api/design` — `{"material":"zn_1mg","site":"cortical","seed":42}` → 候选结构列表
