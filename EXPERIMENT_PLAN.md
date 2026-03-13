# CAPE-SGG 完整实验方案

**目标**: ICANN 2026 (截止3月30日)  
**硬件**: AutoDL RTX PRO 6000 96GB (可租多卡)  
**关键约束**: 每个PredCls实验~9.5h，显存35GB/96GB(可以同GPU跑2个)

---

## 从日志提取的关键事实

```
旧log（修复前，已废弃）: loss_rel=3.93=ln(51)，训练完全崩溃
原因: lr=0.12过高（已修复为effective lr=0.024）

硬件性能:
- PredCls batch=12: 0.53s/iter, 总计~9.5h/50k iter
- SGCls:  ~10h    SGDet batch=8: ~12h
- 显存35.5GB → 一张卡可并行2个PredCls实验
- 验证: ~5min/次, 每2000iter验证一次
```

---

## Phase 0: 修复验证 (第1天前2小时) ⭐最优先

**目的**: 确认代码修复后能正常收敛

```bash
# 在AutoDL上拉取修复后的代码
cd /root/autodl-tmp/penet-main
git pull origin main
rm -rf output/cape_sgg_predcls/  # 删除旧的失败checkpoint

# 快速验证: 5000 iter (~45分钟)
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py \
    --config-file configs/cape_sgg_predcls.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CAPEPrototypeEmbeddingNetwork \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.MAX_ITER 5000 \
    SOLVER.VAL_PERIOD 1000 \
    OUTPUT_DIR ./output/cape_quick_test
```

**检查标准** (看iter 5000的日志):
```
✅ loss_rel < 3.0   (旧log中固定在3.93，如果降到<3说明在学)
✅ loss_dis < 5.0   (旧log中爆炸到1085后固定在1.0)
✅ R@100 > 0.15     (旧log中只有0.025，几乎随机)
✅ mR@100 > 0.05    (旧log中只有0.015)
```

**如果不通过**:
- loss_rel仍然>3.5 → 继续降lr: `SOLVER.BASE_LR 0.001`
- loss_dis爆炸 → 加大warmup: `SOLVER.WARMUP_ITERS 2000`
- R@100<0.10但loss在降 → 正常，只是还没收敛，继续到10000 iter再看

---

## Phase 1: 主实验 (第1-2天) ⭐核心

Phase 0通过后，立即启动。这是论文Table 1的数据来源。

### 实验1.1-1.3: CAPE-SGG Full 三个任务

```bash
# === GPU 0: CAPE-SGG PredCls (Full) ===
CUDA_VISIBLE_DEVICES=0 nohup python tools/relation_train_net.py \
    --config-file configs/cape_sgg_predcls.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CAPEPrototypeEmbeddingNetwork \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    OUTPUT_DIR ./output/cape_predcls_full \
    > logs/cape_predcls_full.log 2>&1 &

# === GPU 0 (第二个进程，同卡): CAPE-SGG SGCls (Full) ===
CUDA_VISIBLE_DEVICES=0 nohup python tools/relation_train_net.py \
    --config-file configs/cape_sgg_predcls.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CAPEPrototypeEmbeddingNetwork \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    OUTPUT_DIR ./output/cape_sgcls_full \
    > logs/cape_sgcls_full.log 2>&1 &
```

⚠️ **重要**: 如果同一张卡跑两个进程时OOM，改为串行跑，或者减batch到8。

```bash
# === 等PredCls和SGCls完成后跑SGDet ===
CUDA_VISIBLE_DEVICES=0 nohup python tools/relation_train_net.py \
    --config-file configs/cape_sgg_predcls.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CAPEPrototypeEmbeddingNetwork \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    SOLVER.IMS_PER_BATCH 8 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    OUTPUT_DIR ./output/cape_sgdet_full \
    > logs/cape_sgdet_full.log 2>&1 &
```

### 实验1.4: PE-NET原始基线 (对照组)

```bash
# === PE-NET baseline PredCls (用原始PrototypeEmbeddingNetwork) ===
CUDA_VISIBLE_DEVICES=0 nohup python tools/relation_train_net.py \
    --config-file configs/e2e_relation_X_101_32_8_FPN_1x.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.MAX_ITER 50000 \
    SOLVER.VAL_PERIOD 2000 \
    GLOVE_DIR ./datasets/vg/ \
    OUTPUT_DIR ./output/penet_baseline_predcls \
    > logs/penet_baseline.log 2>&1 &
```

### Phase 1 检查点 (PredCls跑到iter 10000时，约2小时后)

```bash
# 快速查看当前训练状态
tail -20 logs/cape_predcls_full.log | grep "iter:\|SGG eval"
```

**正常信号**:
```
iter 10000: loss_rel < 2.5, loss_dis < 3.0, R@100 > 0.40
```

**需要调整的信号**:
```
loss_rel > 3.0 at iter 10000 → lr仍偏高，停掉，改BASE_LR到0.001重跑
loss_dis > 50  at iter 2000  → 同上
R@100 < 0.20 at iter 10000  → 可能是FASA权重太大，改FASA_WEIGHT到0.05
```

---

## Phase 2: 消融实验 (第2-4天) ⭐论文核心

消融实验只跑**PredCls任务**，这是SGG论文的标准做法。
所有消融都通过命令行flag控制，不用改代码。

### 消融表设计 (对应论文Table 2)

```
| ID   | 配置                              | 验证什么                    |
|------|-----------------------------------|-----------------------------|
| Base | PE-NET原始                        | 基础对照                    |
| A1   | PE-NET + APT                      | APT独立贡献                 |
| A2   | + APT + CLIP Bridge               | CLIP Bridge独立贡献         |
| A3   | + APT + CLIP Bridge + CAPM        | CAPM独立贡献(核心C1)        |
| A4   | + APT + CLIP Bridge + FASA        | FASA独立贡献                |
| A5   | Full (CAPM + FASA)                | 完整方法(=Phase1实验1.1)    |
| A7   | Full但CAPM→ContextBias            | 原型调制>logit偏置(C3-a)    |
| A8   | Full但CAPM→LearnableGate          | CLIP语义>可学习门控(C3-b)   |
```

### A1: PE-NET + APT (无CLIP Bridge, 无CAPM, 无FASA)

```bash
CUDA_VISIBLE_DEVICES=0 nohup python tools/relation_train_net.py \
    --config-file configs/cape_sgg_predcls.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CAPEPrototypeEmbeddingNetwork \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_RELATION_HEAD.CAPE.USE_CLIP_BRIDGE False \
    MODEL.ROI_RELATION_HEAD.CAPE.USE_CAPM False \
    MODEL.ROI_RELATION_HEAD.CAPE.USE_FASA False \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.MAX_ITER 50000 \
    OUTPUT_DIR ./output/ablation_A1_apt_only \
    > logs/ablation_A1.log 2>&1 &
```

### A2: + CLIP Bridge (无CAPM, 无FASA)

```bash
CUDA_VISIBLE_DEVICES=0 nohup python tools/relation_train_net.py \
    --config-file configs/cape_sgg_predcls.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CAPEPrototypeEmbeddingNetwork \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_RELATION_HEAD.CAPE.USE_CAPM False \
    MODEL.ROI_RELATION_HEAD.CAPE.USE_FASA False \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.MAX_ITER 50000 \
    OUTPUT_DIR ./output/ablation_A2_clip_bridge \
    > logs/ablation_A2.log 2>&1 &
```

### A3: + CAPM (无FASA)

```bash
CUDA_VISIBLE_DEVICES=0 nohup python tools/relation_train_net.py \
    --config-file configs/cape_sgg_predcls.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CAPEPrototypeEmbeddingNetwork \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_RELATION_HEAD.CAPE.USE_FASA False \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.MAX_ITER 50000 \
    OUTPUT_DIR ./output/ablation_A3_capm_only \
    > logs/ablation_A3.log 2>&1 &
```

### A4: + FASA (无CAPM)

```bash
CUDA_VISIBLE_DEVICES=0 nohup python tools/relation_train_net.py \
    --config-file configs/cape_sgg_predcls.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CAPEPrototypeEmbeddingNetwork \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_RELATION_HEAD.CAPE.USE_CAPM False \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.MAX_ITER 50000 \
    OUTPUT_DIR ./output/ablation_A4_fasa_only \
    > logs/ablation_A4.log 2>&1 &
```

### A7: ContextBias替代CAPM (论文贡献C3-a的核心证据)

```bash
CUDA_VISIBLE_DEVICES=0 nohup python tools/relation_train_net.py \
    --config-file configs/cape_sgg_predcls.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CAPEPrototypeEmbeddingNetwork \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_RELATION_HEAD.CAPE.CAPM_VARIANT context_bias \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.MAX_ITER 50000 \
    OUTPUT_DIR ./output/ablation_A7_context_bias \
    > logs/ablation_A7.log 2>&1 &
```

### A8: LearnableGate替代CLIP gate (论文贡献C3-b的核心证据)

```bash
CUDA_VISIBLE_DEVICES=0 nohup python tools/relation_train_net.py \
    --config-file configs/cape_sgg_predcls.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CAPEPrototypeEmbeddingNetwork \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_RELATION_HEAD.CAPE.CAPM_VARIANT learnable_gate \
    SOLVER.IMS_PER_BATCH 12 \
    SOLVER.MAX_ITER 50000 \
    OUTPUT_DIR ./output/ablation_A8_learnable_gate \
    > logs/ablation_A8.log 2>&1 &
```

---

## Phase 3: 超参数敏感性 (第4-5天)

### FASA权重搜索

```bash
# λ = 0.05
... MODEL.ROI_RELATION_HEAD.CAPE.FASA_WEIGHT 0.05 \
    OUTPUT_DIR ./output/hyper_fasa_005 > logs/hyper_fasa_005.log 2>&1 &

# λ = 0.2
... MODEL.ROI_RELATION_HEAD.CAPE.FASA_WEIGHT 0.2 \
    OUTPUT_DIR ./output/hyper_fasa_02 > logs/hyper_fasa_02.log 2>&1 &

# λ = 0.5
... MODEL.ROI_RELATION_HEAD.CAPE.FASA_WEIGHT 0.5 \
    OUTPUT_DIR ./output/hyper_fasa_05 > logs/hyper_fasa_05.log 2>&1 &
```

### CAPM注意力头数搜索

```bash
# heads = 2
... MODEL.ROI_RELATION_HEAD.CAPE.CAPM_NUM_HEADS 2 \
    OUTPUT_DIR ./output/hyper_heads_2 > logs/hyper_heads_2.log 2>&1 &

# heads = 8
... MODEL.ROI_RELATION_HEAD.CAPE.CAPM_NUM_HEADS 8 \
    OUTPUT_DIR ./output/hyper_heads_8 > logs/hyper_heads_8.log 2>&1 &
```

---

## Phase 4: 补充实验 + 定性分析 (第5-7天)

### 4.1 PE-NET基线SGCls + SGDet (补齐Table 1)

```bash
# PE-NET SGCls
... MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    OUTPUT_DIR ./output/penet_baseline_sgcls

# PE-NET SGDet  
... MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    SOLVER.IMS_PER_BATCH 8 \
    OUTPUT_DIR ./output/penet_baseline_sgdet
```

### 4.2 定性分析 (用最佳模型的checkpoint)

```bash
python tools/cape_analysis.py \
    --checkpoint ./output/cape_predcls_full/model_final.pth \
    --config-file configs/cape_sgg_predcls.yaml \
    --output_dir ./analysis_results/
```

生成:
- t-SNE原型可视化 (Figure 2)
- Gate激活热力图 (Figure 3)
- Per-predicate Recall分组柱状图 (Figure 4)
- FASA锚点距离分析 (Figure 5)

---

## 时间调度总览

```
                    单卡调度 (保守方案)
Day 0 (2h):   Phase 0 快速验证
Day 1 (20h):  1.1 CAPE PredCls + 1.4 PE-NET baseline (并行)
Day 2 (20h):  1.2 CAPE SGCls + A1 APT-only (并行)
Day 3 (20h):  A2 CLIP-Bridge + A3 CAPM-only (并行)
Day 4 (20h):  A4 FASA-only + A7 ContextBias (并行)
Day 5 (20h):  A8 LearnableGate + FASA-λ搜索 (并行)
Day 6 (12h):  1.3 CAPE SGDet
Day 7 (20h):  CAPM-heads搜索 + PE-NET SGCls/SGDet
Day 8:        定性分析 + 整理数据
---
总计: ~8天完成全部实验

如果租2张卡: 4天完成
如果租4张卡: 2天完成
```

---

## 实验优先级排序 (如果时间不够)

```
P0 (论文必须有): 
  ① CAPE-Full PredCls          ← Table 1主结果
  ② PE-NET baseline PredCls    ← Table 1对照
  ③ A7 ContextBias             ← C3贡献(a)的证据
  ④ A8 LearnableGate           ← C3贡献(b)的证据

P1 (审稿人大概率要求):
  ⑤ A3 +CAPM only              ← CAPM独立贡献
  ⑥ A4 +FASA only              ← FASA独立贡献
  ⑦ CAPE-Full SGCls            ← Table 1多任务
  ⑧ CAPE-Full SGDet            ← Table 1多任务

P2 (有就更好):
  ⑨ A1 APT-only                ← APT贡献
  ⑩ A2 CLIP-Bridge             ← CLIP贡献
  ⑪ 超参数敏感性               ← Table 4
  ⑫ 定性分析                   ← Figures
```

---

## 目标指标与Go/No-Go判定

### PredCls (主战场)

```
你的直接竞争对手:
PE-NET (CVPR'23):     mR@50=31.5  R@50=64.9  F@50=42.4
RA-SGG (AAAI'25):     mR@50=36.2  R@50=62.2  F@50=45.7
HIERCOM+IE (WACV'25): mR@50=38.0  R@50=60.4  F@50=46.6
```

**CAPE-SGG最终目标**:

| mR@50结果 | 判定 | 论文描述策略 |
|-----------|------|-------------|
| ≥ 39.0 | SOTA | "CAPE-SGG achieves new state-of-the-art F@50" |
| 37.5-38.9 | 强GO | "competitive with HIERCOM while being much simpler" |
| 36.5-37.4 | GO | "surpasses RA-SGG by X points on mR@50" |
| < 36.2 | 需调整 | 调FASA权重/lr，或加Reweight策略 |

**关键约束**: R@50 ≥ 60.0 (不能去偏过度)

### 消融结果预期 (用于验证三个贡献点)

```
C1验证: A3 (有CAPM) 的mR@50 应该比 A2 (无CAPM) 高 ≥ 1.5
        → "CAPM provides +X.X mR@50 improvement"

C2验证: A5 (Full) 应该比任何单组件高
        → "Three components are complementary"

C3-a验证: A5 (Full) 的mR@50 应该比 A7 (ContextBias) 高 ≥ 1.0
        → "Prototype-space modulation > logit-space bias"

C3-b验证: A5 (Full) 的mR@50 应该比 A8 (LearnableGate) 高 ≥ 0.5
        → "CLIP semantic structure provides beneficial inductive bias"
```

---

## 应急方案

### 情况A: mR提升但R下降太多 (R@50 < 58)

```bash
# 降低FASA权重
... MODEL.ROI_RELATION_HEAD.CAPE.FASA_WEIGHT 0.03
```

### 情况B: mR提升不明显 (mR@50 < 35)

```bash
# 方案1: 增大FASA权重推尾部
... MODEL.ROI_RELATION_HEAD.CAPE.FASA_WEIGHT 0.3

# 方案2: 增大CAPM的调制力度
# 在capm_module.py中: self.mod_scale = nn.Parameter(torch.tensor(0.3))
```

### 情况C: 训练不稳定/震荡

```bash
# 使用更保守的lr
SOLVER.BASE_LR 0.001

# 或者使用Adam优化器 (需要改代码)
# 在 maskrcnn_benchmark/solver/build.py 中替换SGD为AdamW
```

### 情况D: SGDet OOM

```bash
# 减小batch
SOLVER.IMS_PER_BATCH 6
# 或减小CAPM chunk size (需要改代码)
# 在forward中: capm_chunk_size = 512
```

---

## 实验日志监控脚本

创建一个快速检查脚本:

```bash
#!/bin/bash
# save as: scripts/check_all.sh

echo "=== 实验进度总览 ==="
for dir in output/*/; do
    name=$(basename $dir)
    last_ckpt=$(ls -t $dir/model_*.pth 2>/dev/null | head -1)
    if [ -n "$last_ckpt" ]; then
        iter=$(echo $last_ckpt | grep -o '[0-9]*' | tail -1)
        echo "✅ $name: iter $iter / 50000"
    else
        echo "⏳ $name: not started or no checkpoint"
    fi
done

echo ""
echo "=== 最新loss ==="
for log in logs/*.log; do
    name=$(basename $log .log)
    last_loss=$(grep "loss_rel:" $log 2>/dev/null | tail -1 | grep -o "loss_rel: [0-9.]*")
    last_iter=$(grep "iter:" $log 2>/dev/null | tail -1 | grep -o "iter: [0-9]*")
    if [ -n "$last_loss" ]; then
        echo "$name: $last_iter $last_loss"
    fi
done
```

---

## 结果收集模板

每个实验完成后，提取这组数字:

```
实验名: _______________
R@20:   ___   R@50:  ___   R@100:  ___
mR@20:  ___   mR@50: ___   mR@100: ___
F@50 = 2×R@50×mR@50/(R@50+mR@50) = ___
F@100 = 2×R@100×mR@100/(R@100+mR@100) = ___
```

从日志中提取的grep命令:
```bash
grep "SGG eval:.*R @\|SGG eval:.*mR @" logs/cape_predcls_full.log | tail -6
```
