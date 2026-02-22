# Response 004 — 执行进度汇总报告

**需求文档:** requirement_003, 用户反馈  
**精炼需求:** requirements_refined_004.md  
**日期:** 2025-02-22

---

## 1. 已完成 (Completed)

### 1.1 更新 requirements_refined_004.md

- 新增 **§3.5 坐标与格式约定**：严格按 TomoPANDA-pick（requirements_refined_002 §2）
- 新增 **§3.6 crop_volume position 约定**：`position = (z, y, x)`
- 新增 **§3.7 embed 与 crop 轴对应**：物理 (x,y,z) ↔ `volume[z_idx, y_idx, x_idx]`
- 更新 **§3.8 Missing wedge**：参照 Dynamo table cols 13-17；默认 Z=beam, Y=tilt；kZ-kY 平面；YAML 显式 wedge_ftype, wedge_ymin/ymax, wedge_xmin/xmax

### 1.2 Synthetic Data 生成逻辑修正

- **内部标准**：全程使用 **RELION star 格式**（rlnCenteredCoordinate Å，rlnAngleRot/Tilt/Psi ZYZ）
- **采样**：`sample_particles_star` 直接输出 star 格式行
- **体积运算**：`star_to_absolute_pixels` 将 star → 绝对像素，embed 用 0-based，crop 用 1-based position
- **输出 tbl**：`_write_tbl_from_star` 调用 TomoPANDA-pick `relion_star_to_dynamo_tbl` 转换
- **输出 star**：直接写入内部 DataFrame
- **旋转**：使用 `euler_zyz_to_rotation_matrix`（ZYZ）与内部角一致

### 1.3 新增改动（粒子信号与 wedge）

- **particle_scale_ratio**：将粒子 std 缩放到 `ratio × noise_sigma` 再 embed
- **wedge（Dynamo tbl 13-17）**：`wedge_ftype`, `wedge_ymin`, `wedge_ymax`, `wedge_xmin`, `wedge_xmax`；ftype=1 默认 kZ-kY 平面

### 1.4 Alignment/Classification（与 Dynamo dcp 一致）

- **cone_range, inplane_range**：搜索范围，默认 [0,180]、[0,360]
- **lowpass**：Å，alignment 前 bandpass；null=无
- 算法原则与 Dynamo 源码一致（requirements_refined_002）

### 1.5 生成数据集

| 数据集 | 路径 | 配置 |
|--------|------|------|
| 无 wedge | synthetic_data/ | `config/synthetic_defaults.yaml`，`apply_missing_wedge: false` |
| 有 wedge | synthetic_data_withMissingWedge/ | `config/synthetic_defaults_withMissingWedge.yaml`，`apply_missing_wedge: true` |

### 1.6 其他讨论结论

- **Missing wedge**：kZ-kY 平面（ftype=1）；Z=beam, Y=tilt
- **内存**：全程在内存中构建 tomogram

### 1.7 代码更新

- **wedge**：`pydynamo/core/wedge.py` get_wedge_mask, apply_wedge（ftype, ymintilt, ymaxtilt, xmintilt, xmaxtilt）
- **gen_synthetic**：使用 core.wedge；YAML wedge_ftype, wedge_ymin/ymax, wedge_xmin/xmax
- **reconstruction**：同上
- **alignment/classification**：cone_range, inplane_range, lowpass, pixel_size

---

## 2. 新增实现（已完成）

### 2.1 Reconstruction: wedge（Dynamo tbl 13-17）

- **wedge**：ftype, wedge_ymin/ymax, wedge_xmin/xmax；Fourier 空间加权累加；fcompensate

### 2.2 Alignment/Classification: Dynamo dcp 参数

- **cone_range, inplane_range**：搜索范围
- **lowpass**：Å bandpass
- **multigrid_levels**

### 2.3 未完成（终止条件）

- **多 CPU**：Crop 支持 num_workers 并行
- **多 GPU（PyTorch）**：Alignment、Classification 多 GPU；实现完成，GPU 测试实现但暂不执行

---

## 3. 验证建议

在沙盒外执行：

```bash
cd pydynamo
OMP_NUM_THREADS=1 pydynamo gen_synthetic --i config/synthetic_defaults.yaml
```

或使用 `config/synthetic_full.yaml`（n_particles: 1000, tomogram_size: [2000,2000,800]）进行完整测试。

生成两类数据集示例：

```bash
pydynamo gen_synthetic --i config/synthetic_defaults.yaml              # -> synthetic_data/
pydynamo gen_synthetic --i config/synthetic_defaults_withMissingWedge.yaml  # -> synthetic_data_withMissingWedge/
```

---

## 4. YAML 配置规范（requirements_refined_004 §5.1）

- **变量排序**：通用信息 → 输入 → 输出 → 功能特有（按算法环节顺序）
- **注释**：每个变量行内注释说明含义与单位
- **多环节命令**：按环节先后整理（如 gen_synthetic: general → sampling → tomogram → wedge）

---

## 5. 输出文件清单

| 路径 | 描述 |
|------|------|
| requirements_refined_004.md | §5.1 YAML 约定；§3.8 wedge；§7 数据集 |
| pydynamo/config/*.yaml | 全部加注释、按层级排序 |
| pydynamo/README.md | 精简安装与运行说明 |
| response_004.md | 本进度报告 |

## 6. 删除

- **config/defaults.yaml**：嵌套结构，未被任何命令加载，已删除
