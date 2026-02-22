# Requirement 004 (Refined) — 数据重做、未完成功能、最小接入

**Source:** 用户反馈 + requirement_003, response_003  
**Previous:** requirements_refined_003.md, response_003.md

---

## 1. 目标与范围

1. **Synthetic data 从头重做** — response_003 生成数据「几乎完全错误」，需按规范完全重新实现
2. **继续完成未实现功能** — wedge/fcompensate, multigrid/GPU alignment
3. **最小接入 (Minimal Human Intervention)** — 目标用法为 README 中的 `pydynamo <command> --i config.yaml`，用户仅提供**一个** config 即可运行，**非** `make` / `run.sh` 多步链式

---

## 2. 最小接入 (Usage Target)

**目标用法**（与 pydynamo/README.md 一致）：

```bash
pydynamo <command> --i config.yaml

# 示例
pydynamo crop           --i config_crop.yaml
pydynamo reconstruction --i config_recon.yaml
pydynamo alignment      --i config_align.yaml
pydynamo classification --i config_mra.yaml
```

**原则：**

- 每个命令**一个** config 文件
- 用户只需准备/编辑该 config，无需多步命令、Makefile、脚本链
- 路径在 config 内显式指定；config 可使用相对路径（相对于 cwd 或 config 所在目录，需文档约定）

**不做：**

- 不依赖 `make gen_data`、`make test_crop` 等链式目标作为主用法
- 不强制用户从固定目录运行
- `run.sh` 仅作为 `pydynamo` 的简单封装，非「一键全流程」

---

## 3. Synthetic Data 重做（从头）

### 3.1 问题与重做理由

- 当前生成逻辑与规范不一致，用户反馈「几乎完全错误」
- 需按 requirement_003 及 Dynamo 约定，重新定义并实现

### 3.2 规范定义

| 项目 | 规范 |
|------|------|
| **Tomogram** | XYZ 尺寸 2000×2000×800，pixel size 与 emd_32820_bin4.mrc 一致 |
| **模板** | synthetic_data/emd_32820_bin4.mrc（64×64×64） |
| **Missing wedge** | ±48°（tilt angle） |
| **粒子数** | 1000 个 |
| **坐标** | Dynamo 1-based 中心坐标 (x,y,z) |
| **Euler** | Dynamo ZXZ (tdrot, tilt, narot)，单位：度 |

### 3.3 采样与生成顺序（重要）

- **先采样，再生成**：取向（Euler）和坐标 (x,y,z) 的采样必须在生成 tomogram **之前**完成
- 按此采样表依次生成：tomogram → subtomo（crop）→ 含 noise 的 classification 集

### 3.4 生成步骤（正确流程）

1. **采样（首步）**
   - 采样 1000 个粒子：(tdrot, tilt, narot) 随机 Euler，(x, y, z) 随机 1-based 坐标（保证在体积内、不越界）
   - 采样表供后续全部步骤复用

2. **Tomogram**
   - 创建 2000×2000×800 体积，**背景填充高斯噪声**（均值 0，方差可配置）
   - 按采样表：每个粒子 模板旋转 → 应用 missing wedge（Fourier 空间）→ 嵌入到 tomogram
   - 保存：`synthetic_path/out_tomograms/tomo1.mrc`，pixel_size 写入 MRC header

3. **元数据 (out_tomograms)**
   - `particles.tbl`：Dynamo tbl 格式，列含 tag, x, y, z, tdrot, tilt, narot, tomo, 等（来自采样表）
   - `tomograms.vll`：tomogram 路径（一行一路径）
   - `particles.star`：Relion star 格式，与 tbl 一致

4. **Crop (out_subtomograms)**
   - 按采样表的坐标，从 tomogram crop 出 subtomogram（或用 crop_volume）
   - 输出：`synthetic_path/out_subtomograms/`，每个粒子一个 MRC，sidelength=64

5. **Classification 集 (out_tomograms4classification)**
   - 1000 真实 subtomogram：从 out_subtomograms 复制，沿用采样表
   - 1000 噪声 subtomogram：同尺寸高斯噪声，对应表项更新为 ref=2，坐标/取向占位
   - 合共 2000 个，生成对应 tbl/vll/star
   - 真实 ref=1，噪声 ref=2

### 3.5 坐标与格式约定（严格按 TomoPANDA-pick）

**参考**：requirements_refined_002.md §2 输入 — TomoPANDA-pick 根目录、utils、notebooks

- **Dynamo tbl**（TomoPANDA-pick/utils/io_dynamo.py）：
  - 列 24–26：x, y, z = **绝对坐标**（原点在左上角），**像素单位**
  - 角：tdrot, tilt, narot = ZXZ（Dynamo）
  - `tomogram_size` = (size_x, size_y, size_z) = (nx, ny, nz)

- **RELION star**（TomoPANDA-pick dynamo_df_to_relion）：
  - rlnCenteredCoordinateXAngst/Y/Z = **相对 tomogram 中心**，**埃**
  - 转换：`centered_pixels = absolute_pixels - (tomogram_size/2)`，`centered_angstrom = centered_pixels * pixel_size`
  - rlnAngleRot/Tilt/Psi = ZYZ（RELION）
  - 逆转换：`absolute_pixels = centered_pixels + (tomogram_size/2)`

- **内部元数据标准**：全程使用 **RELION star 格式**（方案 B）
  - 内部：rlnCenteredCoordinateXAngst/Y/Z（相对中心，埃）、rlnAngleRot/Tilt/Psi（ZYZ）、rlnOriginXAngst/Y/Z、rlnMicrographName 等
  - 体积运算（embed、crop）前：`absolute_pixels = centered_angstrom / pixel_size + (tomogram_size / 2)`
  - 输出 tbl 时：按 TomoPANDA-pick `relion_star_to_dynamo_tbl` 转换；输出 star 时直接写入内部表示

### 3.6 crop_volume 的 position 约定

- MRC volume shape = (nz, ny, nx)，即 dim0=z, dim1=y, dim2=x
- `position` 的 `position[i]` 对应 volume 的第 i 维中心
- 故 `position = (z, y, x)`，与 tbl 的 (x, y, z) 顺序不同

### 3.7 embed 与 crop 的轴对应

- 物理坐标 (x, y, z) ↔ numpy 下标：`volume[z_idx, y_idx, x_idx]`，即 axis0=z, axis1=y, axis2=x
- embed：中心 (cx, cy, cz) → 0-based 下标 (cz-1, cy-1, cx-1)
- crop：position = (z, y, x)，与 volume 维度一一对应

### 3.8 Missing wedge（参照 Dynamo table cols 13-17）

- **规范**：https://www.dynamo-em.org/w/index.php?title=Table
  - col 13 **ftype**：0=full, 1=single(tilt about Y), 2=singlex(tilt about X), 3=cone, 4=double
  - col 14-15 **ymintilt, ymaxtilt**：Y 轴 tilt 角度范围
  - col 16-17 **xmintilt, xmaxtilt**：X 轴 tilt 角度范围（ftype 2/4）
- **默认**：Z=电子束，Y=tomo 摆动方向 → ftype=1，wedge 在 **kZ-kY 平面**，angle = arctan2(|ky|, |kz|)，保留 [ymintilt, ymaxtilt]
- **YAML 显式**：`wedge_ftype`, `wedge_ymin`, `wedge_ymax`, `wedge_xmin`, `wedge_xmax`

### 3.9 gen_synthetic 命令

- `pydynamo gen_synthetic --i config_synthetic.yaml`
- config 含：template, output_root, n_particles, n_noise, tomogram_size, apply_missing_wedge, wedge_ftype, wedge_ymin/ymax, wedge_xmin/xmax, noise_sigma, particle_scale_ratio
- 输出路径均由 config 指定，无硬编码
- 可选：gen_synthetic 完成后自动调用 crop（或单独 crop），由 config 开关控制

---

## 4. 功能实现状态

### 4.1 Reconstruction（已实现）

- **wedge**：`apply_wedge`, `wedge_ftype`, `wedge_ymin/ymax`, `wedge_xmin/xmax`（Dynamo tbl 13-17）；Fourier 空间加权累加
- **fcompensate**：`fcompensate: true` 时除以 wedge 权重和

### 4.2 Alignment / Classification（与 Dynamo dcp 一致）

- **cone_step, cone_range**：tilt 搜索步长与范围 [0, 180]
- **inplane_step, inplane_range**：narot/psi 搜索步长与范围 [0, 360]
- **lowpass**：Å，alignment 前 bandpass；null=无滤波
- **multigrid**：`multigrid_levels: 2` 粗→精
- **device**：`cpu|cuda|auto`（NumPy 已实现，PyTorch GPU 待扩展）
- 算法原则与 Dynamo 源码一致（requirements_refined_002 §3.3）

### 4.3 Classification (MRA)

- 与 alignment 共享 cone/inplane/lowpass 参数
- 支持 references 为多个 MRC，tables 为每 ref 的初始表

### 4.4 多 CPU / 多 GPU（终止条件）

- **Crop**：多 CPU 或 GPU 并行（粒子级）
- **Alignment / Classification**：重度依赖多 GPU（PyTorch）；实现完成并测试
- **GPU 测试**：实现完成但先不运行（待有 GPU 环境时执行）

---

## 5. Config 与路径

- 所有关键变量、默认值在 YAML 中显式
- 路径：支持绝对路径；相对路径以 **cwd** 为基准
- 提供示例 config：`test/config_crop_example.yaml` 等，用户可复制并修改

### 5.1 YAML 变量约定

**层级与排序：**

1. **通用信息**（排前）：`pixel_size`, `tomogram_size`, `log_level`, `seed` 等
2. **输入**：`particles`, `subtomograms`, `reference`/`references`, `vll`, `template` 等
3. **输出**：`output`, `output_dir`, `output_star`, `output_table` 等
4. **功能特有参数**（排后）：按算法环节先后顺序

**多环节命令**（如 gen_synthetic、classification）按环节顺序：

- **gen_synthetic**：general → template/output → sampling → tomogram → wedge
- **crop**：general → input → output → crop-specific
- **reconstruction**：general → input → output → average → wedge
- **alignment**：general → input → output → search → filter → masks
- **classification**：general → input → output → iteration → alignment

**注释**：每个变量需有行内注释说明含义与单位

---

## 6. 任务分解

| 序号 | 任务 | 优先级 |
|------|------|--------|
| 1 | Synthetic data 从头重做：规范、坐标、tbl/star/vll 格式 | P0 |
| 2 | gen_synthetic 命令与 config，输出正确 crop/classification 集 | P0 |
| 3 | Reconstruction: wedge + fcompensate | P0 |
| 4 | Alignment: multigrid + PyTorch/GPU（GPU 测试可暂不跑） | P1 |
| 5 | 所有命令统一使用 load_config()，config 自包含 | P1 |
| 6 | README 更新：Usage 为 `pydynamo <command> --i config.yaml` | P1 |
| 7 | response_004.md 进度输出 | P1 |

---

## 7. 输入与生成数据集

| 资源 | 路径 |
|------|------|
| 模板 MRC | synthetic_data/emd_32820_bin4.mrc |
| 无 wedge 数据集 | synthetic_data/（config: apply_missing_wedge: false） |
| 有 wedge 数据集 | synthetic_data_withMissingWedge/（config: apply_missing_wedge: true） |

### 7.1 生成方式与内存

- **生成方式**：全程在内存中构建 tomogram，最后一次性写入 MRC；无流式/增量写盘
- **峰值内存**：主要取决于 tomogram 体积（如 2000×2000×800 float32 ≈ 12 GB）

---

## 8. 输出

| 路径 | 描述 |
|------|------|
| requirements_refined_004.md | 本精炼需求 |
| synthetic_data/out_tomograms/ | tomogram（无 wedge）+ tbl/vll/star |
| synthetic_data/out_subtomograms/ | crop 输出的 subtomogram |
| synthetic_data/out_tomograms4classification/ | 真实+噪声 + tbl/vll/star |
| synthetic_data_withMissingWedge/... | 同上，含 missing wedge 的体数据 |
| pydynamo/README.md | Usage 明确为单 command + 单 config |
| response_004.md | 本轮进度与问题 |

---

## 9. 终止条件

- [x] Synthetic data 按规范从头重做完成
- [x] Reconstruction wedge（Dynamo tbl 13-17）+ fcompensate 实现
- [x] Alignment/Classification cone_range, inplane_range, lowpass（与 Dynamo dcp 一致）
- [x] Wedge kZ-kY 平面、YAML 显式
- [ ] 多 CPU 与多 GPU（PyTorch）实现和测试完成；Crop 多 CPU/GPU；Alignment/Classification 多 GPU
- [ ] GPU 测试：已实现，暂不执行
- [ ] 主用法为 `pydynamo <command> --i config.yaml`，最小接入
- [x] response_004.md 输出
