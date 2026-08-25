"""
2017 CUMCM A题 第三问（共享模型版）
由附件5的含噪声投影数据重建未知介质的位置、几何形状与吸收率，
并对附件4的10个指定位置给出吸收率。

模型完全沿用第二问（共享模块 ct_shared/ct_model.py 提供的
投影矩阵 A、增益与 LSQR 重建接口）；仅针对附件5的噪声做两项处理：
  (1) 正弦图轻度高斯降噪（探测器向 sigma=1.5，角度向 sigma=1.0）；
  (2) LSQR 迭代次数取 60（半收敛正则化，抑制噪声放大）。

流程：
  第三阶段：LSQR 求解 y = A u（Radon 逆变换），重建附件5
  第四阶段：位置、几何形状与吸收率统计分析（多孔介质版本）
  第六阶段：双线性/双三次插值比较（抽稀交叉验证 + MAE/RMSE/R^2），
            用最优方法给出10个指定位置的吸收率
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"
]
plt.rcParams["axes.unicode_minus"] = False

base_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_dir, "figures")
os.makedirs(save_dir, exist_ok=True)

# -------------------- 共享投影模型 --------------------
sys.path.insert(0, os.path.join(base_dir, "..", "ct_shared"))
from ct_model import CTModel

model = CTModel()
N, delta = model.N, model.delta
model.print_validation()

# =========================
# 第三阶段：Radon 逆变换重建附件5（含噪声）
# =========================
file_path = os.path.join("A题附件.xls")
proj5 = pd.read_excel(file_path, sheet_name="附件5", header=None).values
proj5_s = ndimage.gaussian_filter(proj5, sigma=(1.5, 1.0))   # 正弦图降噪

# 含噪数据下迭代次数即正则化参数：迭代越多噪声放大越明显，取 60 次
U3 = model.reconstruct(proj5, iter_lim=60, denoise_sigma=(1.5, 1.0))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
im0 = axes[0].imshow(proj5_s, aspect="auto", cmap="gray", origin="lower")
axes[0].set_title("附件5：未知介质投影数据（降噪后正弦图）")
axes[0].set_xlabel("扫描序号"); axes[0].set_ylabel("探测器编号")
fig.colorbar(im0, ax=axes[0], label="接收信息")
im1 = axes[1].imshow(U3, cmap="gray", origin="lower", extent=[0, 100, 0, 100])
axes[1].set_title("附件5重建图像 $\\hat{U}(x,y)$")
axes[1].set_xlabel("x / mm"); axes[1].set_ylabel("y / mm")
fig.colorbar(im1, ax=axes[1], label="吸收率")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "attachment5_reconstruction.png"), dpi=150, bbox_inches="tight")
plt.close()

# 保存重建吸收率矩阵（第0行对应 y=0，第0列对应 x=0，像素边长 100/256 mm）
pd.DataFrame(U3).to_csv(
    os.path.join(base_dir, "problem3.csv"),
    index=False, header=False, float_format="%.4f"
)

# =========================
# 第四阶段：位置、形状与吸收率分析（多孔介质）
# =========================

def ellipse_params(mask):
    """由区域二阶矩求等效椭圆：质心、长/短半轴、长轴倾角。"""
    ys, xs = np.where(mask)
    x = (xs + 0.5) * delta
    y = (ys + 0.5) * delta
    cx, cy = x.mean(), y.mean()
    cov = np.cov(np.vstack([x, y]))
    eigval, eigvec = np.linalg.eigh(cov)
    a_minor, a_major = 2 * np.sqrt(eigval[0]), 2 * np.sqrt(eigval[1])
    ang_deg = np.degrees(np.arctan2(eigvec[1, 1], eigvec[0, 1]))
    return cx, cy, a_major, a_minor, ang_deg


records = []


def region_report(name, mask, values):
    cx, cy, amaj, amin, ang_deg = ellipse_params(mask)
    ys, xs = np.where(mask)
    rec = {
        "区域": name,
        "面积_mm2": mask.sum() * delta ** 2,
        "质心x_mm": cx, "质心y_mm": cy,
        "包围盒x_min_mm": xs.min() * delta, "包围盒x_max_mm": (xs.max() + 1) * delta,
        "包围盒y_min_mm": ys.min() * delta, "包围盒y_max_mm": (ys.max() + 1) * delta,
        "等效椭圆长半轴_mm": amaj, "等效椭圆短半轴_mm": amin,
        "长宽比": amaj / amin, "长轴倾角_度": ang_deg,
        "平均吸收率": values.mean(), "吸收率标准差": values.std(),
        "吸收率中位数": np.median(values),
    }
    records.append(rec)
    print("\n%s：" % name)
    print("  面积       = %.1f mm²" % rec["面积_mm2"])
    print("  质心位置   = (%.2f, %.2f) mm" % (cx, cy))
    print("  包围盒     = x:[%.1f, %.1f]，y:[%.1f, %.1f] mm"
          % (rec["包围盒x_min_mm"], rec["包围盒x_max_mm"],
             rec["包围盒y_min_mm"], rec["包围盒y_max_mm"]))
    print("  等效椭圆   ：长半轴 %.2f mm，短半轴 %.2f mm，长宽比 %.2f，长轴倾角 %.1f°"
          % (amaj, amin, amaj / amin, ang_deg))
    print("  平均吸收率 = %.3f ± %.3f（中位数 %.3f）"
          % (values.mean(), values.std(), np.median(values)))


print("\n第四阶段：位置、形状与吸收率分析")

# 显示与区域分割使用轻度平滑版本（抑制噪声碎斑），吸收率统计仍用原始重建值
U3s = ndimage.gaussian_filter(U3, sigma=1.2)

# 1) 介质整体（外轮廓）：开运算断开噪声桥，取最大连通域并填充、闭合
m = ndimage.binary_opening(U3s > 0.6, structure=np.ones((3, 3)))
lab, nreg = ndimage.label(m)
sizes = ndimage.sum(np.ones_like(lab), lab, range(1, nreg + 1))
outer = ndimage.binary_fill_holes(lab == (np.argmax(sizes) + 1))
outer = ndimage.binary_closing(outer, structure=np.ones((3, 3)))
region_report("介质整体（外轮廓）", outer, U3[outer])
ys_o, xs_o = np.where(outer)
print("  -> 介质范围 x:[%.1f, %.1f] mm，y:[%.1f, %.1f] mm，几乎充满整个托盘"
      % (xs_o.min() * delta, (xs_o.max() + 1) * delta,
         ys_o.min() * delta, (ys_o.max() + 1) * delta))

# 2) 固体基体（外轮廓内、剔除孔隙与高吸收核）
solid = outer & (U3s >= 0.5) & (U3s <= 4.5)
region_report("固体基体", solid, U3[solid])

# 3) 高吸收核（u > 4.5 的亮结）
lab_h, nh = ndimage.label(U3s > 4.5)
sizes_h = np.array(ndimage.sum(np.ones_like(lab_h), lab_h, range(1, nh + 1))) * delta ** 2
high_masks = [lab_h == (i + 1) for i in range(nh) if sizes_h[i] > 5]
high_masks.sort(key=lambda mm: -U3[mm].mean())   # 按平均吸收率从大到小
print("\n高吸收核（>4.5 且面积>5 mm²）：共 %d 个" % len(high_masks))
for k, mm in enumerate(high_masks, 1):
    region_report("高吸收核%d" % k, mm, U3[mm])

# 4) 内部孔隙（外轮廓内 u < 0.5）
pores = outer & (U3s < 0.5)
lab_o, no = ndimage.label(pores)
sizes_o = np.array(ndimage.sum(np.ones_like(lab_o), lab_o, range(1, no + 1))) * delta ** 2
hole_masks = [lab_o == (i + 1) for i in range(no) if sizes_o[i] > 5]
hole_areas = np.array([mm.sum() * delta ** 2 for mm in hole_masks])
order = np.argsort(-hole_areas)          # 按面积从大到小
hole_masks = [hole_masks[i] for i in order]
hole_areas = hole_areas[order]
print("\n孔隙（面积>5 mm²）：共 %d 个" % len(hole_masks))
print("  孔隙率 = %.1f%%（孔隙总面积 / 外轮廓面积）"
      % (100 * pores.sum() / outer.sum()))
print("  孔隙面积：中位 %.1f mm²，最大 %.1f mm²，最小 %.1f mm²"
      % (np.median(hole_areas), hole_areas.max(), hole_areas.min()))
for k, mm in enumerate(hole_masks[:5], 1):   # 打印最大的5个，其余存入CSV
    region_report("孔隙%d（按面积降序）" % k, mm, U3[mm])
for k, mm in enumerate(hole_masks[5:], 6):
    ys, xs = np.where(mm)
    records.append({
        "区域": "孔隙%d（按面积降序）" % k,
        "面积_mm2": mm.sum() * delta ** 2,
        "质心x_mm": (xs.mean() + 0.5) * delta, "质心y_mm": (ys.mean() + 0.5) * delta,
        "包围盒x_min_mm": xs.min() * delta, "包围盒x_max_mm": (xs.max() + 1) * delta,
        "包围盒y_min_mm": ys.min() * delta, "包围盒y_max_mm": (ys.max() + 1) * delta,
        "等效椭圆长半轴_mm": np.nan, "等效椭圆短半轴_mm": np.nan,
        "长宽比": np.nan, "长轴倾角_度": np.nan,
        "平均吸收率": U3[mm].mean(), "吸收率标准差": U3[mm].std(),
        "吸收率中位数": np.median(U3[mm]),
    })

pd.DataFrame(records).to_csv(
    os.path.join(base_dir, "区域统计分析.csv"),
    index=False, encoding="utf-8-sig", float_format="%.4f"
)

# 分割结果可视化
seg = np.zeros_like(U3)          # 0=背景, 1=固体基体, 2=高吸收核, 3=孔隙
seg[outer] = 1
for mm in high_masks:
    seg[mm] = 2
for mm in hole_masks:
    seg[mm] = 3

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
im0 = axes[0].imshow(U3, cmap="gray", origin="lower", extent=[0, 100, 0, 100])
axes[0].contour(np.linspace(delta/2, 100-delta/2, N), np.linspace(delta/2, 100-delta/2, N),
                U3s, levels=[0.5, 4.5], colors=["cyan", "red"], linewidths=1.0)
axes[0].set_title("重建图像与区域边界（青: 外轮廓, 红: 高吸收核）")
axes[0].set_xlabel("x / mm"); axes[0].set_ylabel("y / mm")
fig.colorbar(im0, ax=axes[0], label="吸收率")
from matplotlib.colors import ListedColormap
cmap_seg = ListedColormap(["black", "lightgray", "tomato", "white"])
im1 = axes[1].imshow(seg, cmap=cmap_seg, origin="lower", extent=[0, 100, 0, 100], vmin=0, vmax=3)
axes[1].set_title("区域分割（浅灰=固体基体, 红=高吸收核, 白=孔隙）")
axes[1].set_xlabel("x / mm"); axes[1].set_ylabel("y / mm")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "region_segmentation.png"), dpi=150, bbox_inches="tight")
plt.close()

# 吸收率直方图
plt.figure(figsize=(8, 5))
plt.hist(U3[outer].ravel(), bins=200)
plt.xlabel("吸收率")
plt.ylabel("像素数")
plt.title("介质区域内重建吸收率分布")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "absorption_histogram.png"), dpi=150, bbox_inches="tight")
plt.close()

# =========================
# 第六阶段：指定位置吸收率精细估计与插值方法比较
# =========================
# 与第二问相同：双线性/双三次插值 + 抽稀交叉验证（MAE/RMSE/R^2），
# 以 RMSE 为主判据选出最优方法，再计算10个指定位置的最终吸收率。

points = pd.read_excel(file_path, sheet_name="附件4", header=None).values.astype(float)
print("\n第六阶段：10个指定位置吸收率估计")
print("附件4指定位置数量：", len(points))


def interp_values(image, coords_yx, order):
    """在像素中心网格上插值。coords_yx: (M,2)，列为浮点像素坐标 (iy, ix)。"""
    return ndimage.map_coordinates(
        image, [coords_yx[:, 0], coords_yx[:, 1]], order=order, mode="nearest"
    )


def mm_to_pixel(x, y):
    """毫米坐标 -> 浮点像素坐标（像素 k 的中心对应 ((k+0.5)*delta) mm）"""
    return x / delta - 0.5, y / delta - 0.5


# ---- 6.1 抽稀交叉验证 ----
coarse = U3[::2, ::2]
val_mask = np.ones((N, N), dtype=bool)
val_mask[::2, ::2] = False
iy_v, ix_v = np.where(val_mask)
coords_v = np.column_stack([iy_v / 2.0, ix_v / 2.0])
truth_v = U3[val_mask]

pred_L = interp_values(coarse, coords_v, order=1)
pred_C = interp_values(coarse, coords_v, order=3)


def metrics(truth, pred):
    err = truth - pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    r2 = 1 - np.sum(err ** 2) / np.sum((truth - np.mean(truth)) ** 2)
    return mae, rmse, r2


mae_L, rmse_L, r2_L = metrics(truth_v, pred_L)
mae_C, rmse_C, r2_C = metrics(truth_v, pred_C)
print("交叉验证点数 M = %d" % len(truth_v))
print("双线性插值：MAE = %.4f, RMSE = %.4f, R^2 = %.4f" % (mae_L, rmse_L, r2_L))
print("双三次插值：MAE = %.4f, RMSE = %.4f, R^2 = %.4f" % (mae_C, rmse_C, r2_C))

compare = pd.DataFrame({
    "插值方法": ["双线性插值", "双三次插值"],
    "MAE": [mae_L, mae_C],
    "RMSE": [rmse_L, rmse_C],
    "R2": [r2_L, r2_C],
})
compare.to_csv(
    os.path.join(base_dir, "插值方法比较.csv"),
    index=False, encoding="utf-8-sig", float_format="%.6f"
)

if rmse_C <= rmse_L:
    best_order, best_name = 3, "双三次插值"
else:
    best_order, best_name = 1, "双线性插值"
print("按 RMSE 主判据，最优方法：%s" % best_name)

# ---- 6.2 用两种方法计算10个指定位置，最终采用最优方法 ----
ix_f, iy_f = mm_to_pixel(points[:, 0], points[:, 1])
coords_p = np.column_stack([iy_f, ix_f])
val_L = interp_values(U3, coords_p, order=1)
val_C = interp_values(U3, coords_p, order=3)
val_best = val_C if best_order == 3 else val_L

# 标注每个点所属区域（沿用第四阶段的分割结果）
high_all = np.zeros_like(U3, dtype=bool)
for mm in high_masks:
    high_all |= mm
hole_all = np.zeros_like(U3, dtype=bool)
for mm in hole_masks:
    hole_all |= mm


def region_of(x, y):
    ix = int(np.clip(np.floor(x / delta), 0, N - 1))
    iy = int(np.clip(np.floor(y / delta), 0, N - 1))
    if high_all[iy, ix]:
        return "高吸收核"
    if hole_all[iy, ix]:
        return "孔隙"
    if outer[iy, ix]:
        return "固体基体"
    return "背景"


regions = [region_of(px, py) for px, py in points]

result10 = pd.DataFrame({
    "位置编号": np.arange(1, len(points) + 1),
    "x_mm": points[:, 0],
    "y_mm": points[:, 1],
    "双线性插值": val_L,
    "双三次插值": val_C,
    "最终采用(%s)" % best_name: val_best,
    "所在区域": regions,
})
result10.to_csv(
    os.path.join(base_dir, "十个指定位置吸收率.csv"),
    index=False, encoding="utf-8-sig", float_format="%.4f"
)
print("\n10个指定位置的吸收率（最终采用：%s）：" % best_name)
print(result10.to_string(index=False))

# ---- 6.3 可视化：在重建图像上标注10个指定位置 ----
plt.figure(figsize=(7, 7))
plt.imshow(U3, cmap="gray", origin="lower", extent=[0, 100, 0, 100])
plt.colorbar(label="吸收率")
plt.scatter(points[:, 0], points[:, 1], c="red", s=40, marker="x", linewidths=1.5)
label_offsets = {  # 避免相邻编号标注互相遮挡
    1: (6, 6), 2: (6, -14), 3: (6, 8), 4: (-34, 6), 5: (6, 8),
    6: (6, 12), 7: (10, -6), 8: (6, 6), 9: (6, 6), 10: (-44, 6),
}
for k, (px, py) in enumerate(points, 1):
    plt.annotate(
        "%d: %.2f" % (k, val_best[k - 1]), (px, py),
        textcoords="offset points", xytext=label_offsets.get(k, (6, 6)),
        fontsize=9, color="red"
    )
plt.xlim(0, 100); plt.ylim(0, 100)
plt.xlabel("x / mm"); plt.ylabel("y / mm")
plt.title("10个指定位置的吸收率（%s）" % best_name)
plt.savefig(os.path.join(save_dir, "ten_points_absorption.png"), dpi=150, bbox_inches="tight")
plt.close()

print("\n全部完成。")
print("图片已保存到：", save_dir)
print("重建吸收率矩阵：problem3.csv")
print("区域统计结果：区域统计分析.csv")
print("插值方法比较：插值方法比较.csv")
print("10点吸收率：十个指定位置吸收率.csv")
