"""
第二问
由附件3的投影数据重建未知介质的位置、几何形状与吸收率。

方案（四阶段）：
  第一阶段：利用第一问标定参数 (x0, y0, d, theta_1..theta_180) 建立实际CT投影模型
  第二阶段：将 100mm x 100mm 成像区域离散为 256x256 像素，
            用 Siddon 射线追踪计算每条射线在各像素内的穿过长度 a_ijk，
            得到离散投影方程 y = A u
  第三阶段：用 LSQR 迭代求解 y = A u（即 Radon 逆变换），
            先用附件1/2的标准模板验证整条链路，再重建附件3
  第四阶段：对重建图像做位置、几何形状与吸收率统计分析

输入（与本程序同目录）：
  A题附件.xls            附件1/2/3
  calibration_result.csv 第一问标定结果（旋转中心、探测器间距）
  calibrated_angles.csv  第一问标定出的180个射线方向
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.sparse.linalg import LinearOperator, lsqr

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"
]
plt.rcParams["axes.unicode_minus"] = False

base_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_dir, "figures")
os.makedirs(save_dir, exist_ok=True)

# ------------------------- 数据读入 -------------------------
file_path = os.path.join("A题附件.xls")
template = pd.read_excel(file_path, sheet_name="附件1", header=None).values
proj2 = pd.read_excel(file_path, sheet_name="附件2", header=None).values
proj3 = pd.read_excel(file_path, sheet_name="附件3", header=None).values

cal = pd.read_csv(os.path.join("c:\\Users\\51932\\Desktop\\2017-A\\2017-A-Q1\\tables", "calibration_result.csv"))
ang = pd.read_csv(os.path.join("c:\\Users\\51932\\Desktop\\2017-A\\2017-A-Q1\\tables", "calibrated_angles.csv"))
x0, y0, d = cal["x0"][0], cal["y0"][0], cal["d"][0]
theta = np.deg2rad(ang["angle_degree"].values)   # 180个单独修正后的射线方向

print("第一问标定参数：")
print("旋转中心 (x0, y0) = (%.4f, %.4f) mm" % (x0, y0))
print("探测器单元间距 d  = %.4f mm" % d)
print("射线方向范围      = %.4f° ~ %.4f°" % (np.degrees(theta[0]), np.degrees(theta[-1])))

# 探测器零点 b（第一问正弦拟合的中间量，未存盘，这里用同一方法重算）
observed_scans, observed_centers = [], []
for scan in range(proj2.shape[1]):
    y = proj2[:, scan]
    mask = y > 0
    change = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0] - 1
    narrow = [(l, r) for l, r in zip(starts, ends) if 10 <= r - l + 1 <= 45]
    if len(narrow) != 1:
        continue
    l, r = narrow[0]
    idx = np.arange(l, r + 1)
    w = y[l:r + 1]
    observed_scans.append(scan + 1)
    observed_centers.append(np.sum(idx * w) / np.sum(w))
observed_scans = np.array(observed_scans)
observed_centers = np.array(observed_centers)


def sine_model(i, A, omega, phi, b):
    return A * np.sin(omega * i + phi) + b


popt, _ = curve_fit(
    sine_model, observed_scans, observed_centers,
    p0=[200.0, np.pi / 180.0, 2.2, 256.0],
    bounds=([0, np.pi / 360, 0, 0], [500, np.pi / 90, 2 * np.pi, 512]),
    maxfev=10000
)
A_amp, omega, phi, b = popt
print("探测器零点 b = %.4f（第 j 个探测器偏移量 s_j = (j - b) * d）" % b)

# =========================
# 第一阶段：实际CT投影模型
# =========================
# 第 i 个角度、第 j 个探测器对应射线 L_ij：
#   (x - x0) cos(theta_i) + (y - y0) sin(theta_i) = s_j,  s_j = (j - b) d
# 投影值 Y_ij = gain * ∫_{L_ij} U(x,y) dl
#
# 关于增益 gain 的说明：
# 第一问联合精修时增益吸收了解析正演模型的几何误差（1.3892，偏小）。
# 这里改用离散模型直接率定：以附件1真实模板正演后与附件2整体最小二乘，
# 所得增益与直接估计 max(附件2)/80 = 1.7722 一致（见第三阶段验证）。

N = 256                      # 256 x 256 像素
delta = 100.0 / N            # 像素边长 0.390625 mm
ndet, nang = 512, 180
cos_t, sin_t = np.cos(theta), np.sin(theta)
det_s = (np.arange(ndet) - b) * d

# =========================
# 第二阶段：离散化 y = A u
# =========================

def build_angle_block(k):
    """第 k 个角度下 512 条平行射线的 Siddon 精确追踪。

    返回：
      I : (512, 513) int32   每段射线穿过的像素扁平编号（无效段置 0）
      L : (512, 513) float32 每段在对应像素内的穿过长度（mm，无效段置 0）
    """
    n = np.array([cos_t[k], sin_t[k]])       # 探测器法向
    t = np.array([-sin_t[k], cos_t[k]])      # 射线方向
    p0 = np.array([x0, y0])[None, :] + det_s[:, None] * n[None, :]
    grid = np.arange(N + 1) * delta
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_x = (grid[None, :] - p0[:, 0:1]) / t[0]
        lam_y = (grid[None, :] - p0[:, 1:2]) / t[1]
    lam = np.sort(np.concatenate([lam_x, lam_y], axis=1), axis=1)
    lam_mid = 0.5 * (lam[:, :-1] + lam[:, 1:])
    seg_len = lam[:, 1:] - lam[:, :-1]
    xm = p0[:, 0:1] + lam_mid * t[0]
    ym = p0[:, 1:2] + lam_mid * t[1]
    ix = np.floor(xm / delta).astype(np.int32)
    iy = np.floor(ym / delta).astype(np.int32)
    valid = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N) & np.isfinite(lam_mid)
    I = np.where(valid, iy * N + ix, 0).astype(np.int32)
    L = np.where(valid, seg_len, 0.0).astype(np.float32)
    return np.ascontiguousarray(I), np.ascontiguousarray(L)


t_start = time.time()
blocks = [build_angle_block(k) for k in range(nang)]
nnz = int(sum((L > 0).sum() for _, L in blocks))
print("\n第二阶段：投影矩阵 A 构建完成，用时 %.4f s" % (time.time() - t_start))
print("A 的规模：%d 条射线 × %d 个像素，非零元 %.4f 百万" % (ndet * nang, N * N, nnz / 1e6))


def A_mv(u):
    """正投影 y = A u"""
    y = np.empty(nang * ndet, dtype=np.float64)
    for k, (I, L) in enumerate(blocks):
        y[k * ndet:(k + 1) * ndet] = (L * u[I]).sum(axis=1)
    return y


def AT_mv(y):
    """反投影 u = A^T y"""
    u = np.zeros(N * N, dtype=np.float64)
    for k, (I, L) in enumerate(blocks):
        p = y[k * ndet:(k + 1) * ndet]
        u += np.bincount(I.ravel(), weights=(L * p[:, None]).ravel(),
                         minlength=N * N)
    return u


Aop = LinearOperator((nang * ndet, N * N), matvec=A_mv, rmatvec=AT_mv)

# 增益率定：用真实模板正演，与附件2整体最小二乘
Au_tpl = A_mv(template.ravel().astype(np.float64))
p2_flat = proj2.T.ravel()
gain = np.dot(Au_tpl, p2_flat) / np.dot(Au_tpl, Au_tpl)
res_tpl = p2_flat - gain * Au_tpl
print("\n增益率定：gain = %.4f（直接估计 %.4f）" % (gain, proj2.max() / 80.0))
print("模板投影残差 RMSE = %.4f（数据峰值 %.4f）" % (np.sqrt(np.mean(res_tpl ** 2)), proj2.max()))

# =========================
# 第三阶段：Radon 逆变换重建
# =========================

# ---- 3.1 模板验证：重建附件2，应与附件1一致 ----
t0 = time.time()
u2 = lsqr(Aop, (proj2 / gain).T.ravel(), iter_lim=150, atol=1e-10, btol=1e-10)[0]
U2 = np.clip(u2, 0, None).reshape(N, N)
print("\n第三阶段：模板验证重建用时 %.4f s" % (time.time() - t0))
print("模板重建与附件1的相关系数 = %.4f" % np.corrcoef(U2.ravel(), template.ravel())[0, 1])

lab, nreg = ndimage.label(U2 > 0.5)
sizes = ndimage.sum(np.ones_like(lab), lab, range(1, nreg + 1))
for i in [i + 1 for i, s in enumerate(sizes) if s * delta ** 2 > 20]:
    m = lab == i
    ys, xs = np.where(m)
    print("  模板区域%d：面积 %.4f mm²，质心 (%.4f, %.4f) mm，平均吸收率 %.4f"
          % (i, m.sum() * delta ** 2, (xs.mean() + 0.5) * delta,
             (ys.mean() + 0.5) * delta, U2[m].mean()))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im0 = axes[0].imshow(template, cmap="gray", origin="lower", extent=[0, 100, 0, 100])
axes[0].set_title("附件1：标准模板（真值）")
axes[0].set_xlabel("x / mm"); axes[0].set_ylabel("y / mm")
fig.colorbar(im0, ax=axes[0], label="吸收率")
im1 = axes[1].imshow(U2, cmap="gray", origin="lower", extent=[0, 100, 0, 100])
axes[1].set_title("附件2重建结果（验证）")
axes[1].set_xlabel("x / mm"); axes[1].set_ylabel("y / mm")
fig.colorbar(im1, ax=axes[1], label="吸收率")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "template_validation.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---- 3.2 重建附件3（未知介质） ----
t0 = time.time()
u3 = lsqr(Aop, (proj3 / gain).T.ravel(), iter_lim=150, atol=1e-10, btol=1e-10)[0]
U3 = np.clip(u3, 0, None).reshape(N, N)
print("附件3重建用时 %.4f s，重建值范围 %.4f ~ %.4f" % (time.time() - t0, U3.min(), U3.max()))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
im0 = axes[0].imshow(proj3, aspect="auto", cmap="gray", origin="lower")
axes[0].set_title("附件3：未知介质投影数据（正弦图）")
axes[0].set_xlabel("扫描序号"); axes[0].set_ylabel("探测器编号")
fig.colorbar(im0, ax=axes[0], label="接收信息")
im1 = axes[1].imshow(U3, cmap="gray", origin="lower", extent=[0, 100, 0, 100])
axes[1].set_title("附件3重建图像 $\\hat{U}(x,y)$")
axes[1].set_xlabel("x / mm"); axes[1].set_ylabel("y / mm")
fig.colorbar(im1, ax=axes[1], label="吸收率")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "attachment3_reconstruction.png"), dpi=150, bbox_inches="tight")
plt.close()

# 保存重建吸收率矩阵（第0行对应 y=0，第0列对应 x=0，像素边长 100/256 mm）
# 注：.xls是旧版Excel格式，标准写法需要xlwt库(已停止维护，环境里没有)，
# 这里用openpyxl写出内容为xlsx格式、但按要求命名为problem2.xls，
# Excel能正常打开，只是打开时可能提示"文件格式和扩展名不一致"，点确定即可。
pd.DataFrame(U3).round(4).to_excel(
    os.path.join(base_dir, "problem2.xls"),
    index=False, header=False, engine="openpyxl"
)

# =========================
# 第四阶段：位置、形状与吸收率分析
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
    print("  面积       = %.4f mm²" % rec["面积_mm2"])
    print("  质心位置   = (%.4f, %.4f) mm" % (cx, cy))
    print("  包围盒     = x:[%.4f, %.4f]，y:[%.4f, %.4f] mm"
          % (rec["包围盒x_min_mm"], rec["包围盒x_max_mm"],
             rec["包围盒y_min_mm"], rec["包围盒y_max_mm"]))
    print("  等效椭圆   ：长半轴 %.4f mm，短半轴 %.4f mm，长宽比 %.4f，长轴倾角 %.4f°"
          % (amaj, amin, amaj / amin, ang_deg))
    print("  平均吸收率 = %.4f ± %.4f（中位数 %.4f）"
          % (values.mean(), values.std(), np.median(values)))


print("\n第四阶段：位置、形状与吸收率分析")

# 1) 介质整体（外轮廓）
outer = ndimage.binary_fill_holes(U3 > 0.3)
lab, nreg = ndimage.label(outer)
sizes = ndimage.sum(np.ones_like(lab), lab, range(1, nreg + 1))
outer = lab == (np.argmax(sizes) + 1)
region_report("介质整体（外轮廓）", outer, U3[outer])

# 2) 主体介质（外轮廓内、剔除高吸收区与空洞）
body = outer & (U3 >= 0.2) & (U3 <= 1.2)
region_report("主体介质（本底吸收区）", body, U3[body])

# 3) 高吸收区域
lab_h, nh = ndimage.label(U3 > 1.2)
sizes_h = ndimage.sum(np.ones_like(lab_h), lab_h, range(1, nh + 1))
high_masks = [lab_h == (i + 1) for i, s in enumerate(sizes_h) if s * delta ** 2 > 10]
high_masks.sort(key=lambda m: -np.where(m)[0].mean())   # 按 y 从大到小（从上到下）
for k, m in enumerate(high_masks, 1):
    region_report("高吸收区域%d" % k, m, U3[m])

# 4) 内部空洞
holes = outer & (U3 < 0.2)
lab_o, no = ndimage.label(holes)
sizes_o = ndimage.sum(np.ones_like(lab_o), lab_o, range(1, no + 1))
hole_masks = [lab_o == (i + 1) for i, s in enumerate(sizes_o) if s * delta ** 2 > 10]
hole_masks.sort(key=lambda m: np.where(m)[1].mean())    # 按 x 从小到大（从左到右）
for k, m in enumerate(hole_masks, 1):
    region_report("空洞区域%d" % k, m, U3[m])

pd.DataFrame(records).to_csv(
    os.path.join(base_dir, "区域统计分析.csv"),
    index=False, encoding="utf-8-sig", float_format="%.4f"
)

# 分割结果可视化（显示前对重建图做轻度平滑，抑制迭代噪声对等值线的干扰）
U3s = ndimage.gaussian_filter(U3, sigma=1.2)
seg = np.zeros_like(U3)          # 0=背景, 1=主体介质, 2=高吸收区, 3=空洞
seg[outer] = 1
for m in high_masks:
    seg[m] = 2
for m in hole_masks:
    seg[m] = 3

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
im0 = axes[0].imshow(U3, cmap="gray", origin="lower", extent=[0, 100, 0, 100])
axes[0].contour(np.linspace(delta/2, 100-delta/2, N), np.linspace(delta/2, 100-delta/2, N),
                U3s, levels=[0.3, 1.2], colors=["cyan", "red"], linewidths=1.2)
axes[0].set_title("重建图像与区域边界（青: 外轮廓, 红: 高吸收区）")
axes[0].set_xlabel("x / mm"); axes[0].set_ylabel("y / mm")
fig.colorbar(im0, ax=axes[0], label="吸收率")
from matplotlib.colors import ListedColormap
cmap_seg = ListedColormap(["black", "lightgray", "tomato", "white"])
im1 = axes[1].imshow(seg, cmap=cmap_seg, origin="lower", extent=[0, 100, 0, 100], vmin=0, vmax=3)
axes[1].set_title("区域分割（浅灰=主体介质, 红=高吸收区, 白=空洞）")
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
# 10个指定位置一般不落在像素中心，直接取最近像素会有离散误差。
# 这里同时实现双线性（2x2邻域）与双三次（4x4邻域）插值，
# 通过抽稀交叉验证比较 MAE / RMSE / R^2，以 RMSE 为主判据选出最优方法，
# 再用最优方法计算10个指定位置的最终吸收率。

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
# 以偶数行偶数列的像素为插值节点（2Δ 间距），
# 其余像素作为验证点：其真实值已知，但不参与插值，
# 模拟“该位置没有直接像素值时，两种方法谁能更准确恢复”。
coarse = U3[::2, ::2]
val_mask = np.ones((N, N), dtype=bool)
val_mask[::2, ::2] = False
iy_v, ix_v = np.where(val_mask)
coords_v = np.column_stack([iy_v / 2.0, ix_v / 2.0])   # 抽稀网格下的浮点坐标
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
compare.round(4).to_csv(
    os.path.join(base_dir, "插值方法比较.csv"),
    index=False, encoding="utf-8-sig", float_format="%.4f"
)

# 以 RMSE 为主判据，MAE、R^2 辅助
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
for m in high_masks:
    high_all |= m
hole_all = np.zeros_like(U3, dtype=bool)
for m in hole_masks:
    hole_all |= m


def region_of(x, y):
    ix = int(np.clip(np.floor(x / delta), 0, N - 1))
    iy = int(np.clip(np.floor(y / delta), 0, N - 1))
    if high_all[iy, ix]:
        return "高吸收区"
    if hole_all[iy, ix]:
        return "空洞"
    if outer[iy, ix]:
        return "主体介质"
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
result10.round(4).to_csv(
    os.path.join(base_dir, "十个指定位置吸收率.csv"),
    index=False, encoding="utf-8-sig", float_format="%.4f"
)
print("\n10个指定位置的吸收率（最终采用：%s）：" % best_name)
print(result10.round(4).to_string(index=False))

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
        "%d: %.4f" % (k, val_best[k - 1]), (px, py),
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
print("重建吸收率矩阵：problem2.xls")
print("区域统计结果：区域统计分析.csv")
print("插值方法比较：插值方法比较.csv")
print("10点吸收率：十个指定位置吸收率.csv")