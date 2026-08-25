"""
2017 CUMCM A题 第四问（第一部分）：参数标定的精度与稳定性分析

两部分内容：

一、精度分析（基于实测数据统计精度）
  1) 最终拟合的残差指标：RMSE、相对误差、R^2；
  2) 在最优参数处对正演模型做数值 Jacobian，利用 Fisher 信息
     cov = sigma^2 (J^T J)^{-1} 估计各标定参数的标准差与 95% 置信区间。

二、稳定性检验（噪声敏感性）
  以第一问标定结果 Theta* 为基准真值，用正演模型生成无噪声投影 P0，
  叠加不同等级的高斯白噪声（sigma = level * max(P0)），
  对每个噪声等级重复运行第一问的完整标定流程
  （蒙特卡洛随机搜索 -> 分层局部搜索 -> 180个角度单独修正），
  统计标定参数偏离真值的幅度随噪声等级的变化。

说明：小圆正弦拟合得到的先验 (A, b, omega, phi) 在各次实验中固定，
即考察噪声对“非线性参数反演”阶段的影响。
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"
]
plt.rcParams["axes.unicode_minus"] = False

base_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_dir, "figures")
os.makedirs(save_dir, exist_ok=True)

file_path = os.path.join("A题附件.xls")
projection = pd.read_excel(file_path, sheet_name="附件2", header=None).values

cal = pd.read_csv(os.path.join("2017-A-Q1", "tables", "calibration_result.csv"))

# =========================
# 第〇部分：小圆正弦拟合先验（与第一问一致）
# =========================
observed_scans, observed_centers = [], []
for scan in range(projection.shape[1]):
    y = projection[:, scan]
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


def sine_model(i, A, omega, phi, b):
    return A * np.sin(omega * i + phi) + b


popt, _ = curve_fit(
    sine_model, np.array(observed_scans), np.array(observed_centers),
    p0=[200.0, np.pi / 180.0, 2.2, 256.0],
    bounds=([0, np.pi / 360, 0, 0], [500, np.pi / 90, 2 * np.pi, 512]),
    maxfev=10000
)
A_amp, omega, phi, b = popt
print("小圆正弦拟合先验：A=%.4f, omega=%.5f, phi=%.4f, b=%.4f"
      % (A_amp, omega, phi, b))

# =========================
# 正演模型（与第一问完全相同）
# =========================
ellipse_center = np.array([50.0, 50.0])
ellipse_ax, ellipse_ay = 15.0, 40.0
circle_center = np.array([95.0, 50.0])
circle_radius = 4.0


def forward_projection_at_angles(x0, y0, d, angles):
    detector = (np.arange(512) - b) * d
    angles = np.atleast_1d(angles)
    cos_theta = np.cos(angles)
    sin_theta = np.sin(angles)
    ellipse_s = ((ellipse_center[0] - x0) * cos_theta
                 + (ellipse_center[1] - y0) * sin_theta)
    ellipse_half_width = np.sqrt((ellipse_ax * cos_theta) ** 2
                                 + (ellipse_ay * sin_theta) ** 2)
    ellipse_chord_factor = np.sqrt((ellipse_ax * sin_theta) ** 2
                                   + (ellipse_ay * cos_theta) ** 2)
    u = (detector[:, None] - ellipse_s[None, :]) / ellipse_half_width[None, :]
    ellipse_projection = (2.0 * ellipse_chord_factor[None, :]
                          * np.sqrt(np.maximum(0.0, 1.0 - u ** 2)))
    circle_s = ((circle_center[0] - x0) * cos_theta
                + (circle_center[1] - y0) * sin_theta)
    u = (detector[:, None] - circle_s[None, :]) / circle_radius
    circle_projection = (2.0 * circle_radius
                         * np.sqrt(np.maximum(0.0, 1.0 - u ** 2)))
    return ellipse_projection + circle_projection


def forward_projection(theta5):
    x0, y0, d, theta0, delta_theta = theta5
    return forward_projection_at_angles(
        x0, y0, d, theta0 + np.arange(180) * delta_theta)


# 第一问标定结果作为基准真值
theta_star = np.array([
    cal["x0"][0], cal["y0"][0], cal["d"][0],
    np.deg2rad(cal["theta0_degree"][0]), np.deg2rad(cal["delta_theta_degree"][0])
])
gain_star = cal["final_gain"][0]
angles_star = theta_star[3] + np.arange(180) * theta_star[4]
print("\n基准真值 Theta*：")
print("x0=%.4f, y0=%.4f, d=%.4f, theta0=%.4f°, delta_theta=%.4f°/次, gain=%.4f"
      % (theta_star[0], theta_star[1], theta_star[2],
         np.degrees(theta_star[3]), np.degrees(theta_star[4]), gain_star))

# =========================
# 第一部分：精度分析
# =========================
print("\n" + "=" * 50)
print("第一部分：参数标定的精度分析")
print("=" * 50)

# ---- 1.1 拟合残差指标（实测数据） ----
final_unit = forward_projection_at_angles(
    theta_star[0], theta_star[1], theta_star[2],
    pd.read_csv(os.path.join("2017-A-Q1", "tables", "calibrated_angles.csv"))["angle_degree"].values
    * np.pi / 180.0
)
final_proj = gain_star * final_unit
residual = projection - final_proj
rmse = np.sqrt(np.mean(residual ** 2))
r2_fit = 1 - np.sum(residual ** 2) / np.sum((projection - projection.mean()) ** 2)
print("\n实测数据拟合：RMSE = %.4f（相对峰值 %.2f%%），R^2 = %.6f"
      % (rmse, 100 * rmse / projection.max(), r2_fit))

# ---- 1.2 数值 Jacobian + Fisher 信息 ----
# 参数向量 p = [x0, y0, d, theta0(rad), delta_theta(rad), gain]
p_star = np.array([theta_star[0], theta_star[1], theta_star[2],
                   theta_star[3], theta_star[4], gain_star])
steps = np.array([0.01, 0.01, 1e-4, 1e-5, 1e-6, 1e-3])


def model_flat(p):
    return (p[5] * forward_projection(p[:5])).ravel()


F0 = model_flat(p_star)
y_flat = projection.ravel()
J = np.zeros((y_flat.size, 6))
for j in range(6):
    dp = p_star.copy(); dp[j] += steps[j]
    dm = p_star.copy(); dm[j] -= steps[j]
    J[:, j] = (model_flat(dp) - model_flat(dm)) / (2 * steps[j])

res = y_flat - F0
sigma2 = np.sum(res ** 2) / (y_flat.size - 6)
cov = sigma2 * np.linalg.inv(J.T @ J)
se = np.sqrt(np.diag(cov))

names = ["x0", "y0", "d", "theta0", "delta_theta", "gain"]
units = ["mm", "mm", "mm", "度", "度/次", ""]
est_show = [p_star[0], p_star[1], p_star[2],
            np.degrees(p_star[3]), np.degrees(p_star[4]), p_star[5]]
se_show = [se[0], se[1], se[2], np.degrees(se[3]), np.degrees(se[4]), se[5]]

print("\n参数标准差（Fisher 信息 / 数值 Jacobian）：")
precision_rows = []
for k in range(6):
    rel = 100 * se_show[k] / abs(est_show[k])
    precision_rows.append({
        "参数": names[k], "估计值": est_show[k], "标准差": se_show[k],
        "单位": units[k], "相对精度_%": rel,
        "95%置信下限": est_show[k] - 1.96 * se_show[k],
        "95%置信上限": est_show[k] + 1.96 * se_show[k],
    })
    print("  %-11s = %10.5f ± %.5f %s（相对精度 %.4f%%）"
          % (names[k], est_show[k], se_show[k], units[k], rel))
pd.DataFrame(precision_rows).to_csv(
    os.path.join(base_dir, "精度分析.csv"),
    index=False, encoding="utf-8-sig", float_format="%.6f"
)

# =========================
# 第二部分：稳定性检验（噪声敏感性）
# =========================
print("\n" + "=" * 50)
print("第二部分：稳定性检验")
print("=" * 50)

# 无噪声基准投影（用等间隔5参数模型生成，使 Theta* 为精确真值）
P0 = gain_star * forward_projection(theta_star)
max_p0 = P0.max()


def calibrate(proj, seed):
    """第一问标定流程（第四至第八阶段），输入投影数据，输出标定结果。"""
    rng = np.random.RandomState(seed)

    def fit_gain(sim):
        return np.sum(proj * sim) / np.sum(sim * sim)

    def proj_error(sim):
        g = fit_gain(sim)
        return np.mean((proj - g * sim) ** 2), g

    d_min, d_max = 0.2680, 0.2900
    direction_min, direction_max = -0.4000, 0.1000
    delta_theta_sigma = np.deg2rad(0.0300)
    theta0_sigma = np.deg2rad(1.0000)

    # ---- 蒙特卡洛随机搜索 ----
    best_theta, best_error, best_gain = None, np.inf, None
    for _ in range(2000):
        d = rng.uniform(d_min, d_max)
        rotation_distance = A_amp * d
        direction = rng.uniform(direction_min, direction_max)
        x0 = circle_center[0] - rotation_distance * np.cos(direction)
        y0 = circle_center[1] - rotation_distance * np.sin(direction)
        theta0 = phi + omega - np.pi / 2 + direction + rng.normal(0.0, theta0_sigma)
        delta_theta = omega + rng.normal(0.0, delta_theta_sigma)
        th = np.array([x0, y0, d, theta0, delta_theta])
        err, g = proj_error(forward_projection(th))
        if err < best_error:
            best_theta, best_error, best_gain = th, err, g

    # ---- 分层局部搜索 ----
    def center_direction(th):
        return np.arctan2(circle_center[1] - th[1], circle_center[0] - th[0])

    current, current_err, current_gain = best_theta, best_error, best_gain
    for count, d_hw, dir_hw, t0_hw, dt_hw in [
        (2000, 0.0040, 0.0150, np.deg2rad(0.5000), np.deg2rad(0.0300)),
        (2000, 0.0010, 0.0040, np.deg2rad(0.1200), np.deg2rad(0.0080)),
    ]:
        direction0 = center_direction(current)
        for _ in range(count):
            d = rng.uniform(max(d_min, current[2] - d_hw),
                            min(d_max, current[2] + d_hw))
            direction = rng.uniform(direction0 - dir_hw, direction0 + dir_hw)
            rotation_distance = A_amp * d
            x0 = circle_center[0] - rotation_distance * np.cos(direction)
            y0 = circle_center[1] - rotation_distance * np.sin(direction)
            theta0 = rng.uniform(current[3] - t0_hw, current[3] + t0_hw)
            delta_theta = rng.uniform(current[4] - dt_hw, current[4] + dt_hw)
            th = np.array([x0, y0, d, theta0, delta_theta])
            err, g = proj_error(forward_projection(th))
            if err < current_err:
                current, current_err, current_gain = th, err, g

    refined_theta, refined_gain = current, current_gain

    # ---- 180个角度单独修正 ----
    base_angles = refined_theta[3] + np.arange(180) * refined_theta[4]
    offsets = np.linspace(-np.deg2rad(0.2), np.deg2rad(0.2), 81)
    corrected = np.zeros(180)
    for i in range(180):
        cand = base_angles[i] + offsets
        cproj = forward_projection_at_angles(
            refined_theta[0], refined_theta[1], refined_theta[2], cand)
        errs = np.mean((proj[:, i:i + 1] - refined_gain * cproj) ** 2, axis=0)
        corrected[i] = cand[np.argmin(errs)]
    final_unit = forward_projection_at_angles(
        refined_theta[0], refined_theta[1], refined_theta[2], corrected)
    final_gain = np.sum(proj * final_unit) / np.sum(final_unit ** 2)
    final_rmse = np.sqrt(np.mean((proj - final_gain * final_unit) ** 2))
    return refined_theta, final_gain, corrected, final_rmse


noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10]   # 占信号峰值的比例
n_rep = 5
rows = []
t_all = time.time()
for li, level in enumerate(noise_levels):
    sigma_n = level * max_p0
    for rep in range(n_rep):
        rng_data = np.random.RandomState(1000 * li + rep)
        proj_noisy = P0 + rng_data.normal(0.0, sigma_n, P0.shape)
        th, g, ang_c, r_final = calibrate(proj_noisy, seed=100 * li + rep)
        ang_dev = np.degrees(ang_c - angles_star)
        rows.append({
            "噪声等级_%": 100 * level, "重复": rep + 1,
            "x0": th[0], "y0": th[1], "d": th[2],
            "theta0_度": np.degrees(th[3]), "delta_theta_度": np.degrees(th[4]),
            "gain": g, "拟合RMSE": r_final,
            "偏差_x0_mm": th[0] - theta_star[0],
            "偏差_y0_mm": th[1] - theta_star[1],
            "偏差_d_mm": th[2] - theta_star[2],
            "偏差_theta0_度": np.degrees(th[3] - theta_star[3]),
            "偏差_delta_theta_度": np.degrees(th[4] - theta_star[4]),
            "偏差_gain": g - gain_star,
            "角度平均绝对偏差_度": np.mean(np.abs(ang_dev)),
            "角度最大偏差_度": np.max(np.abs(ang_dev)),
        })
        print("噪声 %4.1f%% 第%d次：dx0=%+.4f, dy0=%+.4f, dd=%+.5f, "
              "dθ0=%+.4f°, dΔθ=%+.5f°, RMSE=%.3f"
              % (100 * level, rep + 1, rows[-1]["偏差_x0_mm"],
                 rows[-1]["偏差_y0_mm"], rows[-1]["偏差_d_mm"],
                 rows[-1]["偏差_theta0_度"], rows[-1]["偏差_delta_theta_度"],
                 r_final))
print("稳定性检验总用时 %.1f s" % (time.time() - t_all))

stab = pd.DataFrame(rows)
stab.to_csv(os.path.join(base_dir, "稳定性检验.csv"),
            index=False, encoding="utf-8-sig", float_format="%.6f")

# ---- 按噪声等级统计 ----
dev_cols = ["偏差_x0_mm", "偏差_y0_mm", "偏差_d_mm",
            "偏差_theta0_度", "偏差_delta_theta_度", "偏差_gain",
            "角度平均绝对偏差_度", "角度最大偏差_度", "拟合RMSE"]
stat_rows = []
for level in noise_levels:
    sub = stab[stab["噪声等级_%"] == 100 * level]
    row = {"噪声等级_%": 100 * level}
    for c in dev_cols:
        row[c + "_均值"] = sub[c].mean()
        row[c + "_绝对值最大"] = sub[c].abs().max()
    stat_rows.append(row)
stat = pd.DataFrame(stat_rows)
stat.to_csv(os.path.join(base_dir, "稳定性统计.csv"),
            index=False, encoding="utf-8-sig", float_format="%.6f")
print("\n稳定性统计（偏差绝对值最大）：")
show_cols = ["噪声等级_%"] + [c + "_绝对值最大" for c in dev_cols[:6]] \
            + ["角度最大偏差_度_绝对值最大"]
print(stat[show_cols].to_string(index=False))

# =========================
# 绘图
# =========================

# 图1：含噪投影数据示例
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, level in zip(axes, [0.0, 0.02, 0.10]):
    rng_data = np.random.RandomState(1000 * noise_levels.index(level))
    pn = P0 + rng_data.normal(0.0, level * max_p0, P0.shape)
    im = ax.imshow(pn, aspect="auto", cmap="gray", origin="lower")
    ax.set_title("噪声等级 %.0f%%" % (100 * level))
    ax.set_xlabel("扫描序号"); ax.set_ylabel("探测器编号")
    fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "noisy_sinograms.png"), dpi=150, bbox_inches="tight")
plt.close()

# 图2：五个参数偏差随噪声等级变化（误差棒）
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
items = [("偏差_x0_mm", "x0 偏差 / mm"), ("偏差_y0_mm", "y0 偏差 / mm"),
         ("偏差_d_mm", "d 偏差 / mm"), ("偏差_theta0_度", "θ0 偏差 / 度"),
         ("偏差_delta_theta_度", "Δθ 偏差 / 度"), ("偏差_gain", "增益偏差")]
for ax, (c, label) in zip(axes.ravel(), items):
    means = [stab[stab["噪声等级_%"] == 100 * l][c].mean() for l in noise_levels]
    stds = [stab[stab["噪声等级_%"] == 100 * l][c].std() for l in noise_levels]
    maxs = [stab[stab["噪声等级_%"] == 100 * l][c].abs().max() for l in noise_levels]
    xs = [100 * l for l in noise_levels]
    ax.errorbar(xs, means, yerr=stds, fmt="o-", capsize=4, label="均值±标准差")
    ax.plot(xs, maxs, "s--", color="red", label="绝对值最大")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("噪声等级 / %"); ax.set_ylabel(label)
    ax.set_title(label + " 随噪声等级变化")
    ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "param_deviation_vs_noise.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# 图3：角度偏差与拟合RMSE随噪声等级变化
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
xs = [100 * l for l in noise_levels]
for c, label, ax in [
    ("角度平均绝对偏差_度", "180个角度的平均绝对偏差 / 度", axes[0]),
    ("拟合RMSE", "最终拟合 RMSE", axes[1]),
]:
    means = [stab[stab["噪声等级_%"] == 100 * l][c].mean() for l in noise_levels]
    maxs = [stab[stab["噪声等级_%"] == 100 * l][c].max() for l in noise_levels]
    ax.plot(xs, means, "o-", label="均值")
    ax.plot(xs, maxs, "s--", color="red", label="最大")
    ax.set_xlabel("噪声等级 / %"); ax.set_ylabel(label)
    ax.set_title(label + " 随噪声等级变化")
    ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "angle_rmse_vs_noise.png"),
            dpi=150, bbox_inches="tight")
plt.close()

print("\n全部完成。")
print("精度分析：精度分析.csv")
print("稳定性明细：稳定性检验.csv")
print("稳定性统计：稳定性统计.csv")
print("图片已保存到：", save_dir)