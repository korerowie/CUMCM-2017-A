import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from scipy.optimize import curve_fit

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"
]
plt.rcParams["axes.unicode_minus"] = False

base_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(base_dir, "figures")
os.makedirs(save_dir, exist_ok=True)

file_path = os.path.join("A题附件.xls")
template = pd.read_excel(
    file_path,
    sheet_name="附件1",
    header=None
)
projection = pd.read_excel(
    file_path,
    sheet_name="附件2",
    header=None
)
template = template.values
projection = projection.values

print("附件1形状：", template.shape)
print("附件2形状：", projection.shape)
print("\n附件1数值范围：")
print("最小值：", template.min())
print("最大值：", template.max())


# =========================
# 第一阶段：提取小圆独立投影区域
# =========================

# 连通区域识别
binary_image = template > 0
labeled_image, num_features = ndimage.label(binary_image)
print("\n第一阶段：提取小圆独立投影区域")
print("\n识别出的连通区域数量：", num_features)
for i in range(1, num_features + 1):
    # 获取第 i 个区域的所有坐标
    y_coords, x_coords = np.where(labeled_image == i)
    # 面积
    area = len(x_coords)
    # 几何中心
    center_x = x_coords.mean()
    center_y = y_coords.mean()
    print(f"\n区域 {i}")
    print("面积：", area)
    print("中心：", (center_x, center_y))

plt.figure(figsize=(6, 6))
plt.imshow(
    template,
    cmap="gray",
    origin="lower"
)
plt.colorbar(label="吸收率")
plt.title("附件1：CT标准模板")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(os.path.join(save_dir, "template.png"), dpi=150, bbox_inches="tight")
plt.close()

# 绘制部分投影曲线
scan_indices = [0, 29, 59, 89, 119, 149, 179]
plt.figure(figsize=(10, 6))
for i in scan_indices:
    plt.plot(
        np.arange(512),
        projection[:, i],
        label=f"第{i + 1}次扫描"
    )
plt.xlabel("探测器编号")
plt.ylabel("接收信息")
plt.title("附件2：不同扫描角度下的投影曲线")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "projection_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

# 计算投影的一阶差分
difference = np.diff(projection, axis=0)
print("差分矩阵形状：", difference.shape)

# 投影一阶差分热力图
plt.figure(figsize=(10, 7))
plt.imshow(
    np.abs(difference).T,
    aspect="auto",
    origin="lower"
)
plt.colorbar(label="一阶差分绝对值")
plt.xlabel("探测器编号")
plt.ylabel("扫描序号")
plt.title("180次扫描投影的一阶变化热力图")
plt.savefig(os.path.join(save_dir, "difference_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

# 当小圆与大椭圆在探测器上分离时，投影中会出现一个独立的窄非零区间。
# 这里先只使用这些不受大椭圆重叠影响的数据点。
observed_scans = []
observed_centers = []
observed_widths = []
for scan in range(projection.shape[1]):
    y = projection[:, scan]
    mask = y > 0
    # 求连续非零区间
    change = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0] - 1
    intervals = list(zip(starts, ends))
    # 小圆独立投影的宽度约为29个探测器单元
    narrow_intervals = [
        (left, right)
        for left, right in intervals
        if 10 <= right - left + 1 <= 45
    ]
    if len(narrow_intervals) != 1:
        continue
    left, right = narrow_intervals[0]
    detector_index = np.arange(left, right + 1)
    weight = y[left:right + 1]
    # 按方案使用加权质心表示小圆投影位置
    center = np.sum(detector_index * weight) / np.sum(weight)
    observed_scans.append(scan + 1)
    observed_centers.append(center)
    observed_widths.append(right - left + 1)
observed_scans = np.array(observed_scans)
observed_centers = np.array(observed_centers)
observed_widths = np.array(observed_widths)
print("\n识别到小圆独立投影区间数量：", len(observed_scans))
print("小圆独立投影宽度中位数：{:.4f}".format(np.median(observed_widths)))


# =========================
# 第二阶段：小圆轨迹正弦拟合
# =========================

def sine_model(i, A, omega, phi, b):
    return A * np.sin(omega * i + phi) + b
# 根据数据先验给出初值；扫描180次约对应180度
p0 = [200.0, np.pi / 180.0, 2.2, 256.0]
popt, _ = curve_fit(
    sine_model,
    observed_scans,
    observed_centers,
    p0=p0,
    bounds=(
        [0.0, np.pi / 360.0, 0.0, 0.0],
        [500.0, np.pi / 90.0, 2.0 * np.pi, 512.0]
    ),
    maxfev=10000
)
A, omega, phi, b = popt
all_scans = np.arange(1, projection.shape[1] + 1)
fitted_centers = sine_model(all_scans, A, omega, phi, b)
fitted_observed = sine_model(observed_scans, A, omega, phi, b)
residual = observed_centers - fitted_observed
rmse = np.sqrt(np.mean(residual ** 2))
mae = np.mean(np.abs(residual))
r2 = 1 - np.sum(residual ** 2) / np.sum(
    (observed_centers - np.mean(observed_centers)) ** 2
)
print("\n第二阶段：小圆轨迹正弦拟合")
print("\n小圆正弦拟合结果：")
print("A     = {:.4f}".format(A))
print("omega = {:.4f} rad/次".format(omega))
print("      = {:.4f} 度/次".format(np.degrees(omega)))
print("phi   = {:.4f} rad".format(phi))
print("b     = {:.4f}".format(b))
print("RMSE  = {:.4f}".format(rmse))
print("MAE   = {:.4f}".format(mae))
print("R^2   = {:.4f}".format(r2))

# 在正弦图上检查轨迹
plt.figure(figsize=(10, 6))
plt.imshow(
    projection,
    aspect="auto",
    cmap="gray",
    origin="lower",
    extent=[1, 180, 0, 511]
)
plt.colorbar(label="接收信息")
plt.scatter(
    observed_scans,
    observed_centers,
    s=18,
    c="cyan",
    label="小圆独立区域质心"
)
plt.plot(
    all_scans,
    fitted_centers,
    "r-",
    linewidth=2,
    label="正弦拟合轨迹"
)
plt.xlabel("扫描序号")
plt.ylabel("探测器编号")
plt.title("小圆投影轨迹的正弦规律")
plt.legend()
plt.savefig(
    os.path.join(save_dir, "small_circle_sinogram.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# 绘制正弦拟合结果
plt.figure(figsize=(9, 5))
plt.scatter(
    observed_scans,
    observed_centers,
    s=25,
    label="小圆观测质心"
)
plt.plot(
    all_scans,
    fitted_centers,
    "r-",
    linewidth=2,
    label="s(i)={:.4f}sin({:.4f}i+{:.4f})+{:.4f}".format(
        A, omega, phi, b
    )
)
plt.xlabel("扫描序号 i")
plt.ylabel("小圆投影位置 s(i)")
plt.title("小圆投影位置正弦拟合")
plt.grid(True)
plt.legend()
plt.savefig(
    os.path.join(save_dir, "small_circle_sine_fit.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# 绘制拟合残差
plt.figure(figsize=(9, 4))
plt.stem(
    observed_scans,
    residual,
    basefmt=" "
)
plt.axhline(0, color="red", linewidth=1)
plt.xlabel("扫描序号")
plt.ylabel("观测值 - 拟合值")
plt.title("小圆正弦拟合残差")
plt.grid(True)
plt.savefig(
    os.path.join(save_dir, "small_circle_fit_residual.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# 保存轨迹数据，供后续蒙特卡洛反演使用
result = pd.DataFrame({
    "scan_index": all_scans,
    "sine_center": fitted_centers
})
result.to_csv(
    os.path.join(base_dir, "small_circle_sine_fit.csv"),
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)
print("\n轨迹数据已保存：small_circle_sine_fit.csv")
print("图片已保存到：", save_dir)



# =========================
# 第三阶段：建立低维参数空间
# =========================

# 参数为 Theta=(x0,y0,d,theta0,delta_theta)
# 其中小圆拟合结果给出约束：
# sqrt((95-x0)^2+(50-y0)^2) = A*d

ellipse_center = np.array([50.0, 50.0])
ellipse_ax = 15.0
ellipse_ay = 40.0
circle_center = np.array([95.0, 50.0])
circle_radius = 4.0
d_min, d_max = 0.2680, 0.2900
direction_min, direction_max = -0.4000, 0.1000
delta_theta_sigma = np.deg2rad(0.0300)
theta0_sigma = np.deg2rad(1.0000)
print("\n第三阶段：低维参数空间")
print("参数：Theta=(x0, y0, d, theta0, delta_theta)")
print("d范围：[{:.4f}, {:.4f}]".format(d_min, d_max))
print("平均步长初值：{:.4f} 度/次".format(np.degrees(omega)))
print("投影增益：由完整投影最小二乘估计")

# CT正演函数
# theta = (x0, y0, d, theta0, delta_theta)
# 返回 512 x 180 的模拟投影矩阵。
def forward_projection_at_angles(x0, y0, d, angles):
    detector = (np.arange(512) - b) * d
    angles = np.atleast_1d(angles)
    cos_theta = np.cos(angles)
    sin_theta = np.sin(angles)
    # 大椭圆投影
    ellipse_s = (
        (ellipse_center[0] - x0) * cos_theta
        + (ellipse_center[1] - y0) * sin_theta
    )
    ellipse_half_width = np.sqrt(
        (ellipse_ax * cos_theta) ** 2
        + (ellipse_ay * sin_theta) ** 2
    )
    ellipse_chord_factor = np.sqrt(
        (ellipse_ax * sin_theta) ** 2
        + (ellipse_ay * cos_theta) ** 2
    )
    u = (
        detector[:, None] - ellipse_s[None, :]
    ) / ellipse_half_width[None, :]
    ellipse_projection = (
        2.0
        * ellipse_chord_factor[None, :]
        * np.sqrt(np.maximum(0.0, 1.0 - u ** 2))
    )
    # 小圆投影
    circle_s = (
        (circle_center[0] - x0) * cos_theta
        + (circle_center[1] - y0) * sin_theta
    )
    u = (
        detector[:, None] - circle_s[None, :]
    ) / circle_radius
    circle_projection = (
        2.0
        * circle_radius
        * np.sqrt(np.maximum(0.0, 1.0 - u ** 2))
    )
    return ellipse_projection + circle_projection
def forward_projection(theta):
    x0, y0, d, theta0, delta_theta = theta
    angles = theta0 + np.arange(180) * delta_theta
    return forward_projection_at_angles(x0, y0, d, angles)
def fit_projection_gain(simulated):
    return np.sum(projection * simulated) / np.sum(simulated * simulated)
def projection_error(simulated):
    gain = fit_projection_gain(simulated)
    mse = np.mean((projection - gain * simulated) ** 2)
    return mse, gain


# =========================
# 第四阶段：蒙特卡洛随机搜索
# =========================

np.random.seed(2026)
sample_count = 2000
sample_parameters = np.zeros((sample_count, 5))
sample_errors = np.zeros(sample_count)
sample_gains = np.zeros(sample_count)
print("\n第四阶段：蒙特卡洛随机搜索")
print("随机采样次数：", sample_count)
for k in range(sample_count):
    # 先采样d，再利用振幅约束确定旋转中心到小圆的距离
    d = np.random.uniform(d_min, d_max)
    rotation_distance = A * d
    # direction是小圆圆心相对旋转中心的方向角
    direction = np.random.uniform(direction_min, direction_max)
    x0 = circle_center[0] - rotation_distance * np.cos(direction)
    y0 = circle_center[1] - rotation_distance * np.sin(direction)
    # 由正弦拟合相位得到theta0的估计值，再进行小范围扰动
    theta0 = (
        phi + omega - np.pi / 2 + direction
        + np.random.normal(0.0, theta0_sigma)
    )
    delta_theta = omega + np.random.normal(0.0, delta_theta_sigma)
    theta = np.array([x0, y0, d, theta0, delta_theta])
    simulated = forward_projection(theta)
    # 用完整投影最小二乘估计增益，再计算均方误差
    mse, gain = projection_error(simulated)
    sample_parameters[k] = theta
    sample_errors[k] = mse
    sample_gains[k] = gain
best_index = np.argmin(sample_errors)
best_theta = sample_parameters[best_index]
best_error = sample_errors[best_index]
best_gain = sample_gains[best_index]
best_projection = best_gain * forward_projection(best_theta)
print("\n当前蒙特卡洛最优参数：")
print("x0          = {:.4f} mm".format(best_theta[0]))
print("y0          = {:.4f} mm".format(best_theta[1]))
print("d           = {:.4f} mm".format(best_theta[2]))
print("theta0      = {:.4f} 度".format(np.degrees(best_theta[3])))
print("delta_theta = {:.4f} 度/次".format(np.degrees(best_theta[4])))
print("增益        = {:.4f}".format(best_gain))
print("MSE         = {:.4f}".format(best_error))
print("RMSE        = {:.4f}".format(np.sqrt(best_error)))

# 保存蒙特卡洛样本
monte_carlo_result = pd.DataFrame({
    "x0": sample_parameters[:, 0],
    "y0": sample_parameters[:, 1],
    "d": sample_parameters[:, 2],
    "theta0_degree": np.degrees(sample_parameters[:, 3]),
    "delta_theta_degree": np.degrees(sample_parameters[:, 4]),
    "gain": sample_gains,
    "mse": sample_errors
})
monte_carlo_result.to_csv(
    os.path.join(base_dir, "monte_carlo_samples.csv"),
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)

# 绘制参数采样结果
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    sample_parameters[:, 0],
    sample_parameters[:, 1],
    c=np.log10(sample_errors),
    s=12,
    cmap="viridis"
)
plt.scatter(
    best_theta[0],
    best_theta[1],
    c="red",
    marker="*",
    s=180,
    label="当前最优样本"
)
plt.colorbar(scatter, label="log10(MSE)")
plt.xlabel("x0 / mm")
plt.ylabel("y0 / mm")
plt.title("蒙特卡洛参数采样与完整投影误差")
plt.legend()
plt.grid(True)
plt.savefig(
    os.path.join(save_dir, "monte_carlo_parameter_space.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# 比较当前最优正演结果
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
im0 = axes[0].imshow(
    projection,
    aspect="auto",
    cmap="gray",
    origin="lower"
)
axes[0].set_title("附件2实测投影")
axes[0].set_xlabel("扫描序号")
axes[0].set_ylabel("探测器编号")
fig.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(
    best_projection,
    aspect="auto",
    cmap="gray",
    origin="lower"
)
axes[1].set_title("当前最优正演投影")
axes[1].set_xlabel("扫描序号")
axes[1].set_ylabel("探测器编号")
fig.colorbar(im1, ax=axes[1])
difference = projection - best_projection
im2 = axes[2].imshow(
    difference,
    aspect="auto",
    cmap="seismic",
    origin="lower"
)
axes[2].set_title("实测值 - 正演值")
axes[2].set_xlabel("扫描序号")
axes[2].set_ylabel("探测器编号")
fig.colorbar(im2, ax=axes[2])
plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "monte_carlo_best_projection.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.close()
print("\n蒙特卡洛样本已保存：monte_carlo_samples.csv")
print("第四阶段图片已保存到：", save_dir)


# =========================
# 第五阶段：用完整投影数据评价参数
# =========================

error_ranking = monte_carlo_result.copy()
error_ranking["rmse"] = np.sqrt(error_ranking["mse"])
error_ranking = error_ranking.sort_values("mse").reset_index(drop=True)
error_ranking.insert(0, "rank", np.arange(1, len(error_ranking) + 1))
error_ranking.to_csv(
    os.path.join(base_dir, "full_projection_error_ranking.csv"),
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)
elite_count = min(50, len(error_ranking))
elite_errors = error_ranking["mse"].iloc[:elite_count].values
print("\n第五阶段：完整投影误差评价")
print("完整投影数据点数：", projection.size)
print("最优MSE：{:.4f}".format(error_ranking["mse"].iloc[0]))
print("前{}个样本平均MSE：{:.4f}".format(elite_count, np.mean(elite_errors)))
print("误差排序已保存：full_projection_error_ranking.csv")


# =========================
# 第六阶段：分层缩小搜索范围
# =========================

def center_direction(theta):
    return np.arctan2(
        circle_center[1] - theta[1],
        circle_center[0] - theta[0]
    )
def local_random_search(center_theta, sample_count, d_half_width,
                        direction_half_width, theta0_half_width,
                        delta_theta_half_width):
    direction0 = center_direction(center_theta)
    parameters = np.zeros((sample_count, 5))
    errors = np.zeros(sample_count)
    gains = np.zeros(sample_count)

    for k in range(sample_count):
        d = np.random.uniform(
            max(d_min, center_theta[2] - d_half_width),
            min(d_max, center_theta[2] + d_half_width)
        )
        direction = np.random.uniform(
            direction0 - direction_half_width,
            direction0 + direction_half_width
        )
        rotation_distance = A * d

        x0 = circle_center[0] - rotation_distance * np.cos(direction)
        y0 = circle_center[1] - rotation_distance * np.sin(direction)
        theta0 = np.random.uniform(
            center_theta[3] - theta0_half_width,
            center_theta[3] + theta0_half_width
        )
        delta_theta = np.random.uniform(
            center_theta[4] - delta_theta_half_width,
            center_theta[4] + delta_theta_half_width
        )

        parameters[k] = np.array([x0, y0, d, theta0, delta_theta])
        simulated = forward_projection(parameters[k])
        errors[k], gains[k] = projection_error(simulated)

    return parameters, errors, gains
layered_parameters = []
layered_errors = []
layered_gains = []
layered_rounds = []
search_history = [best_error]
current_theta = best_theta.copy()
current_error = best_error
current_gain = best_gain
round_settings = [
    (1, 2000, 0.0040, 0.0150, np.deg2rad(0.5000), np.deg2rad(0.0300)),
    (2, 2000, 0.0010, 0.0040, np.deg2rad(0.1200), np.deg2rad(0.0080))
]
print("\n第六阶段：分层缩小搜索范围")
for round_index, count, d_width, direction_width, theta0_width, delta_width in round_settings:
    parameters, errors, gains = local_random_search(
        current_theta,
        count,
        d_width,
        direction_width,
        theta0_width,
        delta_width
    )
    round_best_index = np.argmin(errors)
    round_best_error = errors[round_best_index]
    layered_parameters.append(parameters)
    layered_errors.append(errors)
    layered_gains.append(gains)
    layered_rounds.extend([round_index] * count)
    if round_best_error < current_error:
        current_theta = parameters[round_best_index]
        current_error = round_best_error
        current_gain = gains[round_best_index]
    search_history.append(current_error)
    print("第{}轮局部搜索最优MSE：{:.4f}".format(
        round_index, round_best_error
    ))
    print("第{}轮结束后当前MSE：{:.4f}".format(
        round_index, current_error
    ))
layered_parameters = np.vstack(layered_parameters)
layered_errors = np.concatenate(layered_errors)
layered_gains = np.concatenate(layered_gains)
refined_theta = current_theta
refined_error = current_error
refined_gain = current_gain
refined_projection = refined_gain * forward_projection(refined_theta)

print("\n分层搜索后的参数：")
print("x0          = {:.4f} mm".format(refined_theta[0]))
print("y0          = {:.4f} mm".format(refined_theta[1]))
print("d           = {:.4f} mm".format(refined_theta[2]))
print("theta0      = {:.4f} 度".format(np.degrees(refined_theta[3])))
print("delta_theta = {:.4f} 度/次".format(np.degrees(refined_theta[4])))
print("增益        = {:.4f}".format(refined_gain))
print("MSE         = {:.4f}".format(refined_error))
print("RMSE        = {:.4f}".format(np.sqrt(refined_error)))

layered_result = pd.DataFrame({
    "round": layered_rounds,
    "x0": layered_parameters[:, 0],
    "y0": layered_parameters[:, 1],
    "d": layered_parameters[:, 2],
    "theta0_degree": np.degrees(layered_parameters[:, 3]),
    "delta_theta_degree": np.degrees(layered_parameters[:, 4]),
    "gain": layered_gains,
    "mse": layered_errors
})
layered_result.to_csv(
    os.path.join(base_dir, "layered_search_samples.csv"),
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
scatter = axes[0].scatter(
    layered_parameters[:, 0],
    layered_parameters[:, 1],
    c=np.log10(layered_errors),
    s=12,
    cmap="viridis"
)
axes[0].scatter(
    refined_theta[0],
    refined_theta[1],
    c="red",
    marker="*",
    s=180,
    label="分层搜索最优"
)
axes[0].set_xlabel("x0 / mm")
axes[0].set_ylabel("y0 / mm")
axes[0].set_title("分层局部采样结果")
axes[0].legend()
axes[0].grid(True)
fig.colorbar(scatter, ax=axes[0], label="log10(MSE)")
axes[1].plot(
    np.arange(len(search_history)),
    search_history,
    "o-"
)
axes[1].set_xticks(np.arange(len(search_history)))
axes[1].set_xticklabels(["初始", "第1轮", "第2轮"])
axes[1].set_xlabel("搜索阶段")
axes[1].set_ylabel("MSE")
axes[1].set_title("分层搜索误差变化")
axes[1].grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "layered_search.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.close()


# =========================
# 第七阶段：180个角度单独修正
# =========================

base_angles = refined_theta[3] + np.arange(180) * refined_theta[4]
angle_correction_width = np.deg2rad(0.2000)
angle_grid_count = 81
angle_offsets = np.linspace(
    -angle_correction_width,
    angle_correction_width,
    angle_grid_count
)
corrected_angles = np.zeros(180)
angle_corrections = np.zeros(180)
angle_mse = np.zeros(180)
print("\n第七阶段：180个角度单独修正")
print("每个角度搜索范围：±{:.4f} 度".format(
    np.degrees(angle_correction_width)
))
print("每个角度网格数：", angle_grid_count)
for i in range(180):
    candidate_angles = base_angles[i] + angle_offsets
    candidate_projections = forward_projection_at_angles(
        refined_theta[0],
        refined_theta[1],
        refined_theta[2],
        candidate_angles
    )
    errors = np.mean(
        (projection[:, i:i + 1] - refined_gain * candidate_projections) ** 2,
        axis=0
    )

    best_angle_index = np.argmin(errors)
    corrected_angles[i] = candidate_angles[best_angle_index]
    angle_corrections[i] = angle_offsets[best_angle_index]
    angle_mse[i] = errors[best_angle_index]

final_unit_projection = forward_projection_at_angles(
    refined_theta[0],
    refined_theta[1],
    refined_theta[2],
    corrected_angles
)
final_gain = fit_projection_gain(final_unit_projection)
final_projection = final_gain * final_unit_projection
final_error = np.mean((projection - final_projection) ** 2)

print("修正前增益：{:.4f}".format(refined_gain))
print("修正后增益：{:.4f}".format(final_gain))
print("修正前MSE：{:.4f}".format(refined_error))
print("修正后MSE：{:.4f}".format(final_error))
print("修正后RMSE：{:.4f}".format(np.sqrt(final_error)))
print("角度修正量范围：[{:.4f}, {:.4f}] 度".format(
    np.degrees(angle_corrections.min()),
    np.degrees(angle_corrections.max())
))

angle_result = pd.DataFrame({
    "scan_index": np.arange(1, 181),
    "base_angle_degree": np.degrees(base_angles),
    "correction_degree": np.degrees(angle_corrections),
    "angle_degree": np.degrees(corrected_angles),
    "single_scan_mse": angle_mse
})
angle_result.to_csv(
    os.path.join(base_dir, "calibrated_angles.csv"),
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)
calibration_result = pd.DataFrame({
    "x0": [refined_theta[0]],
    "y0": [refined_theta[1]],
    "d": [refined_theta[2]],
    "theta0_degree": [np.degrees(refined_theta[3])],
    "delta_theta_degree": [np.degrees(refined_theta[4])],
    "initial_gain": [best_gain],
    "refined_gain": [refined_gain],
    "final_gain": [final_gain],
    "initial_mse": [best_error],
    "refined_mse": [refined_error],
    "final_mse": [final_error],
    "final_rmse": [np.sqrt(final_error)]
})
calibration_result.to_csv(
    os.path.join(base_dir, "calibration_result.csv"),
    index=False,
    encoding="utf-8-sig",
    float_format="%.4f"
)
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
axes[0].plot(
    np.arange(1, 181),
    np.degrees(base_angles),
    label="等间隔初始角度"
)
axes[0].plot(
    np.arange(1, 181),
    np.degrees(corrected_angles),
    label="单独修正角度"
)
axes[0].set_ylabel("角度 / 度")
axes[0].set_title("180个X射线方向")
axes[0].legend()
axes[0].grid(True)
axes[1].plot(
    np.arange(1, 181),
    np.degrees(angle_corrections),
    color="darkred"
)
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set_xlabel("扫描序号")
axes[1].set_ylabel("修正量 / 度")
axes[1].set_title("每个扫描角度的局部修正量")
axes[1].grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "angle_correction.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.close()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
im0 = axes[0].imshow(
    projection,
    aspect="auto",
    cmap="gray",
    origin="lower"
)
axes[0].set_title("附件2实测投影")
axes[0].set_xlabel("扫描序号")
axes[0].set_ylabel("探测器编号")
fig.colorbar(im0, ax=axes[0])
im1 = axes[1].imshow(
    final_projection,
    aspect="auto",
    cmap="gray",
    origin="lower"
)
axes[1].set_title("最终正演投影")
axes[1].set_xlabel("扫描序号")
axes[1].set_ylabel("探测器编号")
fig.colorbar(im1, ax=axes[1])
final_difference = projection - final_projection
im2 = axes[2].imshow(
    final_difference,
    aspect="auto",
    cmap="seismic",
    origin="lower"
)
axes[2].set_title("实测值 - 最终正演值")
axes[2].set_xlabel("扫描序号")
axes[2].set_ylabel("探测器编号")
fig.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig(
    os.path.join(save_dir, "final_projection_comparison.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("\n角度结果已保存：calibrated_angles.csv")
print("标定结果已保存：calibration_result.csv")
print("后续阶段图片已保存到：", save_dir)