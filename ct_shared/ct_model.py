# -*- coding: utf-8 -*-
"""
CT 共享模型模块（2017 CUMCM A题 第二、三问共用）

封装一次性的耗时部分：
  第一阶段：读取第一问标定参数 (x0, y0, d, theta_1..theta_180)，
            并用小圆正弦拟合补算探测器零点 b；
  第二阶段：Siddon 射线追踪构建离散投影矩阵 A（92160 条射线 x 65536 像素）；
  第三阶段前半：用附件1/2标准模板率定增益 gain，并完成模板验证重建。

首次运行自动完成全部计算并把结果缓存到本目录（ct_shared）：
  geometry_cache.json      标定参数 + 增益
  proj_matrix_I.npy        投影矩阵像素索引（int32,  约190MB）
  proj_matrix_L.npy        投影矩阵穿过长度（float32, 约190MB）
  template_validation.npz  模板验证重建结果与指标
之后 q2 / q3 直接读缓存，几秒钟即可使用。

用法：
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ct_shared"))
    from ct_model import CTModel

    model = CTModel()                      # 自动构建或读取缓存
    model.print_validation()               # 打印模板验证结果（缓存）
    U = model.reconstruct(proj, iter_lim=150)   # proj: 512x180 投影数据
"""

import os
import json
import time
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.sparse.linalg import LinearOperator, lsqr

SHARED_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SHARED_DIR)
XLS_PATH = os.path.join(PROJECT_DIR, "A题附件.xls")
TABLES_DIR = os.path.join(PROJECT_DIR, "2017-A-Q1", "tables")
CAL_PATH = os.path.join(TABLES_DIR, "calibration_result.csv")
ANG_PATH = os.path.join(TABLES_DIR, "calibrated_angles.csv")

GEO_CACHE = os.path.join(SHARED_DIR, "geometry_cache.json")
BLK_I_CACHE = os.path.join(SHARED_DIR, "proj_matrix_I.npy")
BLK_L_CACHE = os.path.join(SHARED_DIR, "proj_matrix_L.npy")
VAL_CACHE = os.path.join(SHARED_DIR, "template_validation.npz")

N = 256                    # 成像区域 256 x 256 像素
NDET, NANG = 512, 180      # 512 个探测器单元，180 次扫描
DELTA = 100.0 / N          # 像素边长 0.390625 mm


class CTModel:
    """CT 投影/重建共享模型。初始化时自动构建或读取缓存。"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.N, self.delta = N, DELTA
        self.ndet, self.nang = NDET, NANG
        t0 = time.time()
        cached = (os.path.exists(GEO_CACHE) and os.path.exists(BLK_I_CACHE)
                  and os.path.exists(BLK_L_CACHE) and os.path.exists(VAL_CACHE))
        if cached:
            self._load_cache()
            self._log("已从缓存载入投影模型（用时 %.1f s）" % (time.time() - t0))
        else:
            self._log("首次运行：开始构建投影模型并写入缓存……")
            self._build_all()
            self._log("模型构建完成（用时 %.1f s），缓存已写入 %s"
                      % (time.time() - t0, SHARED_DIR))
        self._make_operator()

    # -------------------- 日志 --------------------
    def _log(self, msg):
        if self.verbose:
            print(msg)

    # -------------------- 缓存读写 --------------------
    def _load_cache(self):
        with open(GEO_CACHE, encoding="utf-8") as f:
            geo = json.load(f)
        self.x0, self.y0, self.d = geo["x0"], geo["y0"], geo["d"]
        self.b, self.gain = geo["b"], geo["gain"]
        self.theta = np.array(geo["theta"])
        self.I_all = np.load(BLK_I_CACHE)
        self.L_all = np.load(BLK_L_CACHE)
        val = np.load(VAL_CACHE)
        self.U2 = val["U2"]
        self.val_metrics = json.loads(str(val["metrics"].item()))

    def _build_all(self):
        # ---- 第一阶段：标定参数 ----
        cal = pd.read_csv(CAL_PATH)
        ang = pd.read_csv(ANG_PATH)
        self.x0, self.y0, self.d = cal["x0"][0], cal["y0"][0], cal["d"][0]
        self.theta = np.deg2rad(ang["angle_degree"].values)
        self._log("标定参数：x0=%.4f, y0=%.4f, d=%.4f, 角度 %.3f°~%.3f°"
                  % (self.x0, self.y0, self.d,
                     np.degrees(self.theta[0]), np.degrees(self.theta[-1])))

        # 探测器零点 b：与第一问同款的小圆正弦拟合
        proj2 = pd.read_excel(XLS_PATH, sheet_name="附件2", header=None).values
        self.b = self._fit_detector_zero(proj2)
        self._log("探测器零点 b = %.4f" % self.b)

        # ---- 第二阶段：Siddon 构建投影矩阵 A ----
        t0 = time.time()
        cos_t, sin_t = np.cos(self.theta), np.sin(self.theta)
        det_s = (np.arange(NDET) - self.b) * self.d
        I_all = np.zeros((NANG, NDET, 2 * N + 1), dtype=np.int32)
        L_all = np.zeros((NANG, NDET, 2 * N + 1), dtype=np.float32)
        grid = np.arange(N + 1) * DELTA
        for k in range(NANG):
            I_all[k], L_all[k] = self._siddon_block(
                cos_t[k], sin_t[k], det_s, grid)
        self.I_all, self.L_all = I_all, L_all
        self._make_operator()
        self._log("投影矩阵 A 构建完成（%.1f s）：%d 条射线 × %d 个像素，非零元 %.2f 百万"
                  % (time.time() - t0, NDET * NANG, N * N,
                     (self.L_all > 0).sum() / 1e6))

        # ---- 第三阶段前半：增益率定 + 模板验证 ----
        template = pd.read_excel(XLS_PATH, sheet_name="附件1", header=None).values
        Au_tpl = self.A_mv(template.ravel().astype(np.float64))
        p2 = proj2.T.ravel()
        self.gain = float(np.dot(Au_tpl, p2) / np.dot(Au_tpl, Au_tpl))
        rmse = float(np.sqrt(np.mean((p2 - self.gain * Au_tpl) ** 2)))
        self._log("增益率定：gain = %.4f（直接估计 %.4f），模板投影残差 RMSE = %.4f"
                  % (self.gain, proj2.max() / 80.0, rmse))

        t0 = time.time()
        u2 = lsqr(self.Aop, (proj2 / self.gain).T.ravel(),
                  iter_lim=150, atol=1e-10, btol=1e-10)[0]
        self.U2 = np.clip(u2, 0, None).reshape(N, N)
        corr = float(np.corrcoef(self.U2.ravel(), template.ravel())[0, 1])
        regions = []
        lab, nreg = ndimage.label(self.U2 > 0.5)
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, nreg + 1))
        for i in [i + 1 for i, s in enumerate(sizes) if s * DELTA ** 2 > 20]:
            mm = lab == i
            ys, xs = np.where(mm)
            regions.append({
                "area_mm2": float(mm.sum() * DELTA ** 2),
                "cx": float((xs.mean() + 0.5) * DELTA),
                "cy": float((ys.mean() + 0.5) * DELTA),
                "mean_u": float(self.U2[mm].mean()),
            })
        self.val_metrics = {"corr": corr, "regions": regions,
                            "iter_lim": 150}
        self._log("模板验证重建完成（%.1f s），与附件1相关系数 = %.4f"
                  % (time.time() - t0, corr))

        # ---- 写缓存 ----
        with open(GEO_CACHE, "w", encoding="utf-8") as f:
            json.dump({"x0": self.x0, "y0": self.y0, "d": self.d,
                       "b": self.b, "gain": self.gain,
                       "theta": self.theta.tolist()}, f)
        np.save(BLK_I_CACHE, self.I_all)
        np.save(BLK_L_CACHE, self.L_all)
        np.savez(VAL_CACHE, U2=self.U2,
                 metrics=json.dumps(self.val_metrics, ensure_ascii=False))

    @staticmethod
    def _fit_detector_zero(proj2):
        """小圆独立投影区间的质心轨迹正弦拟合，取偏移量 b。"""
        scans, centers = [], []
        for scan in range(proj2.shape[1]):
            y = proj2[:, scan]
            mask = y > 0
            change = np.diff(np.r_[False, mask, False].astype(int))
            starts = np.where(change == 1)[0]
            ends = np.where(change == -1)[0] - 1
            narrow = [(l, r) for l, r in zip(starts, ends)
                      if 10 <= r - l + 1 <= 45]
            if len(narrow) != 1:
                continue
            l, r = narrow[0]
            idx = np.arange(l, r + 1)
            w = y[l:r + 1]
            scans.append(scan + 1)
            centers.append(np.sum(idx * w) / np.sum(w))

        def sine_model(i, A, omega, phi, b):
            return A * np.sin(omega * i + phi) + b

        popt, _ = curve_fit(
            sine_model, np.array(scans), np.array(centers),
            p0=[200.0, np.pi / 180.0, 2.2, 256.0],
            bounds=([0, np.pi / 360, 0, 0], [500, np.pi / 90, 2 * np.pi, 512]),
            maxfev=10000)
        return float(popt[3])

    def _siddon_block(self, cos_k, sin_k, det_s, grid):
        """单个角度下 512 条平行射线的 Siddon 精确追踪。"""
        n = np.array([cos_k, sin_k])
        t = np.array([-sin_k, cos_k])
        p0 = np.array([self.x0, self.y0])[None, :] + det_s[:, None] * n[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            lam_x = (grid[None, :] - p0[:, 0:1]) / t[0]
            lam_y = (grid[None, :] - p0[:, 1:2]) / t[1]
        lam = np.sort(np.concatenate([lam_x, lam_y], axis=1), axis=1)
        lam_mid = 0.5 * (lam[:, :-1] + lam[:, 1:])
        seg_len = lam[:, 1:] - lam[:, :-1]
        ix = np.floor((p0[:, 0:1] + lam_mid * t[0]) / DELTA).astype(np.int32)
        iy = np.floor((p0[:, 1:2] + lam_mid * t[1]) / DELTA).astype(np.int32)
        valid = (ix >= 0) & (ix < N) & (iy >= 0) & (iy < N) & np.isfinite(lam_mid)
        I = np.where(valid, iy * N + ix, 0).astype(np.int32)
        L = np.where(valid, seg_len, 0.0).astype(np.float32)
        return np.ascontiguousarray(I), np.ascontiguousarray(L)

    # -------------------- 投影算子 --------------------
    def _make_operator(self):
        I_all, L_all = self.I_all, self.L_all

        def A_mv(u):
            y = np.empty(NANG * NDET, dtype=np.float64)
            for k in range(NANG):
                y[k * NDET:(k + 1) * NDET] = (L_all[k] * u[I_all[k]]).sum(axis=1)
            return y

        def AT_mv(y):
            u = np.zeros(N * N, dtype=np.float64)
            for k in range(NANG):
                p = y[k * NDET:(k + 1) * NDET]
                u += np.bincount(I_all[k].ravel(),
                                 weights=(L_all[k] * p[:, None]).ravel(),
                                 minlength=N * N)
            return u

        self.A_mv, self.AT_mv = A_mv, AT_mv
        self.Aop = LinearOperator((NANG * NDET, N * N),
                                  matvec=A_mv, rmatvec=AT_mv)

    # -------------------- 对外接口 --------------------
    def reconstruct(self, proj, iter_lim=150, denoise_sigma=None):
        """由 512x180 投影数据重建吸收率分布。

        参数：
          proj          : 附件投影数据（512 行探测器 × 180 列扫描）
          iter_lim      : LSQR 迭代次数（含噪数据可取小，迭代即正则化）
          denoise_sigma : 可选，正弦图高斯降噪的 sigma（如 (1.5, 1.0)）
        返回：
          U : (256, 256) 吸收率分布，U[iy, ix] 对应
              x = (ix+0.5)*0.390625 mm, y = (iy+0.5)*0.390625 mm
        """
        if denoise_sigma is not None:
            proj = ndimage.gaussian_filter(proj, sigma=denoise_sigma)
        t0 = time.time()
        u = lsqr(self.Aop, (proj / self.gain).T.ravel(),
                 iter_lim=iter_lim, atol=1e-10, btol=1e-10)[0]
        U = np.clip(u, 0, None).reshape(N, N)
        self._log("重建完成：iter_lim=%d，用时 %.1f s，值域 %.3f ~ %.3f"
                  % (iter_lim, time.time() - t0, U.min(), U.max()))
        return U

    def print_validation(self):
        """打印模板验证结果（来自缓存或首次构建）。"""
        m = self.val_metrics
        print("模板验证（增益 %.4f，LSQR %d 次迭代）："
              % (self.gain, m["iter_lim"]))
        print("  重建模板与附件1的相关系数 = %.4f" % m["corr"])
        for i, r in enumerate(m["regions"], 1):
            print("  模板区域%d：面积 %.1f mm²，质心 (%.2f, %.2f) mm，平均吸收率 %.3f"
                  % (i, r["area_mm2"], r["cx"], r["cy"], r["mean_u"]))
