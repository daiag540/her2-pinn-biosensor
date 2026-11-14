# -*- coding: utf-8 -*-
"""
生物传感器 PINN 神经网络模型完整工作流
===================================================================
流程：读取 CHI 原始数据 (it / ocp / eis) → 构建训练集 → 训练带 Cottrell
与相变约束的 PINN → 提取临界相变参数 → 超参数敏感性分析 → 导出
Supporting Information Figure S1 四联图 (600 DPI PNG + 矢量 PDF)。

输入数据位于仓库的 data/raw 与 data/processed，输出写入 results/；
可用环境变量 PINN_DATA_DIR 覆盖数据仓库路径。
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import integrate

# ===================== 全局绘图样式配置 =====================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'stixsans',
    'axes.unicode_minus': False,
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'figure.dpi': 300,
})


def set_seed(seed: int = 42):
    """锁定全局随机种子，确保批量与单次运行结果完全一致"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================
# 模块 1: 数据提取器
# ==========================================
class DataExtractor:
    """从原始 CHI 文件提取并构建 PINN 训练数据集"""

    def __init__(self, folder_path: str, static_params: Dict,
                 output_dir: Optional[str] = None):
        self.folder_path = folder_path
        self.output_dir = output_dir or folder_path
        self.static_params = static_params

    def read_chi_file(self, prefix: str, seg_idx: int) -> pd.DataFrame:
        filename_candidates = [
            f"{prefix}_{seg_idx}.txt",
            f"{prefix}_%n{seg_idx}.txt"
        ]

        path = None
        for fname in filename_candidates:
            candidate = os.path.join(self.folder_path, fname)
            if os.path.exists(candidate):
                path = candidate
                break

        if path is None:
            raise FileNotFoundError(f"无法找到文件: {prefix}_{seg_idx}")

        with open(path, 'r', encoding='gbk', errors='ignore') as f:
            lines = f.readlines()

        data_start = 0
        for idx, line in enumerate(lines):
            clean = line.strip()
            if clean and (clean[0].isdigit() or clean[0] in ['-', '+', '.']):
                data_start = idx
                break

        df = pd.read_csv(path, skiprows=data_start, sep=None, engine='python', header=None)
        return df.apply(pd.to_numeric, errors='coerce').dropna()

    def extract_segment_data(self, num_segments: int = 10) -> pd.DataFrame:
        processed_list = []
        total_q = 0.0

        print(f"开始处理 {num_segments} 个数据段...")

        for i in range(1, num_segments + 1):
            try:
                # 1. it 曲线积分计算累计电量 Q (mC)
                df_it = self.read_chi_file("it", i)
                t = df_it.iloc[:, 0].values
                curr = df_it.iloc[:, 1].values
                q_segment = integrate.trapezoid(np.abs(curr), x=t) * 1000.0  # mC
                total_q += q_segment

                # 2. OCP 平均值
                df_ocp = self.read_chi_file("ocp", i)
                ocp_val = df_ocp.iloc[:, 1].mean()

                # 3. EIS 实部阻抗 Z_real 平均值
                df_eis = self.read_chi_file("eis", i)
                z_real = df_eis.iloc[:, 1].mean()

                processed_list.append({
                    'segment': i,
                    'Time': i * 3.0,  # 脉冲周期: 0.15s + 2.85s = 3.0s
                    'Q_mC': total_q,
                    'OCP': ocp_val,
                    'Z_real': z_real,
                    'sigma': self.static_params['sigma'],
                    'E_onset': self.static_params['E_onset'],
                    'A_base': self.static_params['A_base']
                })

            except Exception as e:
                print(f"警告: 第 {i} 段数据处理失败: {e}")
                continue

        final_df = pd.DataFrame(processed_list)
        if len(final_df) == 0:
            raise ValueError("未能成功处理任何数据段")

        diff_Z = np.abs(np.diff(final_df['Z_real'].values, prepend=final_df['Z_real'].iloc[0]))
        diff_Q = np.diff(final_df['Q_mC'].values, prepend=0)
        final_df['dR_dQ'] = np.where(diff_Q != 0, diff_Z / diff_Q, 0.0)
        final_df = final_df.round(4)

        output_path = os.path.join(self.output_dir, "pinn_final_training_set.csv")
        final_df.to_csv(output_path, index=False)

        print("-" * 50)
        print(f"数据提取完成。总电量 Q: {total_q:.2f} mC | 数据点数: {len(final_df)}")
        print(f"   保存路径: {output_path}")
        print("-" * 50)

        return final_df


# ==========================================
# 模块 2: PINN 模型与物理损失函数
# ==========================================
class BioSensorPINN(nn.Module):
    """
    生物传感器物理信息神经网络
    输入: 时间 t
    输出: [I_norm, R_norm, OCP_norm]
    可学习物理参数: log_D (lg扩散系数), Qc (穿透阈值), delta_t (时间修正)
    """

    def __init__(self, hidden_width: int = 64):
        super().__init__()
        self.hidden_width = hidden_width

        self.net = nn.Sequential(
            nn.Linear(1, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 3)
        )

        self.log_D = nn.Parameter(torch.tensor([-6.0]))      # lg(D)
        self.Qc = nn.Parameter(torch.tensor([0.30]))         # 临界电量初始设为 ~0.30 mC
        self.delta_t = nn.Parameter(torch.tensor([0.005]))   # 时间奇点正则化初始 0.005 s

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)

    def get_physical_params(self) -> Dict[str, float]:
        return {
            'D': torch.pow(10, self.log_D).item(),
            'Qc': torch.abs(self.Qc).item(),
            'delta_t': torch.abs(self.delta_t).item()
        }


def compute_pinn_loss(model: BioSensorPINN,
                      t_obs: torch.Tensor,
                      R_obs_norm: torch.Tensor,
                      OCP_obs_norm: torch.Tensor,
                      Q_obs: torch.Tensor,
                      stats: Dict,
                      params: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
    """计算 PINN 的全部物理约束损失分量，返回总损失及明细字典"""
    t_in = t_obs.clone().detach().requires_grad_(True)

    preds_norm = model(t_in)
    R_pred_norm = preds_norm[:, 1]
    OCP_pred_norm = preds_norm[:, 2]

    # A. 数据驱动损失 (L_data = L_data,R + L_data,OCP)
    loss_data_R = torch.mean((R_pred_norm - R_obs_norm.squeeze()) ** 2)
    loss_data_OCP = torch.mean((OCP_pred_norm - OCP_obs_norm.squeeze()) ** 2)
    loss_data = loss_data_R + loss_data_OCP

    # 反归一化
    R_pred_real = R_pred_norm * stats['R_std'] + stats['R_mean']

    # B. Cottrell 方程与扩散系数 D 物理约束 (L_Cottrell)
    n, F, C = 3, 96485.0, 1e-6  # 1.0 mM HAuCl4 -> C = 1e-6 mol/cm^3
    A_eff = params['A_base'] * params['sigma']
    D_real = torch.pow(10, model.log_D)

    t_phys = t_in.squeeze() + torch.abs(model.delta_t)
    I_theory = (n * F * A_eff * torch.sqrt(D_real) * C) / torch.sqrt(np.pi * t_phys)

    dt = (t_in.squeeze()[1] - t_in.squeeze()[0]).item() if len(t_in) > 1 else 3.0
    I_obs_approx = (Q_obs.squeeze() / dt) * 1e-3  # mC/s -> A
    loss_cottrell = torch.mean((I_theory - I_obs_approx) ** 2) * 1e6

    # C. Qc 临界相变物理约束 (L_phase: Sigmoid 阻抗渗流突变模型)
    Q_c_param = torch.abs(model.Qc)
    R_max = stats['R_mean'] + stats['R_std']
    R_min = stats['R_mean'] - stats['R_std']
    R_phase_theory = R_max - (R_max - R_min) / (1.0 + torch.exp(-(Q_obs.squeeze() - Q_c_param) / 0.05))
    loss_phase = torch.mean((R_pred_real - R_phase_theory) ** 2) / (stats['R_std'] ** 2)

    # D. 阻抗梯度平滑正则化 (L_deriv)
    try:
        dR_dt = torch.autograd.grad(
            R_pred_real.sum(), t_in, create_graph=True, retain_graph=True
        )[0]
        loss_deriv = torch.mean(torch.abs(dR_dt)) * 1e-4
    except Exception:
        loss_deriv = torch.tensor(0.0, device=t_in.device)

    total_loss = (
        loss_data
        + params['lambda_phys'] * loss_cottrell
        + 1.0 * loss_phase
        + params.get('lambda_z', 1e-3) * loss_deriv
    )

    metrics = {
        'total_loss': float(total_loss.item()),
        'data_loss': float(loss_data.item()),
        'R_loss': float(loss_data_R.item()),
        'OCP_loss': float(loss_data_OCP.item()),
        'cottrell_loss': float(loss_cottrell.item()),
        'phase_loss': float(loss_phase.item()),
        'deriv_loss': float(loss_deriv.item()),
        'D': float(D_real.item()),
        'Qc': float(Q_c_param.item()),
        'delta_t': float(torch.abs(model.delta_t).item())
    }

    return total_loss, metrics


# ==========================================
# 模块 3: PINN 训练器 (完整损失轨迹记录)
# ==========================================
class PINNTrainer:
    """PINN 模型训练器，记录全周期损失分量并保存训练审计日志"""

    def __init__(self, data_path: str, save_dir: Optional[str] = None):
        self.data_path = data_path
        self.save_dir = save_dir or os.path.dirname(data_path)
        os.makedirs(self.save_dir, exist_ok=True)

        log_file = os.path.join(self.save_dir, "pinn_training.log")
        logging.basicConfig(
            filename=log_file,
            filemode='w',
            format='[%(asctime)s][%(levelname)s] %(message)s',
            level=logging.INFO
        )

    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict, Dict]:
        data = pd.read_csv(self.data_path)

        extracted_params = {
            'A_base': float(data['A_base'].iloc[0]) if 'A_base' in data.columns else 0.1963,
            'sigma': float(data['sigma'].iloc[0]) if 'sigma' in data.columns else 1.6688
        }

        stats = {
            'R_mean': float(data['Z_real'].mean()),
            'R_std': float(data['Z_real'].std()),
            'OCP_mean': float(data['OCP'].mean()),
            'OCP_std': float(data['OCP'].std()),
            'I_mean': 0.0005,
            'I_std': 0.001
        }

        R_norm = (data['Z_real'].values - stats['R_mean']) / stats['R_std']
        OCP_norm = (data['OCP'].values - stats['OCP_mean']) / stats['OCP_std']

        t_train = torch.tensor(data['Time'].values, dtype=torch.float32).view(-1, 1)
        R_train = torch.tensor(R_norm, dtype=torch.float32).view(-1, 1)
        OCP_train = torch.tensor(OCP_norm, dtype=torch.float32).view(-1, 1)
        Q_train = torch.tensor(data['Q_mC'].values, dtype=torch.float32).view(-1, 1)

        return t_train, R_train, OCP_train, Q_train, stats, extracted_params

    def train(self, epochs: int = 5001, lr: float = 1e-3,
              lambda_phys: float = 10.0, lambda_z: float = 1e-3,
              hidden_width: int = 64, seed: int = 42,
              record_interval: int = 10, save_artifacts: bool = True) -> Tuple[BioSensorPINN, pd.DataFrame, Dict]:

        set_seed(seed)
        t_train, R_train, OCP_train, Q_train, stats, extracted_params = self.prepare_data()

        model = BioSensorPINN(hidden_width=hidden_width)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        params = {
            'sigma': extracted_params['sigma'],
            'A_base': extracted_params['A_base'],
            'lambda_phys': lambda_phys,
            'lambda_z': lambda_z
        }

        best_loss = float('inf')
        best_model_path = os.path.join(self.save_dir, "biosensor_pinn_model_best.pth")
        final_model_path = os.path.join(self.save_dir, "biosensor_pinn_model.pth")

        history_records: List[Dict] = []

        if save_artifacts:
            print("\n" + "=" * 72)
            print(f"开始 PINN 训练 (seed={seed}, A_base={params['A_base']} cm², sigma={params['sigma']})")
            print("=" * 72)

        for epoch in range(epochs):
            loss, metrics = compute_pinn_loss(
                model, t_train, R_train, OCP_train, Q_train, stats, params
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if save_artifacts and loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'stats': stats,
                    'hidden_width': hidden_width,
                    'learned_D': metrics['D'],
                    'learned_Qc': metrics['Qc'],
                    'learned_delta_t': metrics['delta_t']
                }, best_model_path)

            if epoch % record_interval == 0:
                row = {'epoch': epoch, **metrics}
                history_records.append(row)

            if save_artifacts and epoch % 500 == 0:
                msg = (
                    f"Epoch {epoch:4d} | Total: {metrics['total_loss']:.4f} | "
                    f"Data: {metrics['data_loss']:.5f} | Cottrell: {metrics['cottrell_loss']:.5f} | "
                    f"Phase: {metrics['phase_loss']:.5f} | Deriv: {metrics['deriv_loss']:.5f} | "
                    f"D: {metrics['D']:.2e} cm²/s | Qc: {metrics['Qc']:.4f} mC"
                )
                print(msg)
                logging.info(msg)

        history_df = pd.DataFrame(history_records)

        if save_artifacts:
            history_df.to_csv(os.path.join(self.save_dir, "pinn_loss_history.csv"), index=False)
            with open(os.path.join(self.save_dir, "training_history.json"), "w", encoding="utf-8") as f:
                json.dump(history_df.to_dict(orient='list'), f, indent=4)

            torch.save({
                'epoch': epochs - 1,
                'model_state_dict': model.state_dict(),
                'final_loss': metrics['total_loss'],
                'stats': stats,
                'hidden_width': hidden_width,
                'learned_D': metrics['D'],
                'learned_Qc': metrics['Qc'],
                'learned_delta_t': metrics['delta_t']
            }, final_model_path)

            print("=" * 72)
            print(f"训练完成。最佳总损失: {best_loss:.4f} | 损失曲线数据已保存至 pinn_loss_history.csv")
            print("=" * 72)

        return model, history_df, stats


# ==========================================
# 模块 4: 刺穿点 (相变节点) 分析器
# ==========================================
class PiercePointAnalyzer:
    """从训练好的 PINN 模型提取刺穿点 (相变节点) 参数"""

    def __init__(self, model_path: Optional[str] = None, data_path: Optional[str] = None,
                 model: Optional[BioSensorPINN] = None, stats: Optional[Dict] = None):
        self.model_path = model_path
        self.data_path = data_path
        self.model = model
        self.stats = stats

    def load_model(self) -> BioSensorPINN:
        checkpoint = torch.load(self.model_path, weights_only=False)
        width = checkpoint.get('hidden_width', 64)
        model = BioSensorPINN(hidden_width=width)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        self.model = model
        self.stats = checkpoint['stats']
        return model

    def find_pierce_point(self, t_range: Tuple[float, float] = None,
                          num_points: int = 2000) -> Dict[str, float]:
        if self.model is None:
            self.load_model()

        data = pd.read_csv(self.data_path)
        if t_range is None:
            t_min, t_max = float(data['Time'].min()), float(data['Time'].max())
        else:
            t_min, t_max = t_range

        t_fine = torch.linspace(t_min, t_max, num_points).view(-1, 1)

        with torch.no_grad():
            preds = self.model(t_fine)
            R_real = preds[:, 1].numpy() * self.stats['R_std'] + self.stats['R_mean']

        t_np = t_fine.squeeze().numpy()
        dR_dt = np.gradient(R_real, t_np)
        max_idx = int(np.argmax(np.abs(dR_dt)))

        t_pierce = float(t_np[max_idx])
        R_pierce = float(R_real[max_idx])
        Q_interp = float(np.interp(t_pierce, data['Time'].values, data['Q_mC'].values))
        phys = self.model.get_physical_params()

        return {
            't_crit': t_pierce,
            'Q_c_learned': phys['Qc'],
            'Q_interp_at_tcrit': Q_interp,
            'R_c': R_pierce,
            'D_eff': phys['D'],
            'delta_t': phys['delta_t'],
            'dR_dt_max': float(np.abs(dR_dt[max_idx]))
        }

    def print_results(self, results: Dict[str, float]):
        print("\n" + "=" * 68)
        print("            金纳米颗粒刺穿 rGO 层临界相变节点参数")
        print("=" * 68)
        print(f" 临界相变时刻 t_crit      = {results['t_crit']:.2f} s")
        print(f" 学习渗流电荷阈值 Q_c     = {results['Q_c_learned']:.3f} mC (插值电量: {results['Q_interp_at_tcrit']:.3f} mC)")
        print(f" 临界相变阻抗 R_c         = {results['R_c']:.2f} Ω")
        print(f" 表观扩散系数 D_eff       = {results['D_eff']:.2e} cm² s⁻¹")
        print(f" 时间奇点正则化 |δt|      = {results['delta_t']:.4f} s")
        print(f" 最大阻抗衰减率 |∂Z/∂t|   = {results['dR_dt_max']:.4f} Ω s⁻¹")
        print("=" * 68)


# ==========================================
# 模块 5: 超参数与损失权重敏感性分析器 (对应 Section S2.6)
# ==========================================
class SensitivityAnalyzer:
    """系统评估 λ_phys、λ_Z、网络宽度与学习率对 t_crit 预测鲁棒性的影响"""

    def __init__(self, data_path: str, save_dir: str):
        self.data_path = data_path
        self.save_dir = save_dir
        self.trainer = PINNTrainer(data_path, save_dir=save_dir)

    def run_sensitivity_study(self, epochs: int = 5001, seed: int = 42) -> pd.DataFrame:
        print("\n" + "=" * 72)
        print("【步骤 4】执行超参数与损失权重敏感性分析 (Supporting Information S2.6)...")
        print("=" * 72)

        records = []

        # 1. 物理权重扫描: λ_phys ∈ [1.0, 50.0]
        for lp in [1.0, 5.0, 10.0, 25.0, 50.0]:
            m, _, st = self.trainer.train(
                epochs=epochs, lr=1e-3, lambda_phys=lp, lambda_z=1e-3,
                hidden_width=64, seed=seed, record_interval=epochs, save_artifacts=False
            )
            res = PiercePointAnalyzer(data_path=self.data_path, model=m, stats=st).find_pierce_point()
            records.append({
                'sweep_type': 'lambda_phys', 'param_value': lp,
                'lambda_phys': lp, 'lambda_z': 1e-3, 'hidden_width': 64, 'lr': 1e-3,
                't_crit': res['t_crit'], 'D_eff': res['D_eff'], 'Q_c': res['Q_c_learned']
            })

        # 2. 导数正则化权重扫描: λ_Z ∈ [1e-4, 1e-2]
        for lz in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
            m, _, st = self.trainer.train(
                epochs=epochs, lr=1e-3, lambda_phys=10.0, lambda_z=lz,
                hidden_width=64, seed=seed, record_interval=epochs, save_artifacts=False
            )
            res = PiercePointAnalyzer(data_path=self.data_path, model=m, stats=st).find_pierce_point()
            records.append({
                'sweep_type': 'lambda_z', 'param_value': lz,
                'lambda_phys': 10.0, 'lambda_z': lz, 'hidden_width': 64, 'lr': 1e-3,
                't_crit': res['t_crit'], 'D_eff': res['D_eff'], 'Q_c': res['Q_c_learned']
            })

        # 3. 隐藏层宽度扫描: width ∈ {32, 64, 128}
        for w in [32, 64, 128]:
            m, _, st = self.trainer.train(
                epochs=epochs, lr=1e-3, lambda_phys=10.0, lambda_z=1e-3,
                hidden_width=w, seed=seed, record_interval=epochs, save_artifacts=False
            )
            res = PiercePointAnalyzer(data_path=self.data_path, model=m, stats=st).find_pierce_point()
            records.append({
                'sweep_type': 'hidden_width', 'param_value': w,
                'lambda_phys': 10.0, 'lambda_z': 1e-3, 'hidden_width': w, 'lr': 1e-3,
                't_crit': res['t_crit'], 'D_eff': res['D_eff'], 'Q_c': res['Q_c_learned']
            })

        # 4. 初始学习率扫描: η ∈ {1e-4, 1e-3, 1e-2}
        for lr_val in [1e-4, 1e-3, 1e-2]:
            m, _, st = self.trainer.train(
                epochs=epochs, lr=lr_val, lambda_phys=10.0, lambda_z=1e-3,
                hidden_width=64, seed=seed, record_interval=epochs, save_artifacts=False
            )
            res = PiercePointAnalyzer(data_path=self.data_path, model=m, stats=st).find_pierce_point()
            records.append({
                'sweep_type': 'learning_rate', 'param_value': lr_val,
                'lambda_phys': 10.0, 'lambda_z': 1e-3, 'hidden_width': 64, 'lr': lr_val,
                't_crit': res['t_crit'], 'D_eff': res['D_eff'], 'Q_c': res['Q_c_learned']
            })

        sens_df = pd.DataFrame(records)
        out_csv = os.path.join(self.save_dir, "pinn_sensitivity_results.csv")
        sens_df.to_csv(out_csv, index=False)

        print(sens_df[['sweep_type', 'param_value', 't_crit', 'D_eff', 'Q_c']].to_string(index=False))
        print(f"\n敏感性分析结果已保存至: {out_csv}")
        return sens_df


# ==========================================
# 模块 6: 图表可视化器 (含 Figure S1 四联图)
# ==========================================
class PINNVisualizer:
    """导出单图拟合结果及 Supporting Information Figure S1 四联分析图"""

    def __init__(self, model_path: str, data_path: str):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.stats = None

    def load_model(self):
        checkpoint = torch.load(self.model_path, weights_only=False)
        width = checkpoint.get('hidden_width', 64)
        model = BioSensorPINN(hidden_width=width)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        self.model = model
        self.stats = checkpoint['stats']

    def plot_figure_s1_four_panel(self, history_df: pd.DataFrame, sens_df: pd.DataFrame,
                                  save_prefix: str):
        """
        生成与 Supporting Information Figure S1 图注严格对应的四联曲线图:
        (a) Total training loss vs. epoch
        (b) Evolution of individual data, Cottrell, phase-transition, and derivative loss components
        (c) Sensitivity of predicted transition time t_crit to λ_phys and λ_Z
        (d) Comparison between experimentally measured responses (Z_real, OCP) and PINN predictions
        """
        if self.model is None:
            self.load_model()

        df = pd.read_csv(self.data_path)
        t_fine = torch.linspace(df['Time'].min(), df['Time'].max(), 2000).view(-1, 1)
        with torch.no_grad():
            preds_norm = self.model(t_fine)
            R_pred_real = preds_norm[:, 1].numpy() * self.stats['R_std'] + self.stats['R_mean']
            OCP_pred_real = preds_norm[:, 2].numpy() * self.stats['OCP_std'] + self.stats['OCP_mean']

        t_np = t_fine.squeeze().numpy()
        dR_dt = np.gradient(R_pred_real, t_np)
        t_crit = float(t_np[np.argmax(np.abs(dR_dt))])

        fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.8), dpi=300)
        plt.subplots_adjust(hspace=0.38, wspace=0.32, left=0.085, right=0.915, top=0.92, bottom=0.09)

        ep = history_df['epoch'].values

        # ---------------- Panel (a): Total Training Loss ----------------
        ax_a = axes[0, 0]
        tot_loss_vals = np.maximum(history_df['total_loss'].values, 1e-12)
        ax_a.semilogy(ep, tot_loss_vals,
                      color='#2C3E6B', linewidth=1.8, label=r'Total objective ($\mathcal{L}_{\mathrm{total}}$)')
        ax_a.set_yticks([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
        ax_a.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax_a.set_xlabel('Training Epoch', fontsize=11.0, fontweight='bold')
        ax_a.set_ylabel('Total Loss (log scale)', fontsize=11.0, fontweight='bold')
        ax_a.tick_params(labelsize=9.8)
        ax_a.legend(fontsize=9.2, loc='upper right', frameon=True, facecolor='white', edgecolor='#CBD5E1')
        ax_a.text(-0.17, 1.08, '(a)', transform=ax_a.transAxes, fontsize=13.5, fontweight='bold', va='top', ha='left')

        # ---------------- Panel (b): Individual Loss Components ----------------
        ax_b = axes[0, 1]
        ax_b.semilogy(ep, np.maximum(history_df['data_loss'].values, 1e-12),
                      color='#2C3E6B', linewidth=1.6, label=r'Data fidelity ($\mathcal{L}_{\mathrm{data}}$)')
        ax_b.semilogy(ep, np.maximum(history_df['cottrell_loss'].values, 1e-12),
                      color='#E05638', linewidth=1.6, label=r'Cottrell transport ($\mathcal{L}_{\mathrm{Cottrell}}$)')
        ax_b.semilogy(ep, np.maximum(history_df['phase_loss'].values, 1e-12),
                      color='#009A88', linewidth=1.6, label=r'Phase transition ($\mathcal{L}_{\mathrm{phase}}$)')
        ax_b.semilogy(ep, np.maximum(history_df['deriv_loss'].values, 1e-12),
                      color='#E69F00', linestyle='--', linewidth=1.5, label=r'Smoothness ($\mathcal{L}_{\mathrm{deriv}}$)')
        ax_b.set_xlabel('Training Epoch', fontsize=11.0, fontweight='bold')
        ax_b.set_ylabel('Component Loss (log scale)', fontsize=11.0, fontweight='bold')
        ax_b.tick_params(labelsize=9.8)
        ax_b.legend(fontsize=8.5, loc='upper right', frameon=True, facecolor='white', edgecolor='#CBD5E1')
        ax_b.text(-0.17, 1.08, '(b)', transform=ax_b.transAxes, fontsize=13.5, fontweight='bold', va='top', ha='left')

        # ---------------- Panel (c): Sensitivity to λ_phys and λ_Z ----------------
        ax_c = axes[1, 0]
        sub_phys = sens_df[sens_df['sweep_type'] == 'lambda_phys']
        sub_z = sens_df[sens_df['sweep_type'] == 'lambda_z']

        t_ref = float(np.median(np.concatenate([sub_phys['t_crit'].values, sub_z['t_crit'].values])))
        band_half = max(0.25, 0.02 * t_ref)  # ±2% 鲁棒区间 (约 ±0.32 s)

        ax_c.axhspan(t_ref - band_half, t_ref + band_half, color='#CBD5E1', alpha=0.45)
        ax_c.axhline(t_ref, color='#64748B', linestyle=':', linewidth=1.1)

        l1 = ax_c.semilogx(
            sub_phys['param_value'], sub_phys['t_crit'], '-o',
            color='#2C3E6B', linewidth=1.7, markersize=6.0,
            label=r'Sweep $\lambda_{\mathrm{phys}} \in [1, 50]$ ($\lambda_Z = 10^{-3}$)'
        )
        ax_c.set_xlabel(r'Physics Weight $\lambda_{\mathrm{phys}}$', fontsize=11.0, fontweight='bold', color='#2C3E6B')
        ax_c.set_ylabel(r'Predicted Transition Time $t_{\mathrm{crit}}$ (s)', fontsize=11.0, fontweight='bold')
        ax_c.set_ylim(t_ref - 1.5, t_ref + 1.5)
        ax_c.tick_params(axis='x', colors='#2C3E6B', labelsize=9.8)
        ax_c.tick_params(axis='y', labelsize=9.8)

        ax_c_top = ax_c.twiny()
        l2 = ax_c_top.semilogx(
            sub_z['param_value'], sub_z['t_crit'], '--s',
            color='#E05638', linewidth=1.6, markersize=5.5,
            label=r'Sweep $\lambda_Z \in [10^{-4}, 10^{-2}]$ ($\lambda_{\mathrm{phys}} = 10$)'
        )
        ax_c_top.set_xlabel(r'Derivative Weight $\lambda_Z$', fontsize=10.5, fontweight='bold', color='#E05638',
                            labelpad=5)
        ax_c_top.tick_params(axis='x', colors='#E05638', labelsize=9.5)

        band_patch = plt.Rectangle((0, 0), 1, 1, fc='#CBD5E1', alpha=0.45)
        ax_c.legend(
            [l1[0], l2[0], band_patch],
            [l1[0].get_label(), l2[0].get_label(), rf'Robust band (${t_ref:.2f} \pm {band_half:.2f}\ \mathrm{{s}}$)'],
            fontsize=8.0, loc='lower left', frameon=True, facecolor='white', edgecolor='#CBD5E1'
        )
        ax_c.text(-0.20, 1.15, '(c)', transform=ax_c.transAxes, fontsize=13.5, fontweight='bold', va='top', ha='left')

        # ---------------- Panel (d): Experimental vs. PINN Predictions ----------------
        ax_d = axes[1, 1]
        p1 = ax_d.scatter(df['Time'], df['Z_real'], color='#2C3E6B', s=38, zorder=4, label=r'Exp. $Z_{\mathrm{real}}$')
        p2, = ax_d.plot(t_np, R_pred_real, color='#2C3E6B', linewidth=1.8, zorder=3, label=r'PINN $Z_{\mathrm{real}}$')
        ax_d.axvline(t_crit, color='#64748B', linestyle='--', linewidth=1.1, zorder=2)

        ax_d.set_xlabel(r'Deposition Time $t$ (s)', fontsize=11.0, fontweight='bold')
        ax_d.set_ylabel(r'Real Impedance $Z_{\mathrm{real}}$ ($\Omega$)', fontsize=11.0, fontweight='bold', color='#2C3E6B')
        ax_d.tick_params(axis='y', colors='#2C3E6B', labelsize=9.8)
        ax_d.tick_params(axis='x', labelsize=9.8)

        ax_d_r = ax_d.twinx()
        p3 = ax_d_r.scatter(df['Time'], df['OCP'], marker='s', facecolors='white',
                            edgecolors='#E05638', linewidths=1.3, s=34, zorder=4, label='Exp. OCP')
        p4, = ax_d_r.plot(t_np, OCP_pred_real, color='#E05638', linestyle='--', linewidth=1.7, zorder=3, label='PINN OCP')
        ax_d_r.set_ylabel('Open-Circuit Potential OCP (V)', fontsize=11.0, fontweight='bold', color='#E05638')
        ax_d_r.tick_params(axis='y', colors='#E05638', labelsize=9.8)

        handles_d = [p1, p2, p3, p4]
        ax_d.legend(handles_d, [h.get_label() for h in handles_d],
                    fontsize=8.2, loc='center right', frameon=True, facecolor='white', edgecolor='#CBD5E1')
        ax_d.text(-0.17, 1.08, '(d)', transform=ax_d.transAxes, fontsize=13.5, fontweight='bold', va='top', ha='left')

        png_path = f"{save_prefix}.png"
        pdf_path = f"{save_prefix}.pdf"
        plt.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"Supporting Information Figure S1 四联图已导出:")
        print(f"   -> 600 DPI PNG: {png_path}")
        print(f"   -> 矢量 PDF:    {pdf_path}")


# ==========================================
# 主工作流入口
# ==========================================
def run_complete_workflow():
    # 所有路径相对仓库根目录解析，克隆后可在 Windows / Linux / macOS 直接运行
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.environ.get("PINN_DATA_DIR", os.path.join(REPO_ROOT, "data"))
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    RESULTS_DIR = os.path.join(REPO_ROOT, "results")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    DATA_FILE = os.path.join(PROCESSED_DIR, "pinn_final_training_set.csv")
    MODEL_BEST = os.path.join(RESULTS_DIR, "biosensor_pinn_model_best.pth")

    STATIC_PARAMS = {
        'sigma': 1.6688,
        'E_onset': 0.5983,
        'A_base': 0.1963  # 5 mm 直径 Working Disk 面积 (cm^2)
    }

    print("=" * 72)
    print("         生物传感器 PINN 神经网络模型完整工作流")
    print("=" * 72)

    # 1. 提取/读取数据集
    if not os.path.exists(DATA_FILE):
        print("\n【步骤 1】提取实验数据集...")
        extractor = DataExtractor(RAW_DIR, STATIC_PARAMS, output_dir=PROCESSED_DIR)
        data = extractor.extract_segment_data(num_segments=10)
    else:
        print(f"\n数据集已存在: {DATA_FILE}")
        data = pd.read_csv(DATA_FILE)

    # 2. 主模型训练 (seed=42, 记录每10轮全部损失分量)
    print("\n【步骤 2】训练主 PINN 模型并记录完整损失轨迹...")
    trainer = PINNTrainer(DATA_FILE, save_dir=RESULTS_DIR)
    trained_model, history_df, stats = trainer.train(
        epochs=5001, lr=1e-3, lambda_phys=10.0, lambda_z=1e-3,
        hidden_width=64, seed=42, record_interval=10, save_artifacts=True
    )

    # 3. 相变节点参数提取
    print("\n【步骤 3】计算相变节点临界参数...")
    analyzer = PiercePointAnalyzer(MODEL_BEST, DATA_FILE)
    pierce_results = analyzer.find_pierce_point()
    analyzer.print_results(pierce_results)

    # 4. 超参数与损失权重敏感性分析
    sens_analyzer = SensitivityAnalyzer(DATA_FILE, save_dir=RESULTS_DIR)
    sens_df = sens_analyzer.run_sensitivity_study(epochs=5001, seed=42)

    # 5. 导出 Supporting Information Figure S1 四联图
    print("\n【步骤 5】生成 Supporting Information Figure S1 四联曲线图...")
    visualizer = PINNVisualizer(MODEL_BEST, DATA_FILE)
    fig_s1_prefix = os.path.join(RESULTS_DIR, "Figure_S1_PINN_Training_and_Sensitivity")
    visualizer.plot_figure_s1_four_panel(history_df, sens_df, save_prefix=fig_s1_prefix)

    print("\n" + "=" * 72)
    print("PINN 工作流执行完毕，输出文件已保存至 results/。")
    print("=" * 72)

    return {
        'model': trained_model,
        'pierce_point': pierce_results,
        'history': history_df,
        'sensitivity': sens_df,
        'data': data
    }


if __name__ == "__main__":
    results = run_complete_workflow()
