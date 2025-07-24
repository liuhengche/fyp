# grobi
import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import gurobipy as gp
from gurobipy import GRB
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Paths（请根据实际路径修改）
PREDICTION_CSV = r"graph_high_0_correlations.csv"
#OD_GROUND_TRUTH_CSV = r"D:\desktop\ta\202504\dataset\od\high\high_0.csv"
OD_GROUND_TRUTH_CSV = r"D:\desktop\ta\202504\dataset\demand\high\high_0.csv"
#DETECTOR_FLOW_CSV = r"D:\desktop\ta\202504\dataset\qkv\high\high_0.csv"
DETECTOR_FLOW_CSV = r"D:\desktop\ta\202504\dataset\demand\high\high_0.csv"

OUTPUT_DIR = "new_rmse_y_reconstruction"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_ground_truth_od(od_path):
    """Load i,j,k → vehsPerHour mapping"""
    df = pd.read_csv(od_path)
    return dict(zip(df['i-j-k'], df['od_num(ijk)']))
    # df['od'] = df['i'].astype(str) + '-' + df['j'].astype(str) + '-' + df['k'].astype(str)
    # return dict(zip(df['od'], df['vehsPerHour']))

def load_detector_flows(detector_path):
    """Load detector flows per interval"""
    df = pd.read_csv(detector_path)
    df['detector_time'] = df['edge'].astype(str) + '_' + df['interval_begin'].astype(int).astype(str)
    return dict(zip(df['detector_time'], df['link_count(lt)']))
    # df['detector_time'] = df['edge'].astype(str) + '_' + df['interval_begin'].astype(int).astype(str)
    # return dict(zip(df['detector_time'], df['edge_flow']))

def build_linear_system(prediction_df, flow_map):
    """构建稀疏系统 A·x = b"""
    od_pairs = prediction_df['od'].unique()
    od_to_idx = {od: i for i, od in enumerate(od_pairs)}
    num_od = len(od_pairs)

    grouped = prediction_df.groupby('detector_time')
    dt_list = []
    A_rows, A_cols, A_data, b_data = [], [], [], []

    for dt, group in grouped:
        if dt not in flow_map:
            continue

        flow = flow_map[dt]
        # b_data.append(flow)
        b_data.append(flow) # normalize by time interval, 30s => 6min


        for _, row in group.iterrows():
            od = row['od']
            alpha = row['predicted_value'] # changed to 'alpha' for ground truth value, 'predicted_value' for predicted value
            A_rows.append(len(dt_list))
            A_cols.append(od_to_idx[od])
            A_data.append(alpha)

        dt_list.append(dt)

    num_lt = len(dt_list)
    A = csr_matrix((A_data, (A_rows, A_cols)), shape=(num_lt, num_od))
    b = np.array(b_data)
    
    print(f"✅ Built sparse system: {A.shape[0]} equations, {A.shape[1]} variables")
    return A, b, od_to_idx, od_pairs

def solve_od_reconstruction(A, b):
    """Gurobi 求解非负约束问题"""
    try:
        # 初始化模型
        model = gp.Model("OD_Reconstruction")
        model.setParam('OutputFlag', 0)  # 关闭日志输出

        # 定义非负变量
        x = model.addMVar(A.shape[1], lb=0.0, name="x")  # 非负约束

        # 构建目标函数 ||A@x - b||²
        residual = A @ x - b
        obj = residual @ residual  # 二次目标
        model.setObjective(obj, GRB.MINIMIZE)

        # 求解
        model.optimize()

        if model.status == GRB.OPTIMAL:
            x_sol = x.X
            cost = model.ObjVal
            print(f"✅ Objective value: {cost:.4f}")
            print(f"✅ Solution norm: {np.linalg.norm(x_sol):.4f}")
            return x_sol
        else:
            raise RuntimeError(f"求解失败，状态码：{model.status}")

    except gp.GurobiError as e:
        print(f"Gurobi 错误：{e}")

def evaluate_reconstruction(x, od_pairs, od_truth):
    """
    Match reconstructed x with true OD demand and compute metrics
    """
    # Create DataFrame from solution vector x
    recon_df = pd.DataFrame({
        'od': od_pairs,
        'reconstructed_od': x
    })
    
    # Split i, j, k for readability
    recon_df[['i', 'j', 'k']] = recon_df['od'].str.split('-', expand=True).astype(int)
    
    # Map true OD values
    recon_df['true_od'] = recon_df['od'].map(od_truth)
    recon_df.dropna(inplace=True)

    # Compute metrics
    true_values = recon_df['true_od'].values
    pred_values = recon_df['reconstructed_od'].values

    # Reject negative predictions
    valid_mask = pred_values > 0
    filtered_true = true_values[valid_mask]
    filtered_pred = pred_values[valid_mask]

    if len(filtered_true) > 1:
        corr, _ = pearsonr(filtered_true, filtered_pred)
        rmse = np.sqrt(mean_squared_error(filtered_true, filtered_pred))
    else:
        corr, rmse = 0.0, float('inf')

    print(f"Pearson Correlation: {corr:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.scatter(filtered_true, filtered_pred, s=10, alpha=0.7, label=f"Pearson = {corr:.4f}")
    plt.plot([0, max(filtered_true)], [0, max(filtered_pred)], 'r--', label='Perfect Fit')
    plt.xlabel("True OD Demand (vehsPerHour)")
    plt.ylabel("Reconstructed OD Demand")
    plt.title(f"True vs Reconstructed OD Demand\nPearson: {corr:.4f}, RMSE: {rmse:.4f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "od_reconstruction.png"), dpi=300)
    plt.close()

    # Save results
    recon_df.to_csv(os.path.join(OUTPUT_DIR, "reconstructed_vs_true.csv"), index=False)
    return corr, rmse, recon_df

def main():
    # Load prediction data
    prediction_df = pd.read_csv(PREDICTION_CSV)
    prediction_df['od'] = prediction_df['i'].astype(str) + '-' + prediction_df['j'].astype(str) + '-' + prediction_df['k'].astype(str)
    prediction_df['detector_time'] = prediction_df['detector_id'].astype(str) + '_' + prediction_df['interval_begin'].astype(int).astype(str)

    # Load detector flows
    flow_map = load_detector_flows(DETECTOR_FLOW_CSV)

    # Build sparse system
    A, b, od_to_idx, od_pairs = build_linear_system(prediction_df, flow_map)

    # Solve for OD demand with non-negative constraint
    x = solve_od_reconstruction(A, b)

    # Load true OD demand
    od_truth = load_ground_truth_od(OD_GROUND_TRUTH_CSV)

    # Evaluate and get recon_df
    corr, rmse, recon_df = evaluate_reconstruction(x, od_pairs, od_truth)

    # Save metrics
    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"Pearson Correlation: {corr:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"Total matched OD pairs: {len(recon_df)}\n")
        f.write(f"Solution norm: {np.linalg.norm(x):.4f}\n")
        
if __name__ == "__main__":
    main()
