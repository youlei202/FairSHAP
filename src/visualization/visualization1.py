import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # 用于平滑曲线

def plot_fairness_improvement(
    folds,
    original_accuracy,  
    original_DR,      
    original_DP,      
    original_EO,      
    original_PQP,     
    stop_when_no_data=3,
    min_action=1,
    baseline=0.0,
    figsize=None,  
    fill_alpha=0.2,
    fill_color='b',
    smooth_window=15,  # 滑动窗口大小（用于平滑曲线）
    smooth_polyorder=3 # 多项式拟合阶数（用于 Savitzky-Golay 滤波）
):
    """
    绘制 1×N 子图，每个子图展示 Accuracy、DR、DP、EO、PQP 指标的
    “相对于原始指标值的改善差值”随 action_number 变化的均值±标准差曲线。

    主要优化：
    1) 计算滑动平均，减少剧烈波动，使曲线更平滑。
    2) 使用 Savitzky-Golay 滤波器，减少噪声但保留趋势。
    3) 适当调整透明度，提升可读性。
    4) 自动调整 `ylim`，避免极端值影响观察。

    """
    num_folds = len(folds)
    if not (len(original_accuracy) == len(original_DR) == len(original_DP) == len(original_EO) == len(original_PQP) == num_folds):
        raise ValueError("original_accuracy, original_DR, original_DP, original_EO, original_PQP 长度必须与 folds 相同。")

    measures_info = [
        ("Accuracy", "new_accuracy", original_accuracy),
        ("DR",  "new_DR",  original_DR),
        ("DP",  "new_DP",  original_DP),
        ("EO",  "new_EO",  original_EO),
        ("PQP", "new_PQP", original_PQP),
    ]

    num_subplots = len(measures_info)
    num_rows = 1  
    num_cols = num_subplots  

    if figsize is None:
        figsize = (num_subplots * 5, 5)

    fig = plt.figure(figsize=figsize)

    for i, (measure_name, measure_col, original_list) in enumerate(measures_info, start=1):
        
        for df, orig_val in zip(folds, original_list):
            df['action_number'] = pd.to_numeric(df['action_number'], errors='coerce')
            df[measure_col] = df[measure_col] - orig_val

        max_actions = [df['action_number'].max() for df in folds if not df.empty]
        if len(max_actions) == 0:
            print(f"警告：所有 fold 的 {measure_col} 数据都是空的或无效，跳过该子图。")
            continue
        overall_max_action = int(np.nanmax(max_actions))

        measure_values = {}
        for action in range(min_action, overall_max_action + 1):
            current_list = []
            count_no_data = 0

            for df in folds:
                row = df.loc[df['action_number'] == action, measure_col]
                if row.empty:
                    count_no_data += 1
                else:
                    current_list.append(row.values[0])
            
            if count_no_data >= stop_when_no_data:
                break
            
            measure_values[action] = current_list

        action_range = sorted(measure_values.keys())
        if len(action_range) == 0:
            print(f"警告：{measure_name} 未找到满足 stop_when_no_data 的有效 action_number，跳过该子图。")
            continue

        means = np.array([np.mean(measure_values[action]) for action in action_range])
        stds = np.array([np.std(measure_values[action]) for action in action_range])

        # 使用 Savitzky-Golay 滤波器进行平滑
        if len(means) > smooth_window:
            smoothed_means = savgol_filter(means, window_length=smooth_window, polyorder=smooth_polyorder)
        else:
            smoothed_means = means  

        ax = fig.add_subplot(num_rows, num_cols, i)
        ax.set_title(f"{measure_name} Improvement from Original")

        # baseline 参考线
        ax.axhline(y=baseline, color='black', linewidth=2, linestyle='-', label=f'Baseline (y={baseline})')

        # 平滑后的均值曲线
        ax.plot(action_range, smoothed_means, color=fill_color, label=f'Mean {measure_name} Gap')

        # 均值 ± 标准差
        ax.fill_between(
            action_range,
            smoothed_means - stds,
            smoothed_means + stds,
            alpha=fill_alpha,
            color=fill_color,
            label='±1 std dev'
        )

        ax.set_xlabel("Action Number")
        ax.set_ylabel(f"{measure_name} - Original")
        ax.grid(True)
        ax.legend()

        # 限制 Y 轴范围，避免极端值影响可视化
        ax.set_ylim([min(smoothed_means - stds) * 1.2, max(smoothed_means + stds) * 1.2])

    plt.subplots_adjust(wspace=0.3)
    plt.show()
