import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # 用于平滑曲线

def plot_multi_dataset_fairness_improvement(
    datasets_info,
    stop_when_no_data=3,
    min_action=1,
    baseline=0.0,
    figsize=None,  
    fill_alpha=0.2,
    color_palette=['b', 'g', 'r', 'c', 'm', 'y'],
    smooth_window=30,
    smooth_polyorder=1
):
    """
    生成符合 ICLR/ICML 风格的公平性改进折线图，去除上边界和右边界。
    
    参数:
    -----------
    datasets_info : list of dicts
        每个数据集的信息，包括：
        - 'name': 数据集名称
        - 'folds': 数据帧列表
        - 'original_accuracy': 原始 Accuracy 值
        - 'original_DR': 原始 DR 值
        - 'original_DP': 原始 DP 值
        - 'original_EO': 原始 EO 值
        - 'original_PQP': 原始 PQP 值
    
    其他参数类似于原始 plot_fairness_improvement 函数。
    """
    measures_info = [
        ("Accuracy", "new_accuracy"),
        ("DR", "new_DR"),
        ("DP", "new_DP"),
        ("EO", "new_EO"),
        ("PQP", "new_PQP"),
    ]
    
    num_datasets = len(datasets_info)
    num_metrics = len(measures_info)
    
    # 自动确定图像大小
    if figsize is None:
        figsize = (num_metrics * 4, num_datasets * 3)
    
    fig = plt.figure(figsize=figsize)
    
    for dataset_idx, dataset_info in enumerate(datasets_info):
        dataset_name = dataset_info['name']
        folds = dataset_info['folds']
        
        for metric_idx, (measure_name, measure_col) in enumerate(measures_info):
            original_values = dataset_info[f'original_{measure_name}']
            
            subplot_idx = dataset_idx * num_metrics + metric_idx + 1
            ax = fig.add_subplot(num_datasets, num_metrics, subplot_idx)
            
            for df, orig_val in zip(folds, original_values):
                df['action_number'] = pd.to_numeric(df['action_number'], errors='coerce')
                df[measure_col] = df[measure_col] - orig_val
            
            max_actions = [df['action_number'].max() for df in folds if not df.empty]
            if len(max_actions) == 0:
                print(f"Warning: {dataset_name} - {measure_name} 数据为空或无效，跳过该子图。")
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
                print(f"Warning: {dataset_name} - {measure_name} 无有效 action 数据，跳过该子图。")
                continue
            
            means = np.array([np.mean(measure_values[action]) for action in action_range])
            stds = np.array([np.std(measure_values[action]) for action in action_range])
            
            # 平滑曲线
            if len(means) > smooth_window:
                smoothed_means = savgol_filter(means, window_length=smooth_window, polyorder=smooth_polyorder)
            else:
                smoothed_means = means
            
            color = color_palette[dataset_idx % len(color_palette)]
            
            ax.axhline(y=baseline, color='black', linewidth=1, linestyle='-', label=f'Baseline' if dataset_idx == 0 and metric_idx == 0 else None)
            
            ax.plot(action_range, smoothed_means, color=color, linewidth=2, label=f'Mean' if dataset_idx == 0 and metric_idx == 0 else None)
            
            ax.fill_between(
                action_range,
                smoothed_means - stds,
                smoothed_means + stds,
                alpha=fill_alpha,
                color=color,
                label='±1 std dev' if dataset_idx == 0 and metric_idx == 0 else None
            )
            
            # 设置标题和标签
            if dataset_idx == 0:
                ax.set_title(f"{measure_name}", fontsize=14)
            
            if metric_idx == 0:
                ax.set_ylabel(f"{dataset_name}\n{measure_name} - Original", fontsize=12)
            else:
                ax.set_ylabel(f"{measure_name} - Original", fontsize=12)
            
            if dataset_idx == num_datasets - 1:
                ax.set_xlabel("Action Number", fontsize=12)
            
            # 移除上边界和右边界，符合 ICLR/ICML 风格
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            # 添加网格
            ax.grid(True, linestyle='--', alpha=0.7)
            
            y_min = min(smoothed_means - stds) * 1.2 if min(smoothed_means - stds) < 0 else min(smoothed_means - stds) * 0.8
            y_max = max(smoothed_means + stds) * 1.2 if max(smoothed_means + stds) > 0 else max(smoothed_means + stds) * 0.8
            ax.set_ylim([y_min, y_max])
    
    # 添加全局图例
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    return fig

# 这个版本已经移除了上边界和右边界，并符合 ICLR/ICML 风格
# 你可以使用 `plot_multi_dataset_fairness_improvement` 来绘制你的数据集

# 如果需要调整字体大小、颜色或者其他格式，告诉我！
