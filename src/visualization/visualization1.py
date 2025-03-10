import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_fairness_improvement(
    folds,
    original_accuracy,  # 新增 Accuracy 变量
    original_DR,      
    original_DP,      
    original_EO,      
    original_PQP,     
    stop_when_no_data=3,
    min_action=1,
    baseline=0.0,
    figsize=(12, 10),  # 调整高度，避免 3×2 布局显得太拥挤
    fill_alpha=0.2,
    fill_color='b'
):
    """
    绘制 3×2 子图，分别展示 Accuracy、DR、DP、EO、PQP 指标的
    “相对于原始指标值的改善差值”随 action_number 变化的均值±标准差曲线。

    参数：
    --------
    1) folds : list of pd.DataFrame
       - 每个 DataFrame 必须包含：
         'action_number', 'new_accuracy', 'new_DR', 'new_DP', 'new_EO', 'new_PQP'
    
    2) original_accuracy, original_DR, original_DP, original_EO, original_PQP : list of float
       - 长度需与 `folds` 相同。

    其余参数与之前相同...
    """

    # ============== 1) 输入检查 ==============
    num_folds = len(folds)
    if not (len(original_accuracy) == len(original_DR) == len(original_DP) == len(original_EO) == len(original_PQP) == num_folds):
        raise ValueError("original_accuracy, original_DR, original_DP, original_EO, original_PQP 长度必须与 folds 相同。")

    # 需要绘制的 5 个指标
    measures_info = [
        ("Accuracy", "new_accuracy", original_accuracy),
        ("DR",  "new_DR",  original_DR),
        ("DP",  "new_DP",  original_DP),
        ("EO",  "new_EO",  original_EO),
        ("PQP", "new_PQP", original_PQP),
    ]

    num_subplots = len(measures_info)  # 计算子图数量
    num_rows = (num_subplots + 1) // 2  # 计算行数，确保合理排版
    
    fig = plt.figure(figsize=figsize)

    # 遍历 5 个指标，每个指标绘制 1 张子图
    for i, (measure_name, measure_col, original_list) in enumerate(measures_info, start=1):
        
        # 2.1) 计算 `new_XXX - original_XXX`
        for df, orig_val in zip(folds, original_list):
            df['action_number'] = pd.to_numeric(df['action_number'], errors='coerce')
            df[measure_col] = df[measure_col] - orig_val

        # 2.2) 找到所有 fold 中最大的 action_number
        max_actions = []
        for df in folds:
            if not df.empty:
                max_val = df['action_number'].max()
                if pd.notna(max_val):
                    max_actions.append(max_val)
        
        if len(max_actions) == 0:
            print(f"警告：所有 fold 的 {measure_col} 数据都是空的或无效，跳过该子图。")
            continue
        overall_max_action = int(np.nanmax(max_actions))

        # 2.3) 收集各 action_number 的改善值
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

        # 2.4) 计算均值与标准差
        means = []
        stds = []
        for action in action_range:
            vals = measure_values[action]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        means = np.array(means)
        stds = np.array(stds)

        # ============== 3) 绘图 ==============
        ax = fig.add_subplot(num_rows, 2, i)  # 3×2 布局，第 i 个子图
        ax.set_title(f"{measure_name} Improvement from Original")

        # 3.1) baseline 参考线
        ax.axhline(y=baseline, color='black', linewidth=2, linestyle='-', label=f'Baseline (y={baseline})')

        # 3.2) 均值曲线
        ax.plot(action_range, means, color=fill_color, label=f'Mean {measure_name} Gap')

        # 3.3) 均值 ± 标准差
        ax.fill_between(
            action_range,
            means - stds,
            means + stds,
            alpha=fill_alpha,
            color=fill_color,
            label='±1 std dev'
        )

        ax.set_xlabel("Action Number")
        ax.set_ylabel(f"{measure_name} - Original")
        ax.grid(True)
        ax.legend()

    # 调整子图之间的间距
    plt.tight_layout()
    plt.show()
