import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_fairness_improvement(
    folds,
    original_DR,      # 与 folds 对应的原始 DR 值列表
    original_DP,      # 与 folds 对应的原始 DP 值列表
    original_EO,      # 与 folds 对应的原始 EO 值列表
    original_PQP,     # 与 folds 对应的原始 PQP 值列表
    stop_when_no_data=3,
    min_action=1,
    baseline=0.0,
    figsize=(10, 6),
    fill_alpha=0.2,
    fill_color='b'
):
    """
    绘制 2×2 子图，分别展示 DR、DP、EO、PQP 这四种 fairness 指标的
    “相对于原始指标值 (original) 的差值”随 action_number 变化的均值±标准差曲线。
    
    每张子图的具体做法：
      1) 将每个 fold 中的 new_XXX 列减去对应的 original_XXX，以得到差值。
      2) 从 action=1 遍历到最大 action_number，当有 stop_when_no_data 个 fold 没数据时停止。
      3) 对剩下有数据的 fold 进行均值和标准差计算并画图。
      4) 在 y=baseline 处添加一条参考线。

    参数：
    --------
    1) folds : list of pd.DataFrame
       - 每个 DataFrame 必须包含下列列：'action_number', 'new_DR', 'new_DP', 'new_EO', 'new_PQP'
         例如：fold1, fold2, fold3, fold4, ...
    
    2) original_DR, original_DP, original_EO, original_PQP : list of float
       - 分别对应 DR、DP、EO、PQP 的原始值列表，每个列表长度应与 folds 相同。
       - 如：original_DR = [fold1_original_DR, fold2_original_DR, fold3_original_DR, ...]

    3) stop_when_no_data : int, 默认=3
       - 当遍历 action_number 时，如果有 >= stop_when_no_data 个 fold 没数据，就停止遍历。

    4) min_action : int, 默认=1
       - 从哪个 action_number 开始遍历。

    5) baseline : float, 默认=0.0
       - 在图中绘制一条 y=baseline 的水平线，便于参考。

    6) figsize : tuple, 默认=(10, 6)
       - 整个图表的大小，会被分成 2×2 四个子图。

    7) fill_alpha : float, 默认=0.2
       - 均值曲线上下方“±标准差”区域的透明度。

    8) fill_color : str, 默认='b'
       - 均值线和填充区域的颜色。

    返回：
    --------
    None
    """

    # ============== 1) 输入检查 ==============
    # 确保 folds 的数量与各 original_xxx 列表长度一致
    num_folds = len(folds)
    if not (len(original_DR) == len(original_DP) == len(original_EO) == len(original_PQP) == num_folds):
        raise ValueError("original_DR, original_DP, original_EO, original_PQP 的长度都必须与 folds 相同。")

    # 我们需要绘制的四个指标信息打包在一起，以便循环
    measures_info = [
        ("DR",  "new_DR",  original_DR),
        ("DP",  "new_DP",  original_DP),
        ("EO",  "new_EO",  original_EO),
        ("PQP", "new_PQP", original_PQP),
    ]

    # ============== 2) 准备图形：2×2 子图 ==============
    fig = plt.figure(figsize=figsize)

    # 遍历四个指标，每个指标做一张子图
    for i, (measure_name, measure_col, original_list) in enumerate(measures_info, start=1):
        
        # 2.1) 把 action_number 转成数值，并把 new_XXX 减去原始值
        for df, orig_val in zip(folds, original_list):
            df['action_number'] = pd.to_numeric(df['action_number'], errors='coerce')
            # 做差值： new_XXX -= original_XXX
            df[measure_col] = df[measure_col] - orig_val

        # 2.2) 找到所有 fold 中最大的 action_number，用于确定遍历上限
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

        # 2.3) 收集各 action_number 对应的差值
        measure_values = {}
        for action in range(min_action, overall_max_action + 1):
            current_list = []
            count_no_data = 0

            for df in folds:
                row = df.loc[df['action_number'] == action, measure_col]
                if row.empty:
                    count_no_data += 1
                else:
                    # 如果出现多行，就取第一行
                    current_list.append(row.values[0])
            
            if count_no_data >= stop_when_no_data:
                # 当有太多 fold 没数据时，就停止
                break
            
            measure_values[action] = current_list

        # 如果收集不到任何 action_number，就跳过
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

        # ============== 3) 在子图中绘图 ==============
        ax = fig.add_subplot(2, 2, i)  # 2 行 2 列，第 i 个子图
        ax.set_title(f"{measure_name} Difference from Original")

        # 3.1) baseline 参考线
        ax.axhline(y=baseline, color='black', linewidth=2, linestyle='-', 
                   label=f'Baseline (y={baseline})')

        # 3.2) 均值线
        ax.plot(action_range, means, color=fill_color, label=f'Mean {measure_name} Gap')

        # 3.3) 均值±标准差区域
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

    # 调整子图之间的布局
    plt.tight_layout()
    plt.show()
