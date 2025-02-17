import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_results(
    folds,
    original_DR, 
    original_DP, 
    original_EO, 
    original_PQP,
    stop_when_no_data=3,
    min_action=1,
    figsize=(12, 8),
    fill_alpha=0.2,
    fill_color='b',
    red_alpha=0.3
):
    """
    一次性绘制 DR / DP / EO / PQP 的 2×2 子图，展示各项指标在不同 action_number 下的
    均值与标准差，并在图中用淡红色虚线标出 5 个原始数值。

    参数说明：
    --------
    1) folds : list of pd.DataFrame
       - 每个 DataFrame 必须含有下列列：
         'action_number', 'new_DR', 'new_DP', 'new_EO', 'new_PQP'.
       - 示例：fold1, fold2, fold3, fold4, ...

    2) original_DR, original_DP, original_EO, original_PQP : list of float
       - 各长度为 5（如果你有 5 个 original value），分别对应 DR, DP, EO, PQP 指标的原始值。
       - 例如 original_DR = [dr1, dr2, dr3, dr4, dr5]（这 5 个值分别是哪几个 fold 的或其它含义，
         取决于你的场景），会在 DR 的子图中画出 5 条淡红色虚线。

    3) stop_when_no_data : int, 默认=3
       - 当遍历 action_number 时，如果有 >= stop_when_no_data 个 fold 没有该 action_number 的数据，就停止。

    4) min_action : int, 默认=1
       - 从 action_number = min_action 开始遍历，一直到无效或停止条件达成为止。

    5) figsize : tuple, 默认=(12, 8)
       - 整个图形的大小，会包含 4 个子图 (2×2)。

    6) fill_alpha : float, 默认=0.2
       - 均值曲线上下方“±1 标准差”区域的透明度。

    7) fill_color : str, 默认='b'
       - 均值曲线以及填充区域的颜色。

    8) red_alpha : float, 默认=0.3
       - 红色虚线的透明度 (0~1 之间)，用于在子图中以淡淡的红色显示原始值。

    返回：
    --------
    None
    """
    # 首先检查 folds 与 4 个 original_xxx 列表的长度是否匹配你的需求
    # 如果你真的只有 4~5 个 folds，却有 5 个 original 值，这里按场景自定义检查
    # 不过你说“针对每个指标有 5 个 original value”，那就不检查 folds 长度了

    # 将要绘制的 4 个指标的信息打包在一起，以方便循环
    measures_info = [
        ("DR",  "new_DR",  original_DR),
        ("DP",  "new_DP",  original_DP),
        ("EO",  "new_EO",  original_EO),
        ("PQP", "new_PQP", original_PQP),
    ]

    # 准备子图
    plt.figure(figsize=figsize)

    for i, (measure_name, measure_col, orig_list) in enumerate(measures_info, start=1):
        # ---- 1) 将 action_number 转为数值，并检查数据 ----
        for df in folds:
            df['action_number'] = pd.to_numeric(df['action_number'], errors='coerce')

        # ---- 2) 找到该指标对应的 overall_max_action ----
        max_actions = []
        for df in folds:
            if measure_col not in df.columns:
                raise ValueError(f"列 {measure_col} 在某个 fold 中不存在，请检查输入数据。")
            if not df.empty:
                max_val = df['action_number'].max()
                if not pd.isna(max_val):
                    max_actions.append(max_val)

        if len(max_actions) == 0:
            print(f"警告：所有 Fold 对于 {measure_col} 都是空表或无效 action_number，跳过该子图。")
            continue
        overall_max_action = int(np.nanmax(max_actions))

        # ---- 3) 收集 measure 数据，遇到 stop_when_no_data 个 fold 没数据就停止 ----
        measure_values = {}
        for action in range(min_action, overall_max_action + 1):
            current_list = []
            count_no_data = 0

            for df in folds:
                row = df.loc[df['action_number'] == action, measure_col]
                if row.empty:
                    count_no_data += 1
                else:
                    # 如果出现多行数据（通常不会），默认取第一个
                    current_list.append(row.values[0])

            # 如果有 >= stop_when_no_data 个 fold 没有数据，就停止
            if count_no_data >= stop_when_no_data:
                break

            measure_values[action] = current_list

        action_range = sorted(measure_values.keys())
        if len(action_range) == 0:
            print(f"警告：{measure_name} 未找到满足条件 (stop_when_no_data < {stop_when_no_data}) 的 action_number，跳过。")
            continue

        means = []
        stds = []
        for action in action_range:
            vals = measure_values[action]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        means = np.array(means)
        stds = np.array(stds)

        # ---- 4) 在子图上绘制 ----
        ax = plt.subplot(2, 2, i)  # 2 行 2 列，第 i 个子图
        ax.set_title(f"{measure_name} vs. Action Number")

        # 4.1) 在子图中用淡红色虚线绘制 orig_list 中的每个值
        for idx, val in enumerate(orig_list):
            # 只给第一条线加上 label，避免图例重复
            label = "Original Values" if idx == 0 else None
            ax.axhline(
                y=val,
                color='red',
                linestyle='--',
                linewidth=1.5,
                alpha=red_alpha,
                label=label
            )

        # 4.2) 均值线
        ax.plot(action_range, means, color=fill_color, label=f"Mean {measure_name}")

        # 4.3) 均值±标准差填充
        ax.fill_between(
            action_range,
            means - stds,
            means + stds,
            alpha=fill_alpha,
            color=fill_color,
            label='±1 std dev'
        )

        ax.set_xlabel("Action Number")
        ax.set_ylabel(f"{measure_name} Value")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()
