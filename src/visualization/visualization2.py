import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_results(
    folds,
    original_accuracy,  
    original_DR, 
    original_DP, 
    original_EO, 
    original_PQP,
    stop_when_no_data=3,
    min_action=1,
    figsize=None,  # 自动调整
    fill_alpha=0.2,
    fill_color='b',
    red_alpha=0.3
):
    """
    一次性绘制 Accuracy / DR / DP / EO / PQP 的 1×5 子图，展示各项指标在不同 action_number 下的
    均值与标准差，并在图中用淡红色虚线标出 5 个原始数值。

    参数说明：
    --------
    1) folds : list of pd.DataFrame
       - 每个 DataFrame 必须含有下列列：
         'action_number', 'new_accuracy', 'new_DR', 'new_DP', 'new_EO', 'new_PQP'.

    2) original_accuracy : list of float
       - 长度为 5，表示 Accuracy 原始值。

    3) original_DR, original_DP, original_EO, original_PQP : list of float
       - 长度为 5，分别对应 DR, DP, EO, PQP 指标的原始值。

    其余参数与之前相同...
    """
    # 更新需要绘制的指标
    measures_info = [
        ("Accuracy", "new_accuracy", original_accuracy),
        ("DR",  "new_DR",  original_DR),
        ("DP",  "new_DP",  original_DP),
        ("EO",  "new_EO",  original_EO),
        ("PQP", "new_PQP", original_PQP),
    ]

    num_subplots = len(measures_info)  # 计算子图数量
    num_rows = 1  # 设为 1 行
    num_cols = num_subplots  # 设为 5 列

    # 自动调整图像大小
    if figsize is None:
        figsize = (num_subplots * 5, 5)

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
        ax = plt.subplot(num_rows, num_cols, i)  # 1×5 布局，第 i 个子图
        ax.set_title(f"{measure_name} vs. Action Number")

        # 4.1) 在子图中用淡红色虚线绘制 orig_list 中的每个值
        for idx, val in enumerate(orig_list):
            label = "Original Values" if idx == 0 else None  # 只给第一条线加上 label，避免图例重复
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

    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.3)  # 增加子图之间的水平间距
    plt.show()




# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# def plot_results(
#     folds,
#     original_accuracy,  # 新增 accuracy 变量
#     original_DR, 
#     original_DP, 
#     original_EO, 
#     original_PQP,
#     stop_when_no_data=3,
#     min_action=1,
#     figsize=(12, 10),  # 调整高度，避免 3×2 布局显得太拥挤
#     fill_alpha=0.2,
#     fill_color='b',
#     red_alpha=0.3
# ):
#     """
#     一次性绘制 Accuracy / DR / DP / EO / PQP 的 3×2 子图，展示各项指标在不同 action_number 下的
#     均值与标准差，并在图中用淡红色虚线标出 5 个原始数值。

#     参数说明：
#     --------
#     1) folds : list of pd.DataFrame
#        - 每个 DataFrame 必须含有下列列：
#          'action_number', 'new_accuracy', 'new_DR', 'new_DP', 'new_EO', 'new_PQP'.
#        - 示例：fold1, fold2, fold3, fold4, ...

#     2) original_accuracy : list of float
#        - 长度为 5，表示 Accuracy 原始值。

#     3) original_DR, original_DP, original_EO, original_PQP : list of float
#        - 长度为 5，分别对应 DR, DP, EO, PQP 指标的原始值。

#     其余参数与之前相同...
#     """
#     # 更新需要绘制的指标
#     measures_info = [
#         ("Accuracy", "new_accuracy", original_accuracy),
#         ("DR",  "new_DR",  original_DR),
#         ("DP",  "new_DP",  original_DP),
#         ("EO",  "new_EO",  original_EO),
#         ("PQP", "new_PQP", original_PQP),
#     ]

#     num_subplots = len(measures_info)  # 计算有多少个子图
#     num_rows = (num_subplots + 1) // 2  # 计算行数（保证 2 列排布）
    
#     plt.figure(figsize=figsize)

#     for i, (measure_name, measure_col, orig_list) in enumerate(measures_info, start=1):
#         # ---- 1) 将 action_number 转为数值，并检查数据 ----
#         for df in folds:
#             df['action_number'] = pd.to_numeric(df['action_number'], errors='coerce')

#         # ---- 2) 找到该指标对应的 overall_max_action ----
#         max_actions = []
#         for df in folds:
#             if measure_col not in df.columns:
#                 raise ValueError(f"列 {measure_col} 在某个 fold 中不存在，请检查输入数据。")
#             if not df.empty:
#                 max_val = df['action_number'].max()
#                 if not pd.isna(max_val):
#                     max_actions.append(max_val)

#         if len(max_actions) == 0:
#             print(f"警告：所有 Fold 对于 {measure_col} 都是空表或无效 action_number，跳过该子图。")
#             continue
#         overall_max_action = int(np.nanmax(max_actions))

#         # ---- 3) 收集 measure 数据，遇到 stop_when_no_data 个 fold 没数据就停止 ----
#         measure_values = {}
#         for action in range(min_action, overall_max_action + 1):
#             current_list = []
#             count_no_data = 0

#             for df in folds:
#                 row = df.loc[df['action_number'] == action, measure_col]
#                 if row.empty:
#                     count_no_data += 1
#                 else:
#                     # 如果出现多行数据（通常不会），默认取第一个
#                     current_list.append(row.values[0])

#             # 如果有 >= stop_when_no_data 个 fold 没有数据，就停止
#             if count_no_data >= stop_when_no_data:
#                 break

#             measure_values[action] = current_list

#         action_range = sorted(measure_values.keys())
#         if len(action_range) == 0:
#             print(f"警告：{measure_name} 未找到满足条件 (stop_when_no_data < {stop_when_no_data}) 的 action_number，跳过。")
#             continue

#         means = []
#         stds = []
#         for action in action_range:
#             vals = measure_values[action]
#             means.append(np.mean(vals))
#             stds.append(np.std(vals))

#         means = np.array(means)
#         stds = np.array(stds)

#         # ---- 4) 在子图上绘制 ----
#         ax = plt.subplot(num_rows, 2, i)  # 3 行 2 列，第 i 个子图
#         ax.set_title(f"{measure_name} vs. Action Number")

#         # 4.1) 在子图中用淡红色虚线绘制 orig_list 中的每个值
#         for idx, val in enumerate(orig_list):
#             label = "Original Values" if idx == 0 else None  # 只给第一条线加上 label，避免图例重复
#             ax.axhline(
#                 y=val,
#                 color='red',
#                 linestyle='--',
#                 linewidth=1.5,
#                 alpha=red_alpha,
#                 label=label
#             )

#         # 4.2) 均值线
#         ax.plot(action_range, means, color=fill_color, label=f"Mean {measure_name}")

#         # 4.3) 均值±标准差填充
#         ax.fill_between(
#             action_range,
#             means - stds,
#             means + stds,
#             alpha=fill_alpha,
#             color=fill_color,
#             label='±1 std dev'
#         )

#         ax.set_xlabel("Action Number")
#         ax.set_ylabel(f"{measure_name} Value")
#         ax.grid(True)
#         ax.legend()

#     plt.tight_layout()
#     plt.show()
