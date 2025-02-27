import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_fairness_improvement(
    folds,
    original_DR,         
    original_DP,         
    original_EO,         
    original_PQP,        
    original_recall,     # 新增：与 folds 对应的原始 recall 值列表
    original_precision,  # 新增：与 folds 对应的原始 precision 值列表
    original_sufficiency,# 新增：与 folds 对应的原始 sufficiency 值列表
    stop_when_no_data=3,
    min_action=1,
    baseline=0.0,
    figsize=(15, 10),    # 修改默认大小以适应更多子图
    fill_alpha=0.2,
    fill_color='b'
):
    """
    绘制多个子图，分别展示 DR、DP、EO、PQP、Recall、Precision、Sufficiency 这七种指标的
    "相对于原始指标值 (original) 的差值"随 action_number 变化的均值±标准差曲线。
    
    每张子图的具体做法：
      1) 将每个 fold 中的 new_XXX 列减去对应的 original_XXX，以得到差值。
      2) 从 action=1 遍历到最大 action_number，当有 stop_when_no_data 个 fold 没数据时停止。
      3) 对剩下有数据的 fold 进行均值和标准差计算并画图。
      4) 在 y=baseline 处添加一条参考线。

    参数：
    --------
    1) folds : list of pd.DataFrame
       - 每个 DataFrame 必须包含下列列：
         'action_number', 'new_DR', 'new_DP', 'new_EO', 'new_PQP', 
         'new_recall', 'new_precision', 'new_sufficiency'
    
    2) original_DR, original_DP, original_EO, original_PQP,
       original_recall, original_precision, original_sufficiency : list of float
       - 分别对应各指标的原始值列表，每个列表长度应与 folds 相同。

    其他参数保持不变
    """
    # ============== 1) 输入检查 ==============
    num_folds = len(folds)
    if not (len(original_DR) == len(original_DP) == len(original_EO) == 
            len(original_PQP) == len(original_recall) == len(original_precision) == 
            len(original_sufficiency) == num_folds):
        raise ValueError("所有 original_xxx 列表的长度都必须与 folds 相同。")

    # 打包所有指标信息
    measures_info = [
        ("DR",          "new_DR",          original_DR),
        ("DP",          "new_DP",          original_DP),
        ("EO",          "new_EO",          original_EO),
        ("PQP",         "new_PQP",         original_PQP),
        ("Recall",      "new_recall",      original_recall),
        ("Precision",   "new_precision",   original_precision),
        ("Sufficiency", "new_sufficiency", original_sufficiency),
    ]

    # ============== 2) 准备图形布局 ==============
    n_metrics = len(measures_info)
    n_cols = 3  # 每行3个子图
    n_rows = (n_metrics + n_cols - 1) // n_cols  # 向上取整得到行数
    
    fig = plt.figure(figsize=figsize)

    # 遍历所有指标，每个指标做一张子图
    for i, (measure_name, measure_col, original_list) in enumerate(measures_info, start=1):
        
        # 2.1) 把 action_number 转成数值，并把 new_XXX 减去原始值
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

        # ============== 3) 在子图中绘图 ==============
        ax = fig.add_subplot(n_rows, n_cols, i)
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



# 
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# def plot_results(
#     folds,
#     original_DR, 
#     original_DP, 
#     original_EO, 
#     original_PQP,
#     original_recall,
#     original_precision,
#     original_sufficiency,
#     stop_when_no_data=3,
#     min_action=1,
#     figsize=(15, 10),
#     fill_alpha=0.2,
#     fill_color='b',
#     red_alpha=0.3
# ):
#     """
#     一次性绘制 DR / DP / EO / PQP / Recall / Precision / Sufficiency 的多子图，
#     展示各项指标在不同 action_number 下的均值与标准差，并在图中用淡红色虚线标出原始数值。

#     参数说明：
#     --------
#     原有参数保持不变，新增：
#     original_recall : list of float
#         - Recall指标的原始值列表
#     original_precision : list of float
#         - Precision指标的原始值列表
#     original_sufficiency : list of float
#         - Sufficiency指标的原始值列表

#     figsize默认值改为(15, 10)以适应更多子图
#     """
#     # 将要绘制的7个指标的信息打包在一起
#     measures_info = [
#         ("DR",          "new_DR",          original_DR),
#         ("DP",          "new_DP",          original_DP),
#         ("EO",          "new_EO",          original_EO),
#         ("PQP",         "new_PQP",         original_PQP),
#         ("Recall",      "new_recall",      original_recall),
#         ("Precision",   "new_precision",   original_precision),
#         ("Sufficiency", "new_sufficiency", original_sufficiency),
#     ]

#     # 计算子图布局的行数和列数
#     n_metrics = len(measures_info)
#     n_cols = 3  # 每行3个子图
#     n_rows = (n_metrics + n_cols - 1) // n_cols  # 向上取整得到行数

#     # 准备子图
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
#                     current_list.append(row.values[0])

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
#         ax = plt.subplot(n_rows, n_cols, i)
#         ax.set_title(f"{measure_name} vs. Action Number")

#         # 4.1) 绘制原始值
#         for idx, val in enumerate(orig_list):
#             label = "Original Values" if idx == 0 else None
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