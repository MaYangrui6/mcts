import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 读取数据
    df = pd.read_csv("/home/ubuntu/project/LSTM+Attention/information/train_with_predict_ratio.csv")

    df = df[df['cost_reduction_ratio'] != 0]
    # 计算相对误差并转换为百分比
    df['relative_error'] = (df['cost_reduction_ratio'] - df['predict_ratio']) / df['cost_reduction_ratio']

    # 计算相对误差的均值和标准误差
    mean_relative_error = df['relative_error'].mean()
    sem_relative_error = df['relative_error'].sem()

    # 计算MAE（绝对误差）的均值和标准误差
    mean_mae = df['relative_error'].abs().mean()
    sem_mae = df['relative_error'].abs().sem()

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制相对误差的柱状图
    ax.bar('Relative error (%)', mean_relative_error, yerr=sem_relative_error, capsize=10, alpha=0.7, color='blue', label='RE')

    # 绘制MAE（绝对误差）的柱状图
    ax.bar('MAE', mean_mae, yerr=sem_mae, capsize=10, alpha=0.7, color='green', label='MAE')

    # 设置图例和标签
    ax.set_ylabel('value')
    ax.set_title('Comparison of relative and mean absolute error')
    ax.legend()

    plt.show()
