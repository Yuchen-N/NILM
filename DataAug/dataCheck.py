import numpy as np
import pandas as pd
import os

folder_path = "D:/Ai/NILM/Dataset/"
output_folder = "D:/Ai/NILM/ProcessedDataset/"

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 处理时间范围
start_time = '2013-10-22 14:41:00'
end_time = '2017-01-02 13:17:00'

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    
    # 读取 CSV 文件并确保 timestamp 被解析为 datetime 格式，并设置为索引
    df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
    
    # 根据时间范围过滤数据
    df_filtered = df[(df.index >= start_time) & (df.index < end_time)]
    
    # 输出文件路径
    output_file_path = os.path.join(output_folder, f'{file_name}')
    
    # 保存处理后的数据，并保持 timestamp 作为索引
    df_filtered.to_csv(output_file_path, index=True)
    
    print(f"{file_name} 已处理并保存到 {output_file_path}")
