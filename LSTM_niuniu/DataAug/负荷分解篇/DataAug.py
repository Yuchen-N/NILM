import pandas as pd
import os

file_name = "D:/Ai/NILM/DALEdataset/house_1/"
channel_data = {}

# 先加载main.dat文件，确定时间范围
main_path = os.path.join(file_name, 'mains.dat')
df_main = pd.read_csv(main_path, sep='\s+', header=None, names=['timestamp', 'ActivePower', 'ApparentPower', 'RMS'])
df_main['timestamp'] = pd.to_datetime(df_main['timestamp'], unit='s')
df_main.set_index('timestamp', inplace=True)

# 设定main.dat文件的时间间隔范围
df_main_resampled = df_main.resample('1min').mean()
start_time = df_main_resampled.index.min()
end_time = df_main_resampled.index.max()

# 循环加载并重采样其他通道数据
for i in range(1, 54):
    file_path = os.path.join(file_name, f'channel_{i}.dat')
    df = pd.read_csv(file_path, sep='\s+', header=None, names=['timestamp', f'power_channel_{i}'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)

    # 重采样之前填补不连续的数据段，处理间隔问题
    df = df.asfreq('1s', method=None)  # 使用1秒的频率，且不插值
    df.fillna(0, inplace=True)  # 对缺失值填充0，表示设备关闭时的功率为0
    
    # 重采样并按照main.dat的时间范围进行截取
    df_resampled = df.resample('1min').mean()
    df_resampled = df_resampled.loc[start_time:end_time]
    
    channel_data[f'channel_{i}'] = df_resampled

# 将main.dat的重采样数据也加入channel_data
df_main_resampled = df_main_resampled.loc[start_time:end_time]
channel_data['main'] = df_main_resampled

# 保存所有通道的重采样数据
for channel, df in channel_data.items():
    save_path = os.path.join('D:/Ai/NILM/Dataset/', f'{channel}_resampled.csv')
    df.to_csv(save_path)

print(f"数据已成功保存到 {file_name} 目录下。")