import pandas as pd
import numpy as np
import os
# 当前时刻：1 表示开启的瞬时动作，0 表示关闭的瞬时动作。
# 后续时刻：2 表示设备保持开启状态，-1 表示设备保持关闭状态

start_time = '2013-10-22 14:41:00'
end_time = '2017-01-02 13:17:00'
file_path = 'D:/Ai/NILM/DALEdataset/house_1/'

for file_name in os.listdir(file_path):
    if('button_press' in file_name):
        channel_num = file_name.split('_')[1]  # 提取通道号
        file = os.path.join(file_path,file_name)
        df = pd.read_csv(file,sep='\s+',header=None, names=['timestamp','event'])
        df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
        df.set_index('timestamp',inplace=True)

        full_range = pd.date_range(start=start_time, end=end_time, freq='min')
        full_df = pd.DataFrame(index=full_range)

        full_df = full_df.join(df['event'])
        full_df['event'] = full_df['event'].fillna(-1)

        current_state = -1 # 当前状态，假设初始设备处于关闭状态
        is_on = False
        for idx, row in full_df.iterrows():
            if row['event'] == 1 and not is_on:
                is_on = True
            elif row['event'] == 0 and is_on:
                is_on = False
            elif row['event'] == -1 and is_on:
                current_state = 2
                full_df.at[idx, 'event'] = current_state
            elif row['event'] == -1 and not is_on:
                current_state = -1
                full_df.at[idx, 'event'] = current_state
        full_df['event'] = full_df['event'].astype(int)
        full_df.rename(columns={'event': f'channel{channel_num}_event'}, inplace=True)
        output_file = os.path.join('D:/Ai/NILM/DALE_pressDataset/', f'{file_name}.csv')
        full_df.to_csv(output_file,index_label='timestamp')
        print(f"Processed and saved: {output_file}")
