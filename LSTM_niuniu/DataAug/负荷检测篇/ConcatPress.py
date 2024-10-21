import pandas as pd
import os

def combine_csv_files(input_folder, output_file):
    combined_df = pd.DataFrame()
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            file = os.path.join(input_folder, file_name)
            df = pd.read_csv(file, index_col='timestamp', parse_dates=True)
            
            # 将当前文件的内容与已有的 DataFrame 进行拼接
            combined_df = combined_df.join(df, how='outer')
    
    # 保存最终拼接后的文件
    combined_df.to_csv(output_file, index_label='timestamp')
    print(f"All channels combined and saved to: {output_file}")

# 调用该函数进行拼接
input_folder = 'D:/Ai/NILM/DALE_pressDataset/'
output_file = 'D:/Ai/NILM/DALE_pressDataset/all_channels_combined.csv'
combine_csv_files(input_folder, output_file)
