
import os
import pandas as pd

# 指定文件夹路径
folder_path = "D:/Ai/NILM/Dataset/"
output_folder = "D:/Ai/NILM/Processed/"
pressfolder_path = "D:/Ai/NILM/DALE_pressDataset/channel_11_button_press.dat.csv"
#for file_name in os.listdir("D:/Ai/NILM/ProcessedDataset/"):
#    file_path = os.path.join("D:/Ai/NILM/ProcessedDataset/",file_name)
#    df = pd.read_csv(file_path)
#    print(f"{file_name} 的行数为 {df.shape[0]}")

df = pd.read_csv(pressfolder_path)
print(df.columns)