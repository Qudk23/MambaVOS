from PIL import Image
import os

# 输入文件夹和输出文件夹的路径
input_folder = '/home/qdk/code/MambaVOS/dataset/TrainSet/YoutubeVOS/youtubevos/Frame'
output_folder = '/home/qdk/code/YoutubeVOS/youtubevos/Frame'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 获取输入文件夹中的所有文件
files = os.listdir(input_folder)
files.sort()
# 用于计数保存的图片数量
count = 0

# 循环遍历文件并保存每隔30张图片
for i, file in enumerate(files):
    if i % 30 == 0:
        # 打开图片
        image = Image.open(os.path.join(input_folder, file))
        
        # 构建输出文件的路径
        output_file = os.path.join(output_folder, file)
        
        # 保存图片
        image.save(output_file)
        
        count += 1

print(f'Saved {count} images to {output_folder}')
