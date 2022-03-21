import os
import sys

os.chdir('dataset/imgs/jinglingseg1/masks')
print(os.getcwd())

for count, f in enumerate(os.listdir()):
    f_name, f_ext = os.path.splitext(f)
    print(f_name)
    f_name = 'mask_img' + str(count)

    new_name = f'{f_name}{f_ext}'
    os.rename(f, new_name)