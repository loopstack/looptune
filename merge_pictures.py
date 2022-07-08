from builtins import all
from curses import color_content
import profile
import sys
from math import ceil, sqrt
import os
import pandas as pd

import pdb

from PIL import Image


def merge_vertical(file_names, file_out) :
    imgs = []    
    img_x, img_y = 0, 0
    for i, file_name in enumerate(file_names):
        img = Image.open(file_name)
        imgs.append(img)
        img_x = max(img_x, img.size[0])
        img_y += img.size[1]

    
    new_im = Image.new('RGB', (img_x, img_y), (250,250,250))

    for i, img in enumerate(imgs):
        new_im.paste(img, (0, i * img.size[1]))

    new_im.save(file_out, "png")


def merge_box(file_names, file_out) :
    imgs = []    
    img_x, img_y = 0, 0
    img_out_side = ceil(sqrt(len(file_names)))

    for i, file_name in enumerate(file_names):
        img = Image.open(file_name)
        imgs.append(img)

        img_x = max(img_x, imgs[i].size[0])
        img_y = max(img_y, imgs[i].size[1])        


    new_im = Image.new('RGB', (img_out_side * img_x, img_out_side * img_y), (250,250,250))

    x_i, y_i = 0, 0
    for i, img in enumerate(imgs):
        new_im.paste(imgs[i], (x_i * img_x, y_i * img_y))
        x_i += 1
        if x_i == img_out_side:
            x_i = 0
            y_i += 1

    new_im.save(file_out, "png")



def get_pictures(pictures_dir, suffix):
    all_files = os.listdir(pictures_dir)
    if len(all_files):
        selected_files = [ os.path.join(pictures_dir, filename) for filename in all_files if filename.endswith(suffix) ]
        selected_files.sort()
        return selected_files
    else:
        print('No files to visualize')
        exit()



def main():
    
    print(sys.argv)
    if len(sys.argv) != 5:
        print('Format: merge_pictures.py how pictures_dir suffix out_file_path')
        return 

    how = sys.argv[1]
    pictures_dir = sys.argv[2]
    suffix =  sys.argv[3]
    out_file_path =  sys.argv[4]
    file_names = get_pictures(pictures_dir, suffix)
        
    if how == 'vertical':
        merge_vertical(file_names, out_file_path)
    elif how == 'box':
        merge_box(file_names, out_file_path)
    else:
        print('Dont know how to merge')
    

if __name__ == '__main__':
    main()