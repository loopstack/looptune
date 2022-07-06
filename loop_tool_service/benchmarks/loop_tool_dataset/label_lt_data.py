from cgi import test
from math import ceil
import sys
import os 
import loop_tool as lt
import random
import shutil
import pickle
import pdb
from pathlib import Path


def split_data(path_to_data):
    
    file_names = [ f for f in os.listdir(path_to_data)]
    
    for file_name in file_names:
        file_path = os.path.join(path_to_data, file_name)
        print(file_path)

        tree = None
        with open(file_path, "r") as file:
            ir = lt.deserialize(file.read())
            tree = lt.LoopTree(ir)
        with open(file_path, "a") as file:
            file.write(str(tree.FLOPS()))
    

def main():
    
    print(sys.argv)
    if len(sys.argv) != 2:
        print('Format: label_lt.data.py path_to_data')
        return 

    path_to_data = sys.argv[1]
    split_data(path_to_data)


if __name__ == '__main__':
    main()






