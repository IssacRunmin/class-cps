
import os
import re
from typing import List

import glob

from PIL import Image

def list_file(file_path: str, file_format: tuple = None) -> List[str]:
    EXTENSION_RE = re.compile(r'\.([a-zA-Z\d]*?)$')
    """
    根据文件路径列出所有扩展名相关的文件，返回绝对路径列表

    examples:

    list_file('img/test.png'):                      匹配文件，返回单文件列表
    list_file('img'):                               ./img下的所有文件；
    list_file('img', ('png', 'jpg')):             ./img下所有png, jpg 后缀名格式文件；
    list_file('./img/test*.png'):                   ./img下通配test*.png的文件
    list_file('./img/test*', ('png', 'jpg')):     ./img下通配test*的图像(png, jpg)文件

    :param file_path:       文件夹路径或通配符路径
    :param file_format:     扩展名tuple
    :return:                文件路径list
    """
    file_list = []
    if os.path.isfile(file_path):                  # is a file
        file_list.append(os.path.abspath(file_path))
    else:               # not a file, maybe a match, dir or no exist
        dir_files = []
        dir_path = os.path.abspath(file_path)
        if "*" in os.path.basename(file_path):     # has wildcard match
            dir_files = glob.glob(file_path)
        elif os.path.isdir(file_path):             # is a directory
            dir_files = os.listdir(file_path)
        for file_name in dir_files:
            if file_format is None:             # no need to check
                file_list.append(os.path.join(dir_path, file_name))
            else:
                res = EXTENSION_RE.search(file_name)  # extract ext
                if res and res.group(1) in file_format:  # check ext
                    file_list.append(os.path.join(dir_path, file_name))
    return file_list