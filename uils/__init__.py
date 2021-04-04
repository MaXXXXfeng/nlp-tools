#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Author  : Max
# @Time    : 2021/4/4
# @Desc:

def check_file_suffix(fname, accept_suffix=['.txt']):
    '''检查文件 格式是否符合要求'''
    for item in accept_suffix:
        if fname.endswith(item):
            return True
    return False