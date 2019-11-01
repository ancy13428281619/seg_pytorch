#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
from utils.utils import loadYaml
from model import getModels

if __name__ == '__main__':
    config = loadYaml('../config/config.yaml')
    getModels(config)
