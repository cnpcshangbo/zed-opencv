#!/bin/bash
# Light from ground:
DIR="$( cd "$( dirname "$0" )" && pwd )"
echo DIR = $DIR
# python ~/sda/zed-opencv/python/zed-opencv.py /home/nvidia/sda/SVO/HD720_SN2353053_16-04-32.svo
python $DIR/python/zed-opencv.py --source /home/bo/SVO/output.svo --ip 0.0.0.0 --port 8000
