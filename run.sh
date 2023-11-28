#!/bin/bash

python3 /home/luuk/development/first-order-model/demo.py --config /home/luuk/development/first-order-model/config/vox-256.yaml --driving_video /home/luuk/development/first-order-model/driving.mp4 --source_image /home/luuk/development/first-order-model/source.png --checkpoint /home/luuk/development/first-order-model/vox-cpk.pth.tar --result_video /home/luuk/development/first-order-model/result.mp4 --relative --adapt_scale
