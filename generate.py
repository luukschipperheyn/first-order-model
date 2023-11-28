from flask import Flask, request, send_from_directory
from demo import generate as generate_fake, load_checkpoints, DefaultOptions
import os
from multiprocessing import Process

app = Flask(__name__)
app.config["RESULT_VIDEOS"] = "/home/luuk/development/first-order-model/results-2"

os.makedirs(os.path.join(app.instance_path, 'upload'), exist_ok=True)
opt=DefaultOptions()
generator, kp_detector = load_checkpoints(
        config_path=opt.config,
        checkpoint_path=opt.checkpoint,
        cpu=opt.cpu
    )


generate_fake(generator, kp_detector, driver_index=int(1))
generate_fake(generator, kp_detector, driver_index=int(2))
generate_fake(generator, kp_detector, driver_index=int(3))
generate_fake(generator, kp_detector, driver_index=int(4))
generate_fake(generator, kp_detector, driver_index=int(5))

