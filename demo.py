import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("helloooo")

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        print("not cpu. doing the cuda generator")
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        print("not cpu. doing the cuda detector")
        kp_detector.cuda()
    
    if cpu:
        print("cpu. doing the cpu torch")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        print("not cpu. doing the cuda torch")
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    print("evaluating generator")
    generator.eval()
    print("evaluating keypoints")
    kp_detector.eval()
    
    return generator, kp_detector


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num

class DefaultOptions():
    def __init__(self):
        self.config = '/home/luuk/development/first-order-model/config/vox-256.yaml'
        self.checkpoint = '/home/luuk/development/first-order-model/vox-cpk.pth.tar'
        self.source_image = '/home/luuk/development/first-order-model/instance/upload/source.png'
        self.driving_videos = [
            '/home/luuk/development/first-order-model/data/food1-scaled.mp4',
            '/home/luuk/development/first-order-model/data/food2-scaled.mp4',
            '/home/luuk/development/first-order-model/data/food3-scaled.mp4',
            '/home/luuk/development/first-order-model/data/food4-scaled.mp4',
            '/home/luuk/development/first-order-model/data/food5-scaled.mp4',
            '/home/luuk/development/first-order-model/data/food6-scaled.mp4',
        ]
        self.result_video = '/home/luuk/development/openFrameworks/of_v0.11.2_linux64gcc6_release/apps/myApps/faceCalibration/bin/data/result-'
        self.relative = True
        self.adapt_scale = True
        self.find_best_frame = False
        self.best_frame = None
        self.cpu = False

def generate(generator, kp_detector, opt=DefaultOptions(), driver_index=0):
    print(opt)
    source_image = imageio.imread(opt.source_image)
    source_image = resize(source_image, (256, 256))[..., :3]
    # generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    print("checkpoints loaded")
    driver = opt.driving_videos[driver_index]
    print("reading " + driver)
    reader = imageio.get_reader(driver)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()
    print("resizing")
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    print("finding best frame")
    if opt.find_best_frame or opt.best_frame is not None:
        print("succes")
        i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_image, driving_video, cpu=opt.cpu)
        print ("Best frame: " + str(i))
        print("making animation")
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        print("best frame not found. making animation")
        predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
    print("got predictions.saving vid")
    imageio.mimsave("/home/luuk/development/first-order-model/results/result-" + str(driver_index) + '.mp4', [img_as_ubyte(frame) for frame in predictions], fps=fps)
    print("saved")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="/home/luuk/development/first-order-model/config/vox-256.yaml", help="path to config")
    parser.add_argument("--checkpoint", default='/home/luuk/development/first-order-model/vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='/home/luuk/development/first-order-model/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='/home/luuk/development/first-order-model/driving.mp4', help="path to driving video")
    parser.add_argument("--result_video", default='/home/luuk/development/first-order-model/result.mp4', help="path to output")
 
    parser.add_argument("--relative", dest="relative", default=True, action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", default=True ,action="store_true", help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--find_best_frame", dest="find_best_frame", default=False, action="store_true", 
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,  
                        help="Set frame to start from.")
 
    parser.add_argument("--cpu", dest="cpu", default=True, action="store_true", help="cpu mode.")
 

    # parser.set_defaults(relative=False)
    # parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    print("options")
    print(opt)
    generate(opt)
