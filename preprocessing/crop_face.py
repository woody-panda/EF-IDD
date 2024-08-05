
import os
from os.path import join
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm

from RetinaFace.data import cfg_mnet, cfg_re50
from RetinaFace.layers.functions.prior_box import PriorBox
from RetinaFace.utils.nms.py_cpu_nms import py_cpu_nms
from RetinaFace.models.retinaface import RetinaFace
from RetinaFace.utils.box_utils import decode
import shutil
import glob

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))

    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    model.to(device)
    return model


def detect(img_list, output_path, resize=1):
    os.makedirs(output_path, exist_ok=True)
    im_height, im_width, _ = img_list[0].shape
    scale = torch.Tensor([im_width, im_height, im_width, im_height])
    img_x = torch.stack(img_list, dim=0).permute([0, 3, 1, 2])
    scale = scale.to(device)

    # batch size
    batch_size = args.bs
    # forward times
    f_times = img_x.shape[0] // batch_size
    if img_x.shape[0] % batch_size != 0:
        f_times += 1
    locs_list = list()
    confs_list = list()
    for _ in range(f_times):
        if _ != f_times - 1:
            batch_img_x = img_x[_ * batch_size:(_ + 1) * batch_size]
        else:
            batch_img_x = img_x[_ * batch_size:]  # last batch
        batch_img_x = batch_img_x.to(device).float()
        l, c, _ = net(batch_img_x)
        locs_list.append(l)
        confs_list.append(c)
    locs = torch.cat(locs_list, dim=0)
    confs = torch.cat(confs_list, dim=0)

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data

    img_cpu = img_x.permute([0, 2, 3, 1]).cpu().numpy()
    i = 0
    for img, loc, conf in zip(img_cpu, locs, confs):
        boxes = decode(loc.data, prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]

        if len(dets) == 0:
            continue
        det = list(map(int, dets[0]))
        x, y, size_bb_x, size_bb_y = get_boundingbox(det, img.shape[1], img.shape[0])
        cropped_img = img[y:y + size_bb_y, x:x + size_bb_x, :] + (104, 117, 123)
        cv2.imwrite(join(output_path, '{:04d}.png'.format(i)), cropped_img)
        i += 1
    pass


def extract_frames(data_path, interval=1):

    """Method to extract frames"""
    reader = cv2.VideoCapture(data_path)

    frame_count_org = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
   
    
    if frame_count_org < interval or interval==-1:
       frame_idxs = [i for i in range(frame_count_org)]
    else:   
       frame_idxs = np.linspace(0, frame_count_org - 1, interval, endpoint=True, dtype=int)
    frames = list()



    for cnt_frame in range(frame_count_org): 

        success, image = reader.read()
        
        if not success:
            break
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image) - torch.tensor([104, 117, 123])
        if cnt_frame in frame_idxs:
            frames.append(image)
    #print('frames', len(frames))
    reader.release()
    #if len(frames) > args.max_frames:
        #samples = np.random.choice(
            #np.arange(0, len(frames)), size=args.max_frames, replace=False)
        #return [frames[_] for _ in samples]
   
    return frames


def get_boundingbox(bbox, width, height, scale=1.3, minsize=None):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    size_bb_x = int((x2 - x1) * scale)
    size_bb_y = int((y2 - y1) * scale)
    if minsize:
        if size_bb_x < minsize:
            size_bb_x = minsize
        if size_bb_y < minsize:
            size_bb_y = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb_x // 2), 0)
    y1 = max(int(center_y - size_bb_y // 2), 0)
    # Check for too big bb size for given x, y
    size_bb_x = min(width - x1, size_bb_x)
    size_bb_y = min(height - y1, size_bb_y)
    return x1, y1, size_bb_x, size_bb_y


def extract_videos(video_list, output_dir, interval, dataset):
  
    num_unqualified = 0

    for video in tqdm(video_list):
   
        try:
            image_list = extract_frames(video, interval)

            if '.mp4' in video:
                v_type = '.mp4'
            elif '.avi' in video:
                v_type = '.avi'
            elif '.mov' in video:
                v_type = '.mov'

            target_path = join(output_dir, video.replace(v_type, ''))
            if os.path.exists(target_path):
                os.makedirs(target_path)

            detect(image_list, target_path)
        except Exception as ex:
            f = open("{}_failure.txt".format(dataset), "w", encoding="utf-8")
            f.writelines(video +
                        f"  Exception for {video}: {ex}\n")
            f.close()
            num_unqualified += 1
    print("Total unqualified: ", num_unqualified)

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('--dataset', default='Kodf', type=str, help='dataset')
    p.add_argument('--output_dir', default='data', type=str, help='path to the ouput data')
    p.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence threshold')
    p.add_argument('--top_k', default=5, type=int, help='top_k')
    p.add_argument('--nms_threshold', default=0.4, type=float, help='nms threshold')
    p.add_argument('--keep_top_k', default=1, type=int, help='keep_top_k')
    p.add_argument('--bs', default=16, type=int, help='batch size')
    p.add_argument('--frame_interval', default=50, type=int, help='frame interval')
    p.add_argument('--device', "-d", default="cuda:0", type=str, help='device')
    #p.add_argument('--max_frames', default=10, type=int, help='maximum frames per video')

    args = p.parse_args()

    torch.set_grad_enabled(False)
    # use resnet-50
    cfg = cfg_re50
    pretrained_weights = './weights/Resnet50_Final.pth'

    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    print(device)

    #if os.path.exists(args.output_dir):
        #os.rmdir(args.output_dir)

    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, pretrained_weights, args.device)
    net.eval()
    print('Finished loading model!')

    video_list = []

    if args.dataset == 'FF++':
        video_list += glob.glob(join('FaceForensics++', 'original_sequences', 'youtube', 'c23', 'videos','*.mp4'),
                                recursive=True)
        video_list += glob.glob(join('FaceForensics++', 'manipulated_sequences', '*', 'c23', 'videos', '*.mp4'),
                                recursive=True)
    elif args.dataset == 'Kodf':
        video_list += glob.glob(join('Kodf', '*', '*', '*.mp4'), recursive=True)
    print('video_list length', len(video_list))

    extract_videos(video_list, args.output_dir, args.frame_interval, args.dataset)