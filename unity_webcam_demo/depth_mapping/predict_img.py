import argparse
import cv2
import math
from depth_mapping import hsm
import numpy as np
import os
import skimage.io
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time
from depth_mapping.submodule import *
from depth_mapping.preprocess import get_transform

def load_model(loadmodel, max_disp, clean,level):
    # construct model
    model = hsm(int(max_disp), clean,level=level)
    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()

    if loadmodel is not None:
        pretrained_dict = torch.load(loadmodel)
        pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
        model.load_state_dict(pretrained_dict['state_dict'],strict=False)
    else:
        print('run with random init')
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    return model

def trial_run(model):
    # dry run
    multip = 48
    imgL = np.zeros((1,3,24*multip,32*multip))
    imgR = np.zeros((1,3,24*multip,32*multip))
    imgL = Variable(torch.FloatTensor(imgL).cuda())
    imgR = Variable(torch.FloatTensor(imgR).cuda())
    with torch.no_grad():
        model.eval()
        pred_disp,entropy = model(imgL,imgR)

    print("::INFO:: Dry Run Sucessfull. Model Loaded.")


def test(model, imgL_o, imgR_o, max_disp, test_res, processed, verbose = 1):

    if max_disp>0:
        if max_disp % 16 != 0:
            max_disp = 16 * math.floor(max_disp/16)
        max_disp = int(max_disp)

    ## change max disp
    tmpdisp = int(max_disp*test_res//64*64)
    if (max_disp*test_res/64*64) > tmpdisp:
        model.module.maxdisp = tmpdisp + 64
    else:
        model.module.maxdisp = tmpdisp
    if model.module.maxdisp ==64: model.module.maxdisp=128
    model.module.disp_reg8 =  disparityregression(model.module.maxdisp,16).cuda()
    model.module.disp_reg16 = disparityregression(model.module.maxdisp,16).cuda()
    model.module.disp_reg32 = disparityregression(model.module.maxdisp,32).cuda()
    model.module.disp_reg64 = disparityregression(model.module.maxdisp,64).cuda()
    # resize
    imgL_o = cv2.resize(imgL_o,None,fx=test_res,fy=test_res,interpolation=cv2.INTER_CUBIC)
    imgR_o = cv2.resize(imgR_o,None,fx=test_res,fy=test_res,interpolation=cv2.INTER_CUBIC)
    imgL = processed(imgL_o).numpy()
    imgR = processed(imgR_o).numpy()

    imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
    imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

    ##fast pad
    max_h = int(imgL.shape[2] // 64 * 64)
    max_w = int(imgL.shape[3] // 64 * 64)
    if max_h < imgL.shape[2]: max_h += 64
    if max_w < imgL.shape[3]: max_w += 64

    top_pad = max_h-imgL.shape[2]
    left_pad = max_w-imgL.shape[3]
    imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    # save_input = np.swapaxes(imgL[0],0,1)
    # save_input = np.swapaxes(save_input, 1,2)
    # cv2.imwrite(os.path.join(output_dir, "input.png"), np.array(save_input*255, dtype=np.uint8))
    # test
    imgL = Variable(torch.FloatTensor(imgL).cuda())
    imgR = Variable(torch.FloatTensor(imgR).cuda())
    with torch.no_grad():
        torch.cuda.synchronize()
        start_time = time.time()
        pred_disp,entropy = model(imgL,imgR)
        torch.cuda.synchronize()
        ttime = (time.time() - start_time)
        if verbose ==1:
            print('Inference time = %.2f seconds' %ttime )
    pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()

    top_pad   = max_h-imgL_o.shape[0]
    left_pad  = max_w-imgL_o.shape[1]
    entropy = entropy[top_pad:,:pred_disp.shape[1]-left_pad].cpu().numpy()
    pred_disp = pred_disp[top_pad:,:pred_disp.shape[1]-left_pad]

    return pred_disp

def predict_dir(model, images_dir , max_disp, test_res, output_dir, subsample =1):

    left_dir = os.path.join(images_dir, "left")
    right_dir = os.path.join(images_dir, "right")
    left_frames = sorted(os.listdir(left_dir))
    right_frames = sorted(os.listdir(right_dir))
    if subsample>1:
        left_frames = left_frames[1::subsample]
        right_frames = right_frames[1::subsample]
    processed = get_transform()
    print("::INFO:: Inferring with max disparity: ",max_disp)
    model.eval()
    for inx in range(len(left_frames)):
        print(left_frames[inx])
        left_frames[inx] = os.path.join(left_dir, left_frames[inx])
        right_frames[inx] = os.path.join(right_dir,right_frames[inx])
        imgL_o = (skimage.io.imread(left_frames[inx]).astype('float32'))[:,:,:3]
        imgR_o = (skimage.io.imread(right_frames[inx]).astype('float32'))[:,:,:3]
        imgsize = imgL_o.shape[:2]

        pred_disp = test(model, imgL_o, imgR_o, max_disp, test_res, processed)

        idxname = left_frames[inx].split('/')[-2]
        idxname = '%s/disp0HSM'%(idxname)

        # resize to highres
        pred_disp = cv2.resize(pred_disp/test_res,(imgsize[1],imgsize[0]),interpolation=cv2.INTER_LINEAR)

        # clip while keep inf
        invalid = np.logical_or(pred_disp == np.inf,pred_disp!=pred_disp)
        pred_disp[invalid] = np.inf

        np.save('%s/%s-disp.npy'% (output_dir, os.path.split(left_frames[inx])[1]),(pred_disp))
        cv2.imwrite('%s/%s-disp.png'% (output_dir, os.path.split(left_frames[inx])[1]),pred_disp/pred_disp[~invalid].max()*255)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    cudnn.benchmark = False
    parser = argparse.ArgumentParser(description='Ideally, use this code to predict/infer on a directory of images using the HSMNet.')
    parser.add_argument('--datapath', default='./data-mbtest/',
                        help='test data path')
    parser.add_argument('--loadmodel', default=None,
                        help='model path')
    parser.add_argument('--outdir', default='output',
                        help='output dir')
    parser.add_argument('--clean', type=float, default=-1,
                        help='clean up output using entropy estimation')
    parser.add_argument('--testres', type=float, default=1,
                        help='test time resolution ratio 0-x')
    parser.add_argument('--max_disp', type=float, default=256,
                        help='maximum disparity to search for')
    parser.add_argument('--level', type=int, default=1,
                        help='output level of output, default is level 1 (stage 3),\
                            can also use level 2 (stage 2) or level 3 (stage 1)')
    args = parser.parse_args()
    model = load_model(args.loadmodel, args.max_disp, args.clean, args.level)
    trial_run(model)
    predict_dir(model,args.datapath, args.max_disp, args.testres, args.outdir)



