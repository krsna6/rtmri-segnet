import itertools
import numpy as np
import os
import cv2
import glob
from collections import Counter

SEED = 428
np.random.seed(SEED) # for reproducibility

def binarylab(labels, n_labels=13):
    w,h = labels.shape
    x = np.zeros([w*h,n_labels])
    for i in range(n_labels):
        l = (labels == i)
        x[:,i] = l.reshape((w*h))
    return x.astype('uint8')


PREP_DATA = True
if PREP_DATA:
    # do LOSO testing - GT=ground truth
    LOO = 1
    # have to reorder labels after excluding 8,9,10 - hence give it the correspondences
    # re_order = [(11,8), (12,9), (13,10), (14,11), (15,12)]
    subjects = np.array(['at1', 'vl1', 'tj1', 'sg2', 'eh3', 'cv2', 'cl2', 'yb1'])
    im_dir = '/data/workspace/span/mri_segmentation_io/subjects/%s/images/'
    gt_dir = '/data/workspace/span/mri_segmentation_io/subjects/%s/GT_12/'
    N_subs = len(subjects)

    tr_idxs = np.random.choice(N_subs, N_subs-LOO, replace=False)
    print(tr_idxs)
    tst_idx = [i for i in range(N_subs) if i not in tr_idxs]
    tr_subjects = subjects[tr_idxs].tolist()
    tst_subjects = subjects[tst_idx].tolist()

    N_samples_per_sub = 11000

    train_data = []
    train_label = []
    for tr_sub in tr_subjects:
        print(tr_sub)
        im_sub_dir = im_dir % (tr_sub)
        gt_sub_dir = gt_dir % (tr_sub)
        GT_list_all = glob.glob(gt_sub_dir + '/*/*.npy')
        total_n = len(GT_list_all)
        GT_picker = np.random.choice(total_n, min(total_n, N_samples_per_sub), replace=False)
        GT_list = [GT_list_all[i] for i in GT_picker]
        # GT_list = [i[:-1] for i in open('train_data_files_except_vl1.txt','rU').readlines()]
        with open('train_data_files_except_%s.txt' % (tst_subjects[0]),'w') as F:
            F.write('\n'.join(GT_list))
            F.close()
        for gt_path in GT_list:
                track_dir = gt_path.split('/')[-2]
                im_track_dir = os.path.join(im_sub_dir, track_dir)
                im_name = os.path.basename(gt_path).split('.npy')[0]+'.png'
                im_path = os.path.join(im_track_dir, im_name)
                train_data.append(cv2.imread(im_path, cv2.IMREAD_GRAYSCALE))
                train_label.append(binarylab(np.load(gt_path)))
                # a small detour to not predict all labels -
                # predict all except 8,9,10 which are just edge cases on the right
                # gt_im = np.load(gt_path)
                # gt_im[pl.logical_and(gt_im>=8, gt_im<=10)]=0
                # for re_ord in re_order:
                #     gt_im[gt_im==re_ord[0]]=re_ord[1]
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    np.save('train_data_except_%s.npy' % (tst_subjects[0]),train_data)
    np.save('train_label_except_%s.npy' % (tst_subjects[0]), train_label)

    test_data = []
    test_label = []
    for tr_sub in tst_subjects:
        print(tr_sub)
        im_sub_dir = im_dir % (tr_sub)
        gt_sub_dir = gt_dir % (tr_sub)
        GT_list_all = glob.glob(gt_sub_dir + '/*/*.npy')
        for gt_path in GT_list_all:
            track_dir = gt_path.split('/')[-2]
            im_track_dir = os.path.join(im_sub_dir, track_dir)
            im_name = os.path.basename(gt_path).split('.npy')[0] + '.png'
            im_path = os.path.join(im_track_dir, im_name)
            test_data.append(cv2.imread(im_path, cv2.IMREAD_GRAYSCALE))
            test_label.append(binarylab(np.load(gt_path)))
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    np.save('test_data_%s.npy' % (tst_subjects[0]), test_data)
    np.save('test_label_%s.npy' % (tst_subjects[0]), test_label)

