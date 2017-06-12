import math
import cv2
import os
import numpy as np
np.random.seed(343) # for reproducibility
from collections import Counter
import itertools
from functools import partial
from pylab import *
from copy import deepcopy
from scipy.spatial.distance import cdist
from utils import BoundingBox
import sys

n_classes = 14
w = 84
h = 84
data_shape = 84*84

test_sub = sys.argv[1]

#else:
if True:
    test_data = np.load('test_data_14labels_%s.npy' % (test_sub))
    test_label = np.load('test_label_14labels_%s.npy' % (test_sub))
    test_data = test_data.reshape((len(test_data),1,w,h))
    if n_classes==12: test_label = test_label[:,:,1:]



#----
get_8 = lambda x,y: [(x,y-1), (x,y+1), (x-1,y), (x+1,y),
                         (x-1,y-1), (x-1,y+1), (x+1, y-1), (x+1,y+1)]
#1. find possible candidates for edge points
def neighbors_8(mat_, (x, y), flagged_list):
    # w,h = mat_.shape
    # all_n8_values = []
    # for (x,y) in zip(xs,ys):
    n8 = [i for i in get_8(x,y) if 0<=i[0]<w and 0<=i[1]<h]
    n8_values = [(i, mat_[i]) for i in n8 if mat_[i]>0 and i not in flagged_list]
    return n8_values

get_16 = lambda x,y: [(x, y - 2), (x, y + 2), (x - 2, y), (x + 2, y),
                      (x - 1, y - 2), (x - 1, y + 2), (x + 1, y - 2), (x + 1, y + 2),
                      (x - 2, y - 1), (x - 2, y + 1), (x + 2, y - 1), (x + 2, y + 1),
                      (x - 2, y - 2), (x - 2, y + 2), (x + 2, y - 2), (x + 2, y + 2)]
#1. find possible candidates for edge points
def neighbors_16(mat_, (x, y), flagged_list):
    # w,h = mat_.shape
    # all_n8_values = []
    # for (x,y) in zip(xs,ys):
    n16 = [i for i in get_16(x,y) if 0<=i[0]<w and 0<=i[1]<h]
    n16_values = [(i, mat_[i]) for i in n16 if mat_[i]>0 and i not in flagged_list]
    return n16_values
#---

def get_greedy_contours(prob_map_, p_thr=0.1, greedier=False, only_8=False):
    # handling prob_map
    prob_map = prob_map_.reshape((w, h)).copy()
    # t_thr = 0.1
    prob_map[prob_map < p_thr] = 0

    xs, ys = np.nonzero(prob_map)
    xys = zip(xs, ys)
    all_n8 = [neighbors_8(prob_map, i, []) for i in xys]

    all_p_values = [prob_map[i] for i in xys]
    sorted_p_values = np.argsort(all_p_values)[::-1].tolist()


    # possible endpoints - points with only 2 neighbors_8
    # - to plot on image space invert x vs y
    idxs_with_n2 = [idx for idx, i in enumerate(all_n8) if len(i) == 2]
    if not idxs_with_n2:
        idxs_with_n2 = [idx for idx, i in enumerate(all_n8) if len(i) == 1]
        if not idxs_with_n2:
            idxs_with_n2 = [idx for idx, i in enumerate(all_n8) if len(i) == 3]
            if not idxs_with_n2:
                idxs_with_n2 = sorted_p_values[:3]
    end_pts = [xys[i] for i in idxs_with_n2]
    if not end_pts: print('uh oh - no end points???')

    # pick a starting point and keep going by choosing the next pt with max prob.
    # stopping criteria : no 8-neighborhood pixels avail
    end_pts_tmp = deepcopy(end_pts)
    paths = []
    paths_values = []
    for end_pt in end_pts:
        path = []
        path_values = []
        next_pt_status = True
        path.append(end_pt)
        end_pts_tmp.remove(end_pt)

        next_pt = end_pt
        flagged_list = [end_pt]
        while next_pt_status:
            next_pts = neighbors_8(prob_map, path[-1], flagged_list + path)
            if not only_8:
                if not next_pts:
                    next_pts = neighbors_16(prob_map, path[-1], flagged_list + path)
            if len(next_pts) > 0:
                next_pt_status = True
                next_pts.sort(key=lambda v: v[1], reverse=True)
                # next_max_prob_idx = np.argmax([i[1] for i in next_pts])
                next_pt = next_pts[0][0]
                next_prob = next_pts[0][1]
                # not_next_idxs = [i for i in range(len(next_pts)) if i != ]
                not_next = [i[0] for i in next_pts[1:]]
                flagged_list += not_next
                # print next_pt, '------'
                # print next_pts
                path.append(next_pt)
                path_values.append(next_prob)

                # if path[-1] in [end_pts_tmp[-1]]: next_pt_status = False
            else:
                next_pt_status = False
        paths.append(path)
        paths_values.append(path_values)

    # max_prob_path_idx = np.argmax([sum(i) for i in paths_values])
    # print len(paths)
    # if paths:
    #     max_path_len_idx = np.argmax([len(i) for i in paths])
    #     return paths[max_path_len_idx], paths_values[max_path_len_idx]
    # # if max_path_len_idx!=max_prob_path_idx: pick_idx = max_path_len_idx
    # else:
    #     return xys, []
    #--- if greedier then check that atleast 25% of the nonzero pixels
    # are used as contour points else return the whole
    p_map = prob_map.copy()
    p_map[p_map < 0.5] = 0
    x_, y_ = np.nonzero(p_map)
    total_pts_traced = np.sum([len(i) for i in paths])
    if total_pts_traced>0 and greedier:
        total_pts_above_thr = len(xys)
        if total_pts_traced/total_pts_above_thr < 0.25:

            all_paths = []
            for p in paths: all_paths+=p
            return paths+[[i for i in zip(x_,y_) if i not in p]], paths_values+[]

    if paths: return paths, paths_values
    else:
        if len(x_)>0:
            return [zip(x_,y_)], []
        else:
            p_map = prob_map.copy()
            p_map[p_map < 0.2] = 0
            x_, y_ = np.nonzero(p_map)
            return [zip(x_,y_)], []

output_labels = {1:'epiglottis',2:'tongue', 3:'incisor', 4:'lower_lip', 5:'jaw', 6:'trachea',
                 7: 'pharynx',
                 8: 'palate',9: 'velum', 10: 'nasal_cavity', 11: 'nose', 12: 'upper_lip'}


PLOT = False #bool(sys.argv[2]) #False

if PLOT:
    ion()
fig, ax = subplots(1,2)
# output = autoencoder.predict_proba(test_data[100:101])
# filname here below
output_ = np.load('output_%s_test_segnet_custom_loss.npy' % (test_sub))
colors = [np.random.rand(3) for i in range(20)]
ALL_costs = []
ALL_snrs = []

for o_i in range(len(output_)):
    if not o_i%100 : print o_i#, '-------------------------------'
    output = output_[o_i:o_i+1]
    im_in = test_data[o_i].reshape((84, 84))
    gt_14 = test_label[o_i:o_i + 1]
    o_all_thr = np.sum(output[:, :, 1:-1], axis=-1).reshape((w, h)) > 0.2

    if PLOT:
        # clf()
        # imshow(im_in, alpha=0.5)
        imshow(np.sum(gt_14[:,:,1:-1], axis=-1).reshape((w,h))*25, alpha=0.2)
        # imshow(o_all_thr, alpha=0.5)

    im = im_in.copy()
    ALL_paths = []
    ALL_path_values = []
    costs = []
    snrs = []
    for art_x in [2, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        # print art_x,
        if art_x==2: paths, path_values = get_greedy_contours(output[:,:,art_x], 0.1, greedier=True)
        elif art_x==4: paths, path_values = get_greedy_contours(output[:,:,art_x], 0.1, greedier=False, only_8=True)
        elif art_x==7: paths, path_values = get_greedy_contours(output[:,:,art_x], 0.1, greedier=True)

        else: paths, path_values = get_greedy_contours(output[:,:,art_x], 0.1)
        # for path in paths:[plot(i[1],i[0], c=colors[art_x], marker='.') for i in path]
        all_paths = []
        if not paths:  paths, path_values = get_greedy_contours(output[:,:,art_x], 0.1, greedier=False)
        for p in paths: all_paths+=p
        # print [len(i) for i in paths],

        if art_x in [2,8]:
            all_paths = []
            for p in paths:
                if len(p)>5: all_paths+=p
            if all_paths:
                l = Counter(all_paths).keys()
                mlat = sum(x[1] for x in l) / len(l)
                mlng = sum(x[0] for x in l) / len(l)
                def algo(x): return (math.atan2(x[1] - mlat, x[0] - mlng) + 2 * math.pi) % (2*math.pi)
                l.sort(key=algo)

                p_arr_sorted =  np.array([(i[1],i[0]) for i in l])

                if PLOT: cv2.polylines(im, [p_arr_sorted], False, (255,0,0))
                else:
                    im_copy = np.zeros(im_in.shape)
                    cv2.polylines(im_copy, [p_arr_sorted], False, (255,0,0))
                    x_, y_ = np.nonzero(im_copy)
                    all_paths = zip(x_,y_)

        elif art_x in [10, 9]:
            if all_paths:
                l = Counter(all_paths).keys()
                mlat = sum(x[0] for x in l) / len(l)
                mlng = sum(x[1] for x in l) / len(l)
                def algo2(x): return (math.atan2(x[1] - mlng, x[0] - mlat) + 2 * math.pi) % (2*math.pi)
                l.sort(key=algo2)

                p_arr_sorted =  np.array([(i[1],i[0]) for i in l])

                if PLOT: cv2.polylines(im, [p_arr_sorted], False, (255,0,0))
                else:
                    im_copy = np.zeros(im_in.shape)
                    cv2.polylines(im_copy, [p_arr_sorted], False, (255,0,0))
                    x_, y_ = np.nonzero(im_copy)
                    all_paths = zip(x_,y_)

        elif art_x in [7]:
            all_paths = [paths[np.argmax([len(i) for i in paths])]][0]
            if all_paths:
                l = Counter(all_paths).keys()
                l.sort()
                im_copy = np.zeros(im_in.shape)
                # all_paths = [(0,l[0][1])]+all_paths+[(84, l[-1][1])]
                if PLOT:
                    cv2.line(im, l[0][::-1], (l[0][1], 0), (255,0,0))
                    for i in range(1, len(l)): cv2.line(im, l[i][::-1], l[i-1][::-1], (255,0,0))
                    cv2.line(im, l[-1][::-1], (l[-1][1],84), (255,0,0))

                else:
                    cv2.line(im_copy, l[0][::-1], (l[0][1], 0), (255,0,0))
                    for i in range(1, len(l)): cv2.line(im_copy, l[i][::-1], l[i-1][::-1], (255,0,0))
                    cv2.line(im_copy, l[-1][::-1], (l[-1][1],84), (255,0,0))
                    x_, y_ = np.nonzero(im_copy)
                    all_paths = zip(x_,y_)

        else:
            for path in paths:
                if len(path)>0:
                    p_arr = np.array([(i[1],i[0]) for i in path])
                    if PLOT: cv2.polylines(im, [p_arr], False, (255,0,0))
            #         else:
            #             im_copy = np.zeros(im_in.shape)
            #             cv2.polylines(im_copy, [p_arr], False, (255,0,0))
            # x_, y_ = np.nonzero(im_copy)
            # all_paths = zip(x_,y_)

        ALL_paths.append(paths)
        ALL_path_values.append(path_values)

        # get unique points from GT
        gt_art_x = gt_14[:, :, art_x].reshape((w, h))
        gt_xs, gt_ys = np.nonzero(gt_art_x)
        gt_xys = zip(gt_xs, gt_ys)
        gt_bbox = BoundingBox(gt_xys)
        im_in_bbox = im_in[gt_bbox.minx:gt_bbox.maxx, gt_bbox.miny: gt_bbox.maxy]
        im_snr = mean(im_in_bbox) / std(im_in_bbox)

        gt_unique_pts = Counter(gt_xys).keys()

        # - - eval starts here - - #
        # get unique predicted points
        if all_paths:
            sys_unique_pts = Counter(all_paths).keys()

            c_dist = cdist(sys_unique_pts, gt_unique_pts, 'cityblock')
            # closest point in GT for each sys
            sys_to_gt = np.min(c_dist,0)
            # closest point in sys for each GT
            gt_to_sys = np.min(c_dist,1)
            art_x_cost = (mean(sys_to_gt)+mean(gt_to_sys))/2
        else: art_x_cost = 999
        # print art_x_cost, '----'
        costs.append(art_x_cost)
        snrs.append(im_snr)


    if PLOT:
        ax[0].imshow(im)
        ax[1].imshow(np.sum(gt_14[:, :, 1:-1], axis=-1).reshape((w, h)) * 25)
        pause(0.1)

    ALL_costs.append(costs)
    ALL_snrs.append(snrs)

np.save('eval_%s_test_manhattan.npy' % (test_sub), np.array(ALL_costs))
