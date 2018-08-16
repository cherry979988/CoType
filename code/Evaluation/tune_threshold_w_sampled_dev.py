__author__ = 'xiang'

import sys, os
from collections import  defaultdict
from emb_prediction import *
from evaluation import *
import random
import numpy as np
import copy

def min_max_nomalization(prediction):
    min_val = sys.maxint
    max_val = -sys.maxint
    prediction_normalized = defaultdict(tuple)
    for i in prediction:
        if prediction[i][1] < min_val:
            min_val = prediction[i][1]
        if prediction[i][1] > max_val:
            max_val = prediction[i][1]
    print(min_val, max_val)
    for i in prediction:
        score_normalized = (prediction[i][1] - min_val) / (max_val - min_val + 1e-8)
        prediction_normalized[i] = (prediction[i][0], score_normalized)
    return prediction_normalized

def evaluate_threshold(_threshold, ground_truth):
    # print 'threshold = ', _threshold
    prediction_cutoff = defaultdict(set)
    for i in prediction:
        if prediction[i][1] > _threshold:
            prediction_cutoff[i] = set([prediction[i][0]])
    result = evaluate_rm(prediction_cutoff, ground_truth)
    # print result
    return result

def evaluate_threshold_neg(_threshold, ground_truth, none_label_index):
    # print 'threshold = ', _threshold
    # for entropy tuning it should be <, for max tuning it should be >
    prediction_cutoff = defaultdict(set)
    for i in prediction:
        if prediction[i][1] < _threshold:
            prediction_cutoff[i] = set([prediction[i][0]])
    result = evaluate_rm_neg(prediction_cutoff, ground_truth, none_label_index)
    return result

def tune_threshold(_threshold_list, ground_truth, none_label_index):
    result = defaultdict(tuple)
    for _threshold in _threshold_list:
        if none_label_index == None:
            result[_threshold] = evaluate_threshold(_threshold, ground_truth)
        else:
            result[_threshold] = evaluate_threshold_neg(_threshold, ground_truth, none_label_index)
    return result

if __name__ == "__main__":

    if len(sys.argv) != 7:
        print 'Usage: tune_threshold.py -TASK (classifer/extract) -DATA(KBP/NYT/BioInfer) -MODE(emb) -METHOD(retype) -SIM(cosine/dot) -rand_seed'
        exit(-1)

    # do prediction here
    _task = sys.argv[1]
    _data = sys.argv[2]
    _mode = sys.argv[3]
    _method = sys.argv[4]
    _sim_func = sys.argv[5]
    _seed = int(sys.argv[6])

    np.random.seed(_seed)
    random.seed(_seed)

    indir = 'data/intermediate/' + _data + '/rm'
    outdir = 'data/results/' + _data + '/rm'
    ground_truth = load_labels(indir + '/mention_type_test.txt')
    prediction = load_label_score(outdir + '/prediction_' + _mode + '_' + _method + '_' + _sim_func + '.txt')
    # print ground_truth
    file_name = outdir + '/tune_thresholds_' + _mode + '_' + _method + '_' + _sim_func +'.txt'
    # print _data, _mode, _method, _sim_func
    prediction = min_max_nomalization(prediction)
    # print(prediction) 
    none_label_index = find_none_index(indir + '/type.txt')
    precision, recall, f1 = evaluate_threshold_neg(1, ground_truth, none_label_index)
    print precision, recall, f1

    step_size = 1
    # prediction = min_max_nomalization(prediction)
    threshold_list = [float(i)/100.0 for i in range(0, 101, step_size)]
    # print threshold_list[0], 'to', threshold_list[-1], ', step-size:', step_size / 100.0

    # split prediction and ground_truth into dev and test set
    print 'total size: ', len(prediction)
    valSize = int(np.floor(0.1 * len(prediction)))
    print 'val size: ', valSize

    iterN = 100
    f1_all = 0
    precision_all = 0
    recall_all = 0
    valF1_all = 0

    prediction_original = prediction
    ground_truth_original = ground_truth

    for i in range(iterN):
        prediction = copy.deepcopy(prediction_original)
        ground_truth = copy.deepcopy(ground_truth_original)
        keys = prediction.keys()
        random.shuffle(keys)
        keys_val = keys[0:valSize]
        keys_eva = keys[valSize:]
        val_prediction = {idx: prediction[idx] for idx in keys_val}
        val_ground_truth = {idx: ground_truth[idx] for idx in keys_val}
        eva_prediction = {idx: prediction[idx] for idx in keys_eva}
        eva_ground_truth = {idx: ground_truth[idx] for idx in keys_eva}

        prediction = val_prediction
        ground_truth = val_ground_truth

        if _task == 'extract':
            none_label_index = find_none_index(indir + '/type.txt')
            # print '[None] label index: ', none_label_index
            result = tune_threshold(threshold_list, ground_truth, none_label_index)
        else:
            result = tune_threshold(threshold_list, ground_truth, None)


        ### Output
        prec_list = []
        recall_list = []
        f1_list = []
        threshold_list_str = []
        max_f1 = -sys.maxint
        max_prec = -sys.maxint
        max_recall = -sys.maxint
        max_threshold = -sys.maxint
        for _threshold in threshold_list:
            threshold_list_str.append(str(_threshold))
            precision, recall, f1 = result[_threshold]
            prec_list.append(str(precision))
            recall_list.append(str(recall))
            f1_list.append(str(f1))
            if max_f1 < f1:
                max_f1 = f1
                max_prec = precision
                max_recall = recall
                max_threshold = _threshold

        with open(file_name, 'w') as f0:
            for i in range(len(threshold_list_str)):
                if _method == 'line':
                    f0.write(recall_list[i] + '\t' + str(float(prec_list[i])) + '\n')
                elif _method == 'retype':
                    f0.write(str(float(recall_list[i])) + '\t' + str(float(prec_list[i])) + '\n')
                else:
                    f0.write(recall_list[i] + '\t' + prec_list[i] + '\n')

        # print 'Best Validation threshold:', max_threshold, '.\tPrecision:', max_prec, '.\tRecall:', max_recall, '.\tF1:', max_f1

        valF1_all += max_f1

        # evaluate on the test set
        ground_truth = eva_ground_truth
        prediction = eva_prediction
        if _task == 'extract':
            precision, recall, f1 = evaluate_threshold_neg(max_threshold, ground_truth, none_label_index)
        else:
            precision, recall, f1 = evaluate_threshold(max_threshold, ground_truth, None)
        # print 'Test \tPrecision:', precision, '.\tRecall:', recall, '.\tF1:', f1

        precision_all += precision
        recall_all += recall
        f1_all += f1

    valF1_all /= iterN
    precision_all /= iterN
    recall_all /= iterN
    f1_all /= iterN

    print 'F1:', f1_all, '.\tprecision:', precision_all, '.\tRecall:', recall_all, '.\tVal_F1:', valF1_all 

    # print eva_prediction, val_prediction
    # precision, recall, f1 = evaluate_threshold_neg(0.48, ground_truth, none_label_index)
    # print 'All together, precision:', precision, '.\tRecall:', recall, '.\tF1:', f1
