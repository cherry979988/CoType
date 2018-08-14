import sys
import os
import pickle
import subprocess

def get_best_params(dataset):
    fin = open('tune_log.pkl', 'rb')
    d = pickle.load(fin)
    best_lr = 0
    best_iter = 0
    best_f1 = 0
    for key in d:
        if key[0]==dataset:
            if d[key][2]>best_f1:
                best_f1 = d[key][2]
                best_lr = key[1]
                best_iter = key[2]
    return best_lr, best_iter

lr_list = [0.1, 0.03, 0.01, 0.003]
iter_list = [500, 1000, 2000]

dataset = sys.argv[1]

# tune_time_seed = 1234

for lr in lr_list:
    for iter in iter_list:
        cmd1 = 'code/Model/retype/retype-rm -data %s -mode m -size 50 -negative 3 -threads 3 -alpha 0.0001 -samples 1 -lr %s -iters %s -rand_seed 1234'\
            % (dataset, lr, iter)
        print(cmd1)
        subprocess.call(cmd1,shell=True)

        cmd2 = 'python2 code/Evaluation/emb_dev_n_test.py extract %s retypeRm cosine 0.0 %s %s' % (dataset, lr, iter)
        print(cmd2)
        subprocess.call(cmd2,shell=True)

        #cmd3 = 'python2 code/Evaluation/tune_threshold_w_validation.py extract %s emb retypeRm cosine' % dataset
        #print(cmd3)
        #subprocess.call(cmd3,shell=True)


print('====TUNING COMPLETED!====')
best_lr, best_iter = get_best_params(dataset)
print('Best Param: Learning Rate = %s, Iteration = %s' % (str(best_lr), str(best_iter)))
