import sys
import os
import pickle
import subprocess

dataset = sys.argv[1]
lr = sys.argv[2]
iter = sys.argv[3]

for i in [1,2,3,4,5]:
    cmd1 = 'code/Model/retype/retype-rm -data %s -mode m -size 50 -negative 3 -threads 3 -alpha 0.0001 -samples 1 -lr %s -iters %s -rand_seed %d'\
        % (dataset, lr, iter, i)
    print(cmd1)
    subprocess.call(cmd1,shell=True)

    cmd2 = 'python2 code/Evaluation/emb_dev_n_test.py extract %s retypeRm cosine 0.0 %s %s' % (dataset, lr, iter)
    print(cmd2)
    subprocess.call(cmd2,shell=True)

    cmd3 = 'python2 code/Evaluation/tune_threshold_w_sampled_dev.py extract %s emb retypeRm cosine %d' % (dataset, i)
    print(cmd3)
    subprocess.call(cmd3,shell=True)