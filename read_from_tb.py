import os
import numpy as np
import tensorflow as tf
import sys

data_num = int(sys.argv[1])
seeds = [1, 2, 3]
versions = [0, 0, 0]

values = []
iters = []

for s, v in zip(seeds, versions):
    path_to_events_file = f"lightning_logs/cifar10_semi_mixmatch_@{data_num}.{s}_LB_NoV_FS_W_Batch_2/version_{v}/"
    # path_to_events_file = f"lightning_logs/cifar10_supervised_mixup_fs_mixup.{s}/version_{v}"

    files = os.listdir(path_to_events_file)
    for file_ in files:
        if file_.startswith("event"):
            event_file = file_ 

    print(event_file)

    path_to_events_file = os.path.join(path_to_events_file, event_file)

    last_iter = 0
    for e in tf.train.summary_iterator(path_to_events_file):
        for v in e.summary.value:
            if v.tag.startswith("test/median_acc"):
                value = v.simple_value
                last_iter += 1

    values.append(value)
    iters.append(last_iter)

# for i in range(1, len(iters)):
#     assert iters[i] == iters[i - 1]

print(iters)
print(values)
print(100 - np.mean(values))
print(np.std(values))
        