import pgl
import paddle
import paddle.nn as nn
from dataloader import BasicDataset, Loader
from model import *
from utils import *
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import world
import time

from pgl.nn import functional as GF
from tqdm import tqdm
from time import time

def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = UniformSample_original_python(dataset)
    users, posItems, negItems = shuffle(S[:, 0], S[:, 1], S[:, 2])
    users = paddle.to_tensor(users, dtype='int64')
    posItems = paddle.to_tensor(posItems, dtype='int64')
    negItems = paddle.to_tensor(negItems, dtype='int64')
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    pbar = tqdm(minibatch(users,
                    posItems,
                    negItems,
                    batch_size=world.config['bpr_batch_size']))
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(pbar):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        pbar.set_description(f'losses: {aver_loss[0]/(batch_i+1)}')
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss[0]:.3f}-{time_info}"
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: LightGCN
    # eval mode with no dropout
    Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        CORES = multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with paddle.no_grad():
        users = list(testDict.keys())
        print(len(users))
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in tqdm(minibatch(users, batch_size=u_batch_size)):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = paddle.to_tensor(batch_users, dtype='int64')
            rating = Recmodel.getUsersRating(batch_users_gpu)
           
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = paddle.topk(rating, k=max_K)
            rating = rating.numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if multicore == 1:
            pool.close()
        print(results)
        return results
def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")
if __name__ == '__main__':
    train_dataset = Loader(path=world.dataset)
    Recmodel = LightGCN(world.config, train_dataset)
    Neg_k = 1
    bpr=BPRLoss(Recmodel, world.config)
    f = open (f'train_logger_{world.dataset}.txt','w')
    f_test = open (f'test_logger_{world.dataset}.txt','w')
    for epoch in range(world.TRAIN_epochs):
        if epoch %10 == 0:
            cprint("[TEST]")
            result = Test(train_dataset, Recmodel, epoch, world.config['multicore'])
            print(epoch, result, file=f_test, flush=True)
        output_information = BPR_train_original(train_dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=None)
        log_output = f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}'
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        print(log_output, file=f, flush=True)
    f.close()
    f_test.close()