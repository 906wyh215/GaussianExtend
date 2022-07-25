import pandas as pd
import numpy as np
import random
import itertools
from scipy import stats
import heapq
import copy


def R_MSE_score(y_true, y_pred):  # RMSE
    """
    Define root mean square error
    @param y_true: Observed value
    @param y_pred: predicted value
    @return:
    """
    return round(np.sqrt(np.mean((y_true - y_pred) ** 2)), 2)


def Pearson_score(y_ture, y_pred):
    """
    Define Pearson correlation coefficient
    @param y_ture:
    @param y_pred:
    @return:
    """
    return sum((y_ture - y_ture.mean()) * (y_pred - y_pred.mean())) / (
            ((sum((y_ture - y_ture.mean()) * (y_ture - y_ture.mean()))) *
             (sum((y_pred - y_pred.mean()) * (y_pred - y_pred.mean())))) ** 0.5)


def r_2(y_true, y_pred):
    """
    Define the coefficient of determination
    @param y_true:
    @param y_pred:
    @return:
    """
    return 1 - (sum((y_true - y_pred) ** 2) / sum((y_true - y_true.mean()) ** 2))


def Partition_data(n_samples, n_splits, shuffle=False, random_state=None):
    """
    @param random_state: seed
    @param shuffle: random shuffle,set the same shuffle method for each operation
    @param n_samples: the number of data
    @param n_splits: cv
    """
    sample_index = np.arange(n_samples)
    if shuffle:
        seed = random_state
        random.seed(seed)
        random.shuffle(sample_index)
    split_size = np.full(n_splits, n_samples // n_splits, dtype=np.int)
    split_size[:n_samples % n_splits] += 1
    res = []
    curr = 0
    for step in split_size:
        start, stop = curr, curr + step
        res.append(list(sample_index[start:stop]))
        curr = stop
    return res


def split_train_validation(i, k_fold, X, y, noise):
    k_fold_ = k_fold[:]
    test_fold = k_fold_.pop(i)
    x_test = X[test_fold]
    y_test = y[test_fold]
    train_fold = [i for ii in k_fold_ for i in ii]
    x_train = X[train_fold]
    y_train = y[train_fold]
    alpha = noise[train_fold]
    if len(alpha) == 0 and len(x_train) == 0 and len(y_train) == 0:
        x_train, y_train, alpha = x_test, y_test, noise
    return x_train, y_train, x_test, y_test, alpha.reshape(-1)


def get_virtual_sample(parameters_boundary: dict, step):
    """
    Get parameter space.
    @param parameters_boundary: Parameter value range
    @param step: Parameter value step
    @return: parameter space
    """
    parameters_value = []
    for i in parameters_boundary.keys():
        parameters_value.append(list(np.arange(parameters_boundary[i][0],
                                               parameters_boundary[i][1] + 1,
                                               step)))
    virtual_X = itertools.product(*parameters_value)
    virtual_X = list(map(lambda x: list(x), virtual_X))
    return virtual_X


def Thompson_sampling(mean, std, virtual_X):
    # x* is derived by searching at the vistual space
    # sample = np.len(virtual_samples)
    optimal_value_set = []
    for i in range(len(virtual_X)):
        y_value = np.random.normal(loc=mean[i], scale=std[i])
        optimal_value_set.append(y_value)
    index = np.where(np.array(optimal_value_set) == np.array(optimal_value_set).max())[0][0]
    return virtual_X[index], optimal_value_set[index]


def EI(mean, std, f_max, virtual_X):
    EI_ = (mean - f_max) * (stats.norm.cdf((mean - f_max) / std)) \
          + std * (stats.norm.pdf((mean - f_max) / std))
    maxEI_index = heapq.nlargest(1, range(len(EI_)), EI_.take)
    return virtual_X[maxEI_index[0]]


def UCB(mean, std, virtual_X, beta=1):
    UCB_ = mean + beta * std
    maxUCB_index = heapq.nlargest(1, range(len(UCB_)), UCB_.take)
    return virtual_X[maxUCB_index[0]]


def POI(mean, std, f_max, virtual_X):
    POI_ = stats.norm.cdf((mean - f_max) / std)
    maxPOI_index = heapq.nlargest(1, range(len(POI_)), POI_.take)
    return virtual_X[maxPOI_index[0]]


def PES(GPR, opt_X, opt_y, mean, std, virtual_X, sam_num=500):
    Entropy_y_ori = 0.5 * np.log(2 * np.pi * np.e * (std ** 2))
    Entropy_y_conditional = np.zeros(len(virtual_X))
    for i in range(sam_num):
        sample_x, sample_y = Thompson_sampling(mean, std, virtual_X)

        archive_sample_x = copy.deepcopy(np.array(opt_X))
        archive_sample_y = copy.deepcopy(np.array(opt_y))

        archive_sample_x = np.append(archive_sample_x, sample_x)
        archive_sample_y = np.append(archive_sample_y, sample_y)
        fea_num = len(opt_X[0])
        GPR.fit(archive_sample_x.reshape(-1, fea_num), archive_sample_y)
        _, post_std = GPR.predict(virtual_X, return_std=True)
        Entropy_y_conditional += 0.5 * np.log(2 * np.pi * np.e * (post_std ** 2))
    estimated_Entropy_y_conditional = Entropy_y_conditional / sam_num
    PES_ = Entropy_y_ori - estimated_Entropy_y_conditional
    maxPES_index = heapq.nlargest(1, range(len(PES_)), PES_.take)
    return virtual_X[maxPES_index[0]]


def KD(GPR, opt_X, opt_y, mean, std, virtual_X, MC_num=500):
    current_max = mean.max()
    KD_list = []
    n = len(mean)
    for i in range(n):
        V_X = virtual_X[i]
        MC_batch_min = 0
        for j in range(MC_num):
            y_value = np.random.normal(mean[i], scale=std[i])
            archive_sample_x = copy.deepcopy(np.array(opt_X))
            archive_sample_y = copy.deepcopy(np.array(opt_y))

            archive_sample_x = np.append(archive_sample_x, V_X)
            archive_sample_y = np.append(archive_sample_y, y_value)
            fea_num = len(opt_X[0])
            # return a callable model
            GPR.fit(archive_sample_x.reshape(-1, fea_num), archive_sample_y)
            post_mean, _ = GPR.predict(virtual_X, return_std=True)
            MC_batch_min += post_mean.max()
            # MC_times = i * MC_num + j + 1
            # if MC_times % 2000 == 0:
            #     print('The {num}-th Monte carlo simulation'.format(num=MC_times))
        MC_result = MC_batch_min / MC_num
        KD_list.append(MC_result - current_max)
    KD_ = np.array(KD_list)
    maxKD_index = heapq.nlargest(1, range(len(KD_)), KD_.take)
    return virtual_X[maxKD_index[0]]


if __name__ == '__main__':
    param_grid = {'a': (50, 100),
                  'l1': (5, 80),
                  'l2': (5, 80),
                  'l3': (5, 80),
                  'l4': (5, 80)}
    # print(param_grid.keys())
    # for i in param_grid.keys():
    #     print(i)
    get_virtual_sample(param_grid, 10)
    a = [[1, 2], [3, 4]]
    res = itertools.product(*a)
    print(list(res))
    print(len(param_grid))
    print([1] * 4)
