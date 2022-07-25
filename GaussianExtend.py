from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from tool_function import *
from bayes_opt import BayesianOptimization


class GaussianExtend(object):
    def __init__(self, data, target):
        # self.model = estimator
        # self.kernel = kernel
        self.X = np.array(data)
        self.y = np.array(target)

    def anisotropic_noise_cv(self, kernel_gp, noise, cv=10, scoring='R_MSE', shuffle=False, random_state=None):
        """
        Cross validation method of Gaussian process regression with heterogeneous noise,
        return the average score on k_fold
        @param kernel_gp: Refer to package Kernels for Gaussian Processes of scikit-learn.
                        https://scikit-learn.org/stable/modules/gaussian_process.html#kernels-for-gaussian-processes
        @param noise: ndarray of shape (n_samples,)
        @param scoring: str，scoring='R_MSE' or 'R' or 'r2',default='R_MSE';
                        'R_MSE' is root mean square error;
                        'R' is Pearson correlation coefficient
                        'r2' is the coefficient of determination
        @param cv: int, cross-validation generator , default=10
        @param shuffle: Randomly shuffle data sets,default=False
        @param random_state: seed,default=None
        """
        n = len(self.X)
        if cv <= 0:
            raise ValueError("cv must be a positive int")
        elif cv > n:
            raise ValueError("cv must be less than n_samples")
        noise = np.array(noise)
        k_fold = Partition_data(n, cv, shuffle=shuffle, random_state=random_state)
        score = []
        for i in range(cv):
            x_train, y_train, x_test, y_test, alpha = split_train_validation(i, k_fold, self.X, self.y, noise)
            GPR = GaussianProcessRegressor(kernel=kernel_gp, alpha=alpha).fit(x_train, y_train)
            predict_y = GPR.predict(x_test)
            if scoring == 'R_MSE':
                score_ = R_MSE_score(y_test, predict_y)
            elif scoring == 'R':
                score_ = Pearson_score(y_test, predict_y)
            elif scoring == 'r2':
                score_ = r_2(y_test, predict_y)
            else:
                err = "The loss function " \
                      f"{scoring} has not been implemented, " \
                      "please choose one of 'R_MSE', 'R', or 'r2'."
                raise NotImplementedError(err)
            score.append(score_)
        return np.mean(score)

    @staticmethod
    def bayes_opt_parameters(function, parameters_boundary: dict, n_iter=25, acquisition_function='ei'):
        """
        Application of package bayes_opt,Only the optimized maximum value can be returned.
        @param n_iter: After n_iter iterations, the optimal parameters are selected
        @param function: Define an objective function that contains parameters that need to be optimized,
                          such as cross validation score.
        @param parameters_boundary: The value range of the parameter needs to be optimized,
                                    which is passed in the form of a dictionary
        @param acquisition_function: Acquisition function: 'ei','ucb' or 'poi',default='ei'
        """
        opt = BayesianOptimization(function,
                                   parameters_boundary,
                                   random_state=10)
        opt.maximize(n_iter=n_iter, acq=acquisition_function)
        print(opt.max['target'])
        print(opt.max['params'])

    @staticmethod
    def BGO_parameters(function, parameters, step, n_iter=50, acq='ei', alpha=1e-10):
        """
        By first generating the parameter space, and then optimizing the parameter space
        through different methods.
        @param alpha: Gaussian process regression noise.default=1e-10
                        Value added to the diagonal of the kernel matrix during fitting.
                        It is suggested to input a large noise when using PES and KD methods.
        @param acq: acquisition function，'ei','ucb','poi','pes' or 'kg'.default='ei'
                    'ei':Expected Improvement
                    'ucb':Upper confidence bound
                    'poi':Probability of Improvement
                    'kg':Knowledge Gradient
                    'pes':Predictive Entropy Search
        @param n_iter: After n_iter iterations, the optimal parameters are selected.default=50
        @param function: Define an objective function that contains parameters that need to be optimized,
                          such as cross validation score.
        @param parameters: The value range of the parameter needs to be optimized,
                                    which is passed in the form of a dictionary
        @param step: Parameter value step.
        """
        n = len(parameters)
        parameters_key = list(parameters.keys())
        virtual_X = get_virtual_sample(parameters, step)
        GPR = GaussianProcessRegressor(kernel=1 * RBF(length_scale=1), alpha=alpha, random_state=10)
        start_X = [i for i in range(n)]
        start_y = [function(*start_X)]
        opt_X = [start_X]
        opt_y = [start_y]
        print('-' * 100)
        for i in range(n_iter):
            GPR.fit(np.array(opt_X), np.array(opt_y))
            mean, std = GPR.predict(virtual_X, return_std=True)
            f_max = max(opt_y)
            if acq == 'ei':
                next_X = EI(mean, std, f_max, virtual_X)  # EI method
                next_target = function(*next_X)
            elif acq == 'ucb':
                next_X = UCB(mean, std, virtual_X, beta=1)  # UCB method
                next_target = function(*next_X)
            elif acq == 'poi':
                next_X = POI(mean, std, f_max, virtual_X)  # POI method
                next_target = function(*next_X)
            elif acq == 'pes':
                next_X = PES(GPR, opt_X, opt_y, mean, std, virtual_X)  # PES method
                next_target = function(*next_X)
            elif acq == 'kg':
                next_X = KD(GPR, opt_X, opt_y, mean, std, virtual_X)  # KD method
                next_target = function(*next_X)
            else:
                err = "The acquisition function " \
                      f"{acq} has not been implemented, " \
                      "please choose one of 'ei','ucb','poi','pes' or 'kg'."
                raise NotImplementedError(err)
            dic = {}
            for k in range(n):
                dic[parameters_key[k]] = next_X[k]
            dic['target'] = round(next_target, 3)
            print(f"The {i + 1}-th optimized:", dic)
            opt_X.append(next_X)
            opt_y.append([next_target])
        print('-' * 100)
        max_target = max(opt_y)
        max_target_index = opt_y.index(max_target)
        opt_parameters = opt_X[max_target_index]
        para_opt = {}
        for j in range(n):
            para_opt[parameters_key[j]] = opt_parameters[j]
        para_opt['target'] = round(max_target[0], 3)
        print("Optimized parameters:", para_opt)

    @staticmethod
    def read_data(fileName):
        data = pd.read_csv(fileName)
        X = data.iloc[:, :-2]
        y = data.iloc[:, -2:-1]
        noise = data.iloc[:, -1:]
        return X, y, noise

