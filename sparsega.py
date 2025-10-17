import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling


class SparseGASampling(Sampling):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def _do(self, problem, n_samples, **kwargs):
        # 创建一个全0的矩阵，形状为 (n_samples, self.dim)
        X = np.zeros((n_samples, self.dim), dtype=int)

        for i in range(n_samples):
            indices = np.random.choice(self.dim, 12, replace=False)
            X[i, indices] = 1

        return X


class SparseGACrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)  # 两个父代生成一个子代

    def _do(self, problem, X, **kwargs):
        # 获取父代种群的决策变量（本身就是掩码）
        parents_mask = X
        n_parents, n_matings, n_var = X.shape
        off_mask = np.zeros((self.n_offsprings, n_matings, n_var))

        for i in range(n_matings):
            parent1_mask = parents_mask[0, i]
            parent2_mask = parents_mask[1, i]

            # 交叉操作
            cross_point = np.random.randint(0, n_var)
            child_mask = np.concatenate([parent1_mask[:cross_point], parent2_mask[cross_point:]])

            # 稀疏性控制：以 50% 的概率选择一个位置进行调整
            if np.random.rand() < 0.5:
                idx = np.where(child_mask == 1)[0]
                if len(idx) > 0:
                    selected_idx = idx[np.random.choice(len(idx), 1)[0]]
                    child_mask[selected_idx] = 0
            else:
                idx = np.where(child_mask == 0)[0]
                if len(idx) > 0:
                    selected_idx = idx[np.random.choice(len(idx), 1)[0]]
                    child_mask[selected_idx] = 1

            off_mask[0, i] = child_mask  # 填充到正确的维度

        # 返回子代种群，决策变量本身就是掩码
        return off_mask


class SparseGAMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        # X 是种群的特征矩阵，每行表示一个个体（掩码）
        N, D = X.shape
        X_mutated = X.copy()

        for i in range(N):
            # 随机决定是否进行变异
            if np.random.rand() < 0.5:
                # 找到掩码为1的位置
                index = np.where(X_mutated[i] == 1)[0]
                if len(index) > 0:
                    # 随机选择一个位置将其设为0
                    selected_idx = np.random.choice(index)
                    X_mutated[i, selected_idx] = 0
            else:
                # 找到掩码为0的位置
                index = np.where(X_mutated[i] == 0)[0]
                if len(index) > 0:
                    # 随机选择一个位置将其设为1
                    selected_idx = np.random.choice(index)
                    X_mutated[i, selected_idx] = 1

        return X_mutated

