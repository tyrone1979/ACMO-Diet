import numpy as np
import pandas as pd

from tqdm import tqdm

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.util.ref_dirs import get_reference_directions
from sparsega import SparseGACrossover, SparseGAMutation,SparseGASampling


class Evaluator:
    def __init__(self, problem):
        # 读取数据
        self.problem = problem

    def calculate_mean_and_variance(self, data, metrics):
        mean_variance_results = []
        mean_results = []
        # 提取所有算法名称
        algorithms = list({result['Algorithm'] for repeat in data for result in repeat})
        for algo in algorithms:
            row = {"Algorithm": algo}
            row_ = {"Algorithm": algo}
            for metric in metrics:
                values = [result[metric] for repeat in data for result in repeat if result['Algorithm'] == algo]
                # 检查值是否为numpy数组且不为空
                if isinstance(values, list) and len(values) > 0:
                    mean = np.mean(values)
                    var = np.var(values)
                    row[metric] = f"{mean:.4e}({var:.4e})"
                    row_[metric] = mean
            mean_variance_results.append(row)
            mean_results.append(row_)
        return mean_variance_results, mean_results

    def calculate_average_rank(self, results, metrics):
        num_algorithms = len(results)
        num_metrics = len(metrics)
        ranks = np.zeros(num_algorithms)

        for i, metric in enumerate(metrics):
            values = [result[metric] for result in results]
            if metric in ["Hypervolume", "Spread", "Compliance", "Time"]:  # 对于越大越好的指标，使用降序排序
                sorted_indices = np.argsort(-np.array(values))
            elif metric == 'Compliance':
                pass
            else:  # 对于越小越好的指标，使用升序排序
                sorted_indices = np.argsort(np.array(values))

            for j, idx in enumerate(sorted_indices):
                ranks[idx] += (num_algorithms - j)

        avg_ranks = ranks / num_metrics
        return avg_ranks

    def calculate_guideline_compliance(self, X):
        total = len(X)
        compliance = 0
        for i in range(X.shape[0]):
            x = X[i, :]
            selected_foods = self.problem.get_selected_foods(x)
            if self.problem.isCompliance(selected_foods, self.problem.target_nutrients):
                compliance += 1
            else:
                pass
        return compliance / total




class EvaluatorPymoo(Evaluator):

    def __init__(self,algorithms, problem, repeat_times, generations):
        super().__init__(problem)
        self.algorithms=algorithms
        self.repeat_times = repeat_times
        self.generations = generations
        self.F = [[] for _ in range(self.repeat_times)]  # 初始化F为一个包含repeat_times个空列表的列表
        self.X = [[] for _ in range(self.repeat_times)]  # 同样初始化X
        self.T = [[] for _ in range(self.repeat_times)]  #各算法执行的时间。
        self.ref_points = []  # 同样初始化ref_points

    def evaluate_algorithm(self, name, F, X, HV_ref_point, T):
        from pymoo.indicators.hv import HV
        from pymoo.indicators.gd import GD
        from pymoo.indicators.igd import IGD
        from pymoo.indicators.spacing import SpacingIndicator

        # 评估指标
        hv = HV(HV_ref_point)
        hv_value = hv.do(F)

        gd = GD(self.ref_points)
        gd_value = gd.do(F)

        igd = IGD(self.ref_points)
        igd_value = igd.do(F)

        spacing = SpacingIndicator(self.ref_points)
        try:
            spacing_value = spacing.do(F)
        except ValueError as e:
            spacing_value = 1
            X=X.reshape(1,-1)

        compliance = self.calculate_guideline_compliance(X)

        return {
            "Algorithm": name,
            "Hypervolume": hv_value,
            "GD": gd_value,
            "IGD": igd_value,
            "Spacing": spacing_value,
            "Compliance": compliance,
            "Time": T
        }

    def run(self):
        from pymoo.optimize import minimize
        for i in tqdm(range(self.repeat_times), desc='reading data or minimize...'):
            for algo, name in self.algorithms:
                seed = np.random.randint(0, 1000)  # 每次运行时生成不同的随机种子
                try:
                    self.F[i].append(
                        np.genfromtxt(f'./results/{self.problem.name}/{name}/{name}_f_{i}.csv', delimiter=',',
                                      skip_header=1))
                    self.X[i].append(
                        np.genfromtxt(f'./results/{self.problem.name}/{name}/{name}_x_{i}.csv', delimiter=',',
                                      skip_header=1))
                    self.T[i].append(
                        np.genfromtxt(f'./results/{self.problem.name}/{name}/{name}_t_{i}.csv', delimiter=',',
                                      skip_header=1))
                except FileNotFoundError:
                    res = minimize(
                        self.problem,
                        algo,
                        ('n_gen', self.generations),
                        seed=seed,
                        verbose=False,
                        progress=True
                    )
                    if res.F is None:
                        print(f'{name} has no solution')
                        return None, None, None, None

                    self.problem.save_results_csv(self.problem.name, name, res.F, res.X, i, [[res.exec_time]])
                    self.F[i].append(res.F)
                    self.X[i].append(res.X)
                    self.T[i].append([[res.exec_time]])

        self.ref_points, _ = problem.pareto_front(self.F[0], self.X[0])
        return self.F, self.X, self.ref_points, self.T

    def evaluate(self, HV_ref_point):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        results = [[] for _ in range(self.repeat_times)]
        for i in tqdm(range(self.repeat_times), desc='evaluating...'):
            for idx, (algo, name) in enumerate(self.algorithms):
                result = self.evaluate_algorithm(name, self.F[i][idx], self.X[i][idx], HV_ref_point, self.T[i][idx])
                results[i].append(result)

        metrics = ["Hypervolume", "GD", "IGD", "Spacing", "Compliance", 'Time']
        mean_variance_results, mean_results = self.calculate_mean_and_variance(results, metrics)
        df_mean_variance = pd.DataFrame(mean_variance_results)

        # 平均排名评估
        metrics = ["Hypervolume", "GD", "IGD", "Spacing"]
        avg_ranks = self.calculate_average_rank(mean_results, metrics)
        df_mean_variance["Average Rank"] = avg_ranks
        print(df_mean_variance.sort_values(by="Average Rank", ascending=False))


def show_graph(TF, F1, F2):
    from pymoo.visualization.scatter import Scatter

    '''
    plot = Petal(bounds=[0, 2.0],
                 labels=['Calories', 'Nutrients', 'Taste', 'Variety'], reverse=True,
                 title=["Solution %s" % t for t in range(0, 5)])
    for i in range(0, 5):
        plot.add(np.abs(F1[i]))
    '''

    plot = Scatter(labels=['Calories', 'Nutrients', 'Taste', 'Variety'])
    plot.add(TF, marker='^')
    plot.add(F1, marker='o')
    plot.add(F2, marker='x')
    plot.show()
    #PCP().add(F1).show()


if __name__ == "__main__":
    import argparse
    from objective_function import Nutrition4

    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Evaluate with reference directions")
    parser.add_argument("-p", type=int, default=8, required=False,
                        help="Number of partitions for reference directions")
    parser.add_argument("-r", type=int, default=10, required=False,
                        help="Repeat times")
    parser.add_argument("-g", type=int, default=100, required=False,
                        help="Generations")
    parser.add_argument("-e", type=str, default="b", required=False,
                        help="b=baseline or o=operator")
    parser.add_argument("-f", type=bool, default=True, required = False,
                        help="show graph")
    food_data = pd.read_csv("./data/food_data.csv")
    food_data['calories'] = food_data['calories'] / 1000.0
    food_data['protein'] = food_data['protein'] / 100.0
    food_data['carbs'] = food_data['carbs'] / 100.0
    food_data['fat'] = food_data['fat'] / 100.0
    food_data['rating'] = food_data['rating'] / 5.0

    # 解析命令行参数
    args = parser.parse_args()
    target_nutrients = {'calories': (2.0, 2.0), 'protein': (0.5, 1.75), 'carbs': (2.25, 3.25), 'fat': (0.44, 0.78),
                        'variety': (3, 12), 'cook_time': (0, 1.0)}
    hv_ref_point4 = np.array([5.0, 3.0, -0.8, 1.0])


    problem = Nutrition4(
        food_database=food_data,
        target_nutrients=target_nutrients
    )
    dim = len(problem.food_database)
    parameters_baseline = {"sampling": SparseGASampling(dim=dim),
                           "crossover": SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
                           "mutation": PM(prob=1.0, eta=20, vtype=float, repair=RoundingRepair()),
                           "ref_dirs": get_reference_directions("das-dennis", problem.n_obj,
                                                                n_partitions=args.p)
                           }

    algorithms_baseline = [
        (UNSGA3(ref_dirs=parameters_baseline['ref_dirs'],
                sampling=parameters_baseline['sampling'],
                crossover=SparseGACrossover(),
                mutation=SparseGAMutation()), "Sparse+U-NSGA-III"),
        (UNSGA3(ref_dirs=parameters_baseline['ref_dirs'],
                sampling=parameters_baseline['sampling'],
                crossover=parameters_baseline['crossover'],
                mutation=parameters_baseline['mutation']), "U-NSGA-III"),
        (CTAEA(ref_dirs=parameters_baseline['ref_dirs'],
               sampling=parameters_baseline['sampling'],
               crossover=parameters_baseline['crossover'],
               mutation=parameters_baseline['mutation']), "CTAEA"),
        (AGEMOEA(sampling=parameters_baseline['sampling'],
                 crossover=parameters_baseline['crossover'],
                 mutation=parameters_baseline['mutation'],
                 eliminate_duplicates=True), "AGEMOEA"),
        (NSGA2(sampling=parameters_baseline['sampling'],
               crossover=parameters_baseline['crossover'],
               mutation=parameters_baseline['mutation'],
               eliminate_duplicates=True), "NSGA-II"),
        (AGEMOEA2(sampling=parameters_baseline['sampling'],
                  crossover=parameters_baseline['crossover'],
                  mutation=parameters_baseline['mutation'],
                  eliminate_duplicates=True), "AGEMOEA2"),
        (SMSEMOA(sampling=parameters_baseline['sampling'],
                 crossover=parameters_baseline['crossover'],
                 mutation=parameters_baseline['mutation'],
                 eliminate_duplicates=True), "SMSEMOA"),
        (NSGA3(ref_dirs=parameters_baseline['ref_dirs'],
               sampling=parameters_baseline['sampling'],
               crossover=parameters_baseline['crossover'],
               mutation=parameters_baseline['mutation']), "NSGA-III"),
    ]
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.crossover.ux import UniformCrossover
    from pymoo.operators.mutation.gauss import GM
    from pymoo.operators.mutation.bitflip import BFM

    algorithms_operators = [
        (UNSGA3(ref_dirs=parameters_baseline['ref_dirs'],
                sampling=parameters_baseline['sampling'],
                crossover=SparseGACrossover(),
                mutation=SparseGAMutation()), "Sparse"),
        (UNSGA3(ref_dirs=parameters_baseline['ref_dirs'],
                sampling=parameters_baseline['sampling'],
                crossover=parameters_baseline['crossover'],
                mutation=parameters_baseline['mutation']), "SBX+Polynomial"),
        (UNSGA3(ref_dirs=parameters_baseline['ref_dirs'],
                sampling=parameters_baseline['sampling'],
                crossover=parameters_baseline['crossover'],
                mutation=GM()), "SBX+Guass"),
        (UNSGA3(ref_dirs=parameters_baseline['ref_dirs'],
                sampling=parameters_baseline['sampling'],
                crossover=parameters_baseline['crossover'],
                mutation=BFM()), "SBX+Bitflip"),
        (UNSGA3(ref_dirs=parameters_baseline['ref_dirs'],
                sampling=parameters_baseline['sampling'],
                crossover=UniformCrossover(),
                mutation=parameters_baseline['mutation']), "UX+Polynomial"),
        (UNSGA3(ref_dirs=parameters_baseline['ref_dirs'],
                sampling=parameters_baseline['sampling'],
                crossover=UniformCrossover(),
                mutation=GM()), "UX+Guass"),
        (UNSGA3(ref_dirs=parameters_baseline['ref_dirs'],
                sampling=parameters_baseline['sampling'],
                crossover=UniformCrossover(),
                mutation=BFM()), "UX+Bitflip"),
    ]
    algorthms = algorithms_baseline if args.e == 'b' else algorithms_operators

    evaluator = EvaluatorPymoo(algorthms,problem, args.r, args.g)
    F, X, TF, T = evaluator.run()

    if F is None:
        print('No solution. End!')
        exit(-1)
    if args.f==False:
        evaluator.evaluate(hv_ref_point4)
    else:
        show_graph(TF,F[0][0],F[0][1])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    #print(problem.recommend_foods(X[0][1]))
