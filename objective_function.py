import numpy as np
from pymoo.core.problem import Problem
import csv
import os


class NutritionOptimizationProblem(Problem):
    def __init__(self, food_database, target_nutrients, n_obj=3, n_constr=0):
        super().__init__(n_var=len(food_database), n_obj=n_obj, n_constr=n_constr, xl=0, xu=1, type_var=int)
        self.food_database = food_database
        self.target_nutrients = target_nutrients
        self.name = "3_objectives"

    def get_selected_foods(self, x):
        selected_indices = np.where(x == 1)[0]
        selected_foods = self.food_database.iloc[selected_indices]
        return selected_foods

    def isCompliance(self, selected_foods, target_nutrients):
        # 假设guidelines是预定义的字典，如{'protein': (min, max), 'calories': (min, max), ...}
        current = self._calculate_current_nutrients(selected_foods)
        compliant_nutrients = 0
        excluded_keys = {'calories', 'cook_time', 'variety'}
        for nutrient, (min_val, max_val) in target_nutrients.items():
            if nutrient not in excluded_keys:
                if min_val <= current.get(nutrient, 0) <= max_val:
                    compliant_nutrients += 1
        if compliant_nutrients / (len(target_nutrients) - len(excluded_keys)) == 1.0:  # 返回合规比例
            return True
        return False

    def _evaluate(self, X, out, *args, **kwargs):
        pass

    def _merge_pareto_fronts(self, all_F, all_X):
        """合并所有帕累托前沿"""
        return np.vstack(all_F), np.vstack(all_X)

    def _extract_pareto_front(self, merged_F, merged_X):
        """从合并后的解集中提取新的帕累托前沿"""
        n_points = merged_F.shape[0]
        pareto_front = []
        best_solution = []
        for i in range(n_points):
            is_dominated = False
            for j in range(n_points):
                if i != j and np.all(merged_F[j] <= merged_F[i]) and np.any(merged_F[j] < merged_F[i]):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(merged_F[i])
                best_solution.append(merged_X[i])
        return np.array(pareto_front), np.array(best_solution)

    def _calc_pareto_front(self, all_F, all_X):
        """计算综合的帕累托前沿"""
        merged_F, merged_X = self._merge_pareto_fronts(all_F, all_X)
        best_pareto_front, best_solution = self._extract_pareto_front(merged_F, merged_X)
        return best_pareto_front, best_solution

    def pareto_front(self, *args, **kwargs):
        from pymoo.util.misc import at_least_2d_array
        pf, px = self._calc_pareto_front(*args, **kwargs)
        pf = at_least_2d_array(pf, extend_as='r')
        px = at_least_2d_array(px, extend_as='r')
        return pf, px

    def _calculate_nutritional_deviation(self, current_nutrients):
        deviation = 0.0
        for nutrient, (min, max) in self.target_nutrients.items():
            if nutrient in current_nutrients:
                deviation += np.abs(current_nutrients[nutrient] - min) if current_nutrients[nutrient] < min else np.abs(
                    current_nutrients[nutrient] - max)
        return deviation

    def _calculate_current_calories(self, selected_foods):
        current_calories = {}
        for nutrient in selected_foods.columns:
            if nutrient == 'calories':
                current_calories[nutrient] = selected_foods[nutrient].sum()
        return current_calories

    def _calculate_current_nutrients(self, selected_foods):
        current_nutrients = {}
        excluded_keys = {'food_name', 'rating', 'category', 'calories', 'total_time'}
        for nutrient in selected_foods.columns:
            if nutrient not in excluded_keys:
                current_nutrients[nutrient] = selected_foods[nutrient].sum()
        return current_nutrients

    def save_results_csv(self, problem_name, subdir, F, X, i,T):
        # 确保目录存在
        path = f'../results/{problem_name}/{subdir}'
        self._ensure_directory_exists(path)

        # 保存 F 结果
        filename_f = f'{path}/{subdir}_f_{i}.csv'
        with open(filename_f, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f'f{i + 1}' for i in range(F.shape[1])])
            for row in F:
                writer.writerow(row)
        # 保存 X 结果
        filename_x = f'{path}/{subdir}_x_{i}.csv'
        with open(filename_x, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f'x{i + 1}' for i in range(X.shape[1])])
            for row in X:
                writer.writerow(row)
        filename_t = f'{path}/{subdir}_t_{i}.csv'
        with open(filename_t, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['time'])
            for row in T:
                writer.writerow(row)

    def _ensure_directory_exists(self, dir_path):
        """确保目录存在，如果不存在则创建"""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"目录已创建：{dir_path}")
        else:
            print(f"目录已存在：{dir_path}")

    def recommend_foods(self, x):
        selected_indices = np.where(x[0] == 1)[-1]
        foods = self.food_database.iloc[selected_indices]
        return foods


class Nutrition4(NutritionOptimizationProblem):
    def __init__(self, food_database, target_nutrients, n_obj=4,n_constr=8):
        super().__init__(food_database, target_nutrients, n_obj, n_constr)
        self.food_database = food_database
        self.target_nutrients = target_nutrients
        self.name = "4_objectives"
        self.penalty_weight = 0.6
    '''
    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((X.shape[0], self.n_obj))
        G = np.zeros((X.shape[0], self.n_constr))
        for i in range(X.shape[0]):
            x = X[i, :]
            selected_indices = np.where(x == 1)[0]
            selected_foods = self.food_database.iloc[selected_indices]

            # 营养充足性
            current_calories = self._calculate_current_calories(selected_foods)
            calories_deviation = self._calculate_nutritional_deviation(current_calories)
            # 计算当前营养素
            current_nutrients = self._calculate_current_nutrients(selected_foods)

            # 计算营养偏差
            nutritional_deviation = self._calculate_nutritional_deviation(current_nutrients)

            # 口味偏好
            selected_indices = np.where(x == 1)[0]
            if len(selected_indices) > 0:
                taste_preference = selected_foods['rating'].values.mean()
            else:
                taste_preference = 0.0

            # 多样性
            variety = len(selected_foods)
            #total_time = selected_foods['total_time'].values.mean()

            # 目标函数值
            F[i, 0] = calories_deviation
            F[i, 1] = nutritional_deviation
            F[i, 2] = -taste_preference
            F[i, 3] = -variety / 10.0
            #F[i, 4] = total_time
            G[i, 0] = self.target_nutrients['protein'][0] - current_nutrients['protein']
            G[i, 1] = current_nutrients['protein'] - self.target_nutrients['protein'][1]
            G[i, 2] = self.target_nutrients['carbs'][0] - current_nutrients['carbs']
            G[i, 3] = current_nutrients['carbs'] - self.target_nutrients['carbs'][1]
            G[i, 4] = self.target_nutrients['fat'][0] - current_nutrients['fat']
            G[i, 5] = current_nutrients['fat'] - self.target_nutrients['fat'][1]
            G[i, 6] = self.target_nutrients['variety'][0] - variety
            G[i, 7] = variety - self.target_nutrients['variety'][1]
        out["F"] = F
        out["G"] = G
'''

    def _evaluate(self, X, out, *args, **kwargs):
        X_bool = X.astype(bool)
        # 确保营养素列名匹配（示例）
        nutrient_cols = ['protein', 'carbs', 'fat']  # 实际需与target_nutrients的键一致
        # 营养计算矩阵（网页7的向量化操作）
        nutrient_values = self.food_database[nutrient_cols].values  # 形状(n_foods, 3)
        current_nutrients = X_bool @ nutrient_values  # 形状(n_samples,3)
        # 热量计算（网页6的内置函数优化）
        calories = X_bool @ self.food_database['calories'].values


        target_cal = self.target_nutrients['calories'][0]
        F_calories = np.abs(calories - target_cal)

        # 目标2：营养偏差（网页1的向量化约束处理）
        target_mins = np.array([self.target_nutrients[k][0] for k in nutrient_cols])[None, :]
        target_maxs = np.array([self.target_nutrients[k][1] for k in nutrient_cols])[None, :]
        F_nutrition = np.sum(
            np.where(current_nutrients < target_mins, target_mins - current_nutrients,
                     np.where(current_nutrients > target_maxs, current_nutrients - target_maxs, 0)),
            axis=1
        )


        # 目标3：口味偏好（网页3的均值向量化）
        ratings = self.food_database['rating'].values
        sum_ratings = X_bool @ ratings
        count_selected = X_bool.sum(axis=1)
        F_taste = -sum_ratings / np.maximum(count_selected, 1)
        F_variety = -count_selected / 10.0

        # 约束条件向量化（网页4的约束处理）-----------------------------------------
        G = np.empty((X.shape[0], self.n_constr))
        # 蛋白质约束
        G[:, 0] = self.target_nutrients['protein'][0] - current_nutrients[:, 0]
        G[:, 1] = current_nutrients[:, 0] - self.target_nutrients['protein'][1]
        # 碳水约束
        G[:, 2] = self.target_nutrients['carbs'][0] - current_nutrients[:, 1]
        G[:, 3] = current_nutrients[:, 1] - self.target_nutrients['carbs'][1]
        # 脂肪约束
        G[:, 4] = self.target_nutrients['fat'][0] - current_nutrients[:, 2]
        G[:, 5] = current_nutrients[:, 2] - self.target_nutrients['fat'][1]
        # 多样性约束
        G[:, 6] = self.target_nutrients['variety'][0] - count_selected
        G[:, 7] = count_selected - self.target_nutrients['variety'][1]

        out["F"] = np.column_stack([F_calories, F_nutrition, F_taste, F_variety])
        out["G"] = G
