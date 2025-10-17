# ACMO-Diet: Accelerated Constrained Multi-Objective Optimization for Diet Recommendation
## Abstract
ACMO-Diet is an advanced multi-objective optimization framework specifically designed for diet recommendation systems. It addresses the challenge of balancing nutritional adequacy, taste preferences, and dietary diversity in high-dimensional sparse food spaces.
- paper: https://doi.org/10.1109/CISCE65916.2025.11065817

## 🚀 Key Features

- **Sparse-Aware Optimization**: Novel Sparse Mask Crossover (SMX) and Sparse Mask Mutation (SMM) operators for ultra-sparse binary representations (<0.1% density)
- **Multi-Objective Optimization**: Simultaneously optimizes four objectives:
  - Calorie adherence
  - Macronutrient balance  
  - Taste satisfaction
  - Dietary variety
- **Constraint Handling**: Enforces WHO-aligned nutritional constraints with 100% compliance
- **Efficient Performance**: 18.5% faster runtime compared to traditional SBX/PM operators

## 📊 Performance Highlights

- **Highest Hypervolume**: 6.5993 (outperforms 7 state-of-the-art MOO algorithms)
- **Best Convergence**: Generational Distance = 0.01583 (73.2% improvement over NSGA-III)
- **100% Constraint Compliance**: All solutions meet WHO nutritional guidelines
- **Fast Execution**: 33.3s runtime vs 51.2s for SBX_PM baseline

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ACMO-Diet.git
cd ACMO-Diet

# Install dependencies
pip install -r requirements.txt

python evaluate.py
```


# 🧪 Experimental Setup
## Dataset
- 1,274 recipes from AllRecipe.com
- Nutritional profiles: calories, protein, carbohydrates, fat
- User ratings: 1-5 stars
## Baseline Algorithms
- Compared against 7 state-of-the-art MOO algorithms:
  - NSGA-II, NSGA-III, U-NSGA-III
  - MOEA/D, CTAEA
  - AGEMOEA2, SparseEA

## Evaluation Metrics
- Hypervolume (HV)
- Generational Distance (GD)
- Inverted Generational Distance (IGD)
- Spacing
- Compliance Rate

## Result
Find result in ./results/ directory.

Citation
```
@INPROCEEDINGS{11065817,
  author={Zhao, Lei},
  booktitle={2025 IEEE 7th International Conference on Communications, Information System and Computer Engineering (CISCE)}, 
  title={ACMO-Diet: Accelerated Constrained Multi-Objective Optimization for Diet Recommendation}, 
  year={2025},
  volume={},
  number={},
  pages={1249-1254},
  keywords={Constraint handling;Runtime;Planning;Optimization;Recommender systems;Faces;Genetic operators;Information systems;Guidelines;Convergence;Diet recommendation;Constrained multi-objective optimization;Sparse genetic operators;Nutritional constraints},
  doi={10.1109/CISCE65916.2025.11065817}}
```