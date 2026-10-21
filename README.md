# Matrix Completion for Recommender Systems: Analysis of SGD Optimization on MovieLens 1M

An end-to-end data science and machine learning project implementing **Low-Rank Matrix Factorization** via **Stochastic Gradient Descent (SGD)** for **Matrix Completion** in Recommender Systems. The project demonstrates how **Spectral Initialization** (Truncated SVD) overcomes non-convexity challenges in sparse rating matrices compared to standard Random Initialization, evaluated on the **MovieLens 1M** dataset.

Developed as the final term project for the **Signal Processing and Optimization for Big Data (SPOBD)** course (A.Y. 2025–2026) at the **University of Perugia (UniPG)** – Department of Engineering.

---

## Project Overview

In the era of Big Data, recommendation engines must process extremely sparse user-item interaction matrices. This project addresses the **Matrix Completion** problem by predicting unobserved user ratings for movies based on latent factor modeling ($R \approx U V^T$).

Key contributions include:
- **Theoretical Formulation & Derivations**: Complete mathematical derivations of Frobenius norm loss with $L_2$ regularization, partial derivatives, and SGD update rules.
- **Spectral vs. Random Initialization**: Empirical proof that Truncated SVD initialization ($U_0 = U_r \Sigma_r^{1/2}$, $V_0 = V_r \Sigma_r^{1/2}$) prevents gradient explosion and ensures faster, deeper convergence compared to Gaussian random initialization.
- **Multidimensional Hyperparameter Tuning**: Systematic 2D Grid Search over learning rate ($lpha$) and regularization ($eta$), followed by model complexity analysis across latent ranks ($r \in \{10, 20, 50, 100\}$) with heuristic regularization adaptation.
- **Diagnostic 4-Panel Visualization**: Visual decomposition separating known ratings, training fit, unobserved predictions ("hallucination"), and the fully dense completed matrix.
- **Top-$N$ Recommendation Engine**: An inference module generating personalized top-5 movie recommendations alongside ground-truth comparison.

---

## Theoretical Background & Core Algorithms

### 1. Problem Formulation
Let $R \in \mathbb{R}^{m \times n}$ be the sparse user-movie rating matrix ($m = 6,040$ users, $n = 3,706$ movies). Observed ratings are restricted to sample set $\Omega$:
$$P_\Omega(R)_{i,j} = \begin{cases} R_{i,j} & \text{if } (i,j) \in \Omega \\ 0 & \text{otherwise} \end{cases}$$

Assuming low rank $r \ll \min(m, n)$, we factorize $R \approx U V^T$ where $U \in \mathbb{R}^{m \times r}$ (user latent factors) and $V \in \mathbb{R}^{n \times r}$ (movie latent factors).

### 2. Regularized SGD Optimization
We minimize the regularized Frobenius loss:
$$\min_{U,V} \sum_{(i,j) \in \Omega} \left( u_i^T v_j - r_{i,j} \right)^2 + \beta \left( \|u_i\|_2^2 + \|v_j\|_2^2 \right)$$

Iterative updates for sample $(i, j, r_{i,j})$ at step $k$:
$$u_i^{(k)} \leftarrow u_i^{(k-1)} + \alpha \left( e_{i,j} v_j^{(k-1)} - \beta u_i^{(k-1)} \right)$$
$$v_j^{(k)} \leftarrow v_j^{(k-1)} + \alpha \left( e_{i,j} u_i^{(k-1)} - \beta v_j^{(k-1)} \right)$$
where $e_{i,j} = r_{i,j} - u_i^T v_j$.

### 3. Spectral Initialization vs. Random Initialization
- **Random Initialization**: $U, V \sim \mathcal{N}(0, 1/r^2)$ scaled to prevent initial gradient explosion. Slower convergence and risk of local minima stagnations.
- **Spectral Initialization**: Truncated SVD decomposition $R_\Omega \approx U_r \Sigma_r V_r^T$ setting initial weights to $U^{(0)} = U_r \Sigma_r^{1/2}$ and $V^{(0)} = V_r \Sigma_r^{1/2}$. Balances energy across factors and accelerates SGD convergence.

---

## Experimental Results & Workflow

### Dataset Metrics (MovieLens 1M)
- **Users**: 6,040
- **Movies**: 3,706
- **Total Ratings**: ~1,000,209
- **Sparsity**: **95.53%**
- **Data Split**: 80% Train / 10% Validation / 10% Test

### Experimental Phases
1. **Phase 1: Initialization Comparison ($r=50, \alpha=0.005, \beta=0.02$)**
   - **Spectral Init Min RMSE**: `0.8653`
   - **Random Init Min RMSE**: `0.9307`
   - *Result*: Spectral initialization achieves significantly faster convergence and lower validation error.
2. **Phase 2: 2D Grid Search ($lpha \times \beta$)**
   - Evaluated $\alpha \in \{0.001, 0.005, 0.01, 0.02\}$ and $\beta \in \{0.01, 0.02, 0.05, 0.1\}$.
   - **Optimal Configuration**: $\alpha^* = 0.01$, $\beta^* = 0.05$.
3. **Phase 3: Rank Sensitivity Analysis ($r \in \{10, 20, 50, 100\}$)**
   - Heuristic regularization applied for high ranks ($\beta \leftarrow 2\beta$ for $r \ge 100$).
   - **Optimal Latent Rank**: $r^* = 50$. Higher ranks ($r=100$) suffered from overfitting due to excess variance relative to dataset density.
4. **Phase 4: Diagnostic 4-Panel Matrix Completion Visualization**
   - **Panel 1**: Sparse Original Data ($50 \times 50$ matrix slice).
   - **Panel 2**: Training Fit (Reconstruction of known values).
   - **Panel 3**: Inference / "Hallucination" (Filled unobserved entries).
   - **Panel 4**: Fully Completed Rating Matrix ready for Top-$N$ recommendations.

---

## Repository Structure

```
.
├── SPOBD_project_Matrix_Completion_Movie_Rank.pdf   # Complete academic report (LaTeX formatted PDF)
└── SPOBD_project_Matrix_Completion_Movie_Rank.ipynb # Complete Jupyter / Google Colab notebook
```

### File Descriptions
- **`SPOBD_project_Matrix_Completion_Movie_Rank.pdf`**: The official academic report detailing mathematical proofs, optimization theory, experimental setup, learning curves, and diagnostic heatmaps.
- **`SPOBD_project_Matrix_Completion_Movie_Rank.ipynb`**: The interactive Python notebook containing:
  1. Automated dataset download & extraction (`ml-1m.zip`).
  2. Sparse matrix generation (`scipy.sparse.coo_matrix`).
  3. Optimized SGD training routines with early stopping.
  4. Sensitivity plotting modules using `matplotlib` & `seaborn`.
  5. 4-Panel diagnostic heatmap visualizer.
  6. Inference engine for generating user recommendations.

---

## Tech Stack & Dependencies

- **Language**: Python 3.10+
- **Data Manipulation & Sparse Algebra**: `numpy`, `pandas`, `scipy` (`scipy.sparse.linalg.svds`)
- **Machine Learning & Evaluation**: `scikit-learn` (`train_test_split`, `metrics`)
- **Visualization**: `matplotlib`, `seaborn`

---

## How to Run

1. **Open in Google Colab or Jupyter Notebook**:
   Launch `SPOBD_project_Matrix_Completion_Movie_Rank.ipynb` in your environment.
2. **Execute All Cells**:
   The notebook automatically downloads the MovieLens 1M dataset from GroupLens:
   ```bash
   wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
   ```
3. **Run Inference & Recommendations**:
   Call the recommendation helper function at the end of the notebook:
   ```python
   recommend_movies(user_id_real=1, U=U_opt, V=V_opt, n=5)
   ```

---

## Academic Context

**Course**: Signal Processing and Optimization for Big Data (SPOBD)  
**Degree**: M.S. in Computer and Robotics Engineering – Curriculum Data Science  
**Institution**: Department of Engineering, University of Perugia (UniPG)  
**Instructor**: Prof. Paolo Banelli  
**Academic Year**: 2025/2026  

**Author**:
- **Aurora Traversini**
