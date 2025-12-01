# Threshold-Based Naive Bayes (Tb-NB) & Iterative Tb-NB (iTb-NB)

> **A modern, threshold-driven reformulation of the Naive Bayes classifier**
> for binary text classification and decision boundary refinement.

---

##  Overview

This repository implements the **Threshold-Based Naïve Bayes (Tb-NB)** classifier and its iterative variant (iTb-NB) as proposed by Romano, M., Contu, G., Mola, F., & Conversano, C. (2023).
and Romano, M., Zammarchi, G., & Conversano, C. (2024).


The Tb-NB framework extends the classical Naïve Bayes model by introducing a tunable decision threshold (τ) that separates classes in score space.
The threshold is optimized via cross-validation using a set of performance criteria — e.g., accuracy, Matthews Correlation Coefficient, F1-score, etc.
The iterative version (iTb-NB) further refines this threshold locally, improving boundary adaptation and robustness to class overlap.

---

##  Features

- Threshold-based decision rule replacing posterior-based classification
- Cross-validation threshold optimization for using `ThresholdOptimizer`
- Modular design inspired by scikit-learn and compatible with similar API (`fit`, `predict`, `predict_scores`)
- Multiple metrics available for optimization (accuracy, F1, MCC, balanced error, etc.).
- Iterative refinement (iTb-NB) for samples near the decision boundary  
  (as described in Romano et al. 2024).  
- Vectorized computations featuring scipy sparse matrices and numpy 
- Compatible with Pipeline and GridSearchCV operations
---

##  Project Structure

```
tbnnb/
│
├── models/
│   ├── tbnb.py                # Core TbNB classifier
│   ├── threshold.py           # ThresholdOptimizer for τ selection
│   
│
├── utils/
│   ├── validation.py          # sklearn compliant input checks (X, y)
│   ├── decision.py            # Iterative threshold refinement logic
│   ├── confusion.py           # metric computations (TP, FP, F1, MCC, …)
│
├── notebooks/
│   ├── demo_notebook.ipynb    # end-to-end example on sentiment dataset
│
├── preprocessing/
│   ├── nltk_pipeline.py       # simple TextPreprocessor class implementing featuring nltk 
│
└── README.md
```


##  Quick Start

### Installation

Clone and install locally:

```bash
git clone https://github.com/francescotiddia/iTBNB.git
cd tbnb
pip install -e .
```

### Basic Usage

```python
import numpy as np
from scipy.sparse import csr_matrix
from models.tbnb import TbNB

# Example Bag-of-Words dataset
X = csr_matrix([
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1]
])
y = np.array([1, 0, 1, 0])

# Initialize Tb-NB with threshold optimization
model = TbNB(
    alpha=1.0,
    iterative=False,
    optimize_threshold=True,
    criterion="balanced_error",  # or accuracy, f1, mcc, fnr, fpr, …
    K=5
)

model.fit(X, y)

# Predictions
pred = model.predict(X)
scores = model.predict_scores(X)
```

---

##  Optimization Criteria

During training, the threshold is optimized using one of the following criteria:

| Criterion                 | Description                                                        |
|---------------------------|--------------------------------------------------------------------|
| `accuracy`                | Maximizes overall proportion of correctly classified samples        |
| `precision`               | Maximizes proportion of predicted positives that are true positives |
| `recall`                  | Maximizes true positive rate (sensitivity)                          |
| `specificity`             | Maximizes true negative rate                                        |
| `fpr`                     | Minimizes false positive rate                                       |
| `fnr`                     | Minimizes false negative rate                                       |
| `f1`                      | Maximizes harmonic mean of precision and recall                     |
| `mcc`                     | Maximizes Matthews Correlation Coefficient                          |
| `misclassification_error` | Minimizes (FP + FN) / total                                         |
| `balanced_error`          | Minimizes average of FPR and FNR                                    |


---

##  Iterative Refinement (iTb-NB)

The iterative thresholding process focuses on uncertain regions near the decision boundary.
At each step:

1. Identify samples near the current threshold 
2. Estimate local class densities (via KDE).
3. Compute their intersection → new refined τ.
4. Repeat until convergence or insufficient samples.

This yields a sequence of refined thresholds that can adapt the decision boundary locally.


---

##  References

>Romano, M., Contu, G., Mola, F., & Conversano, C. (2023).
Threshold-based Naïve Bayes classifier.
Advances in Data Analysis and Classification, 18, 325–361.
DOI: 10.1007/s11634-023-00536-8

>Romano, M., Zammarchi, G., & Conversano, C. (2024).
Iterative Threshold-Based Naïve Bayes Classifier.
Statistical Methods & Applications, 33, 235–265.
DOI: 10.1007/s10260-023-00721-1
---
