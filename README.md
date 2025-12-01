# Threshold-Based Naive Bayes (Tb-NB) & Iterative Tb-NB (iTb-NB)

> **A modern, threshold-driven reformulation of the Naive Bayes classifier**
> for binary text classification and decision boundary refinement.

---

##  Overview

This repository implements the **Threshold-Based Naïve Bayes (Tb-NB)** classifier and its iterative variant (iTb-NB) as proposed by:

> Romano, M., Zammarchi, G., & Conversano, C. (2024).
> *Iterative Threshold-Based Naïve Bayes Classifier*.
> *Statistical Methods & Applications, 33*, 235–265.
> [https://doi.org/10.1007/s10260-023-00721-1](https://doi.org/10.1007/s10260-023-00721-1)

The Tb-NB framework extends the classical Naïve Bayes model by introducing a tunable decision threshold (τ) that separates classes in score space.
The threshold is optimized via cross-validation using a set of performance criteria — e.g., accuracy, Type I or Type II error balance.
The iterative version (iTb-NB) further refines this threshold locally, improving boundary adaptation and robustness to class overlap.

---

##  Features

- Threshold-based decision rule replacing posterior-based classification
- Cross-validation optimization for τ using `ThresholdOptimizer`
- Laplace smoothing and empirical prior estimation
- Iterative refinement (iTb-NB) via localized density intersections
- Modular design compatible with scikit-learn–like interfaces (`fit`, `predict`, `predict_scores`)


---

##  Project Structure

```
tbnnb/
│
├── models/
│   ├── tbnb.py                # Core TbNB classifier
│   ├── threshold         # ThresholdOptimizer for τ selection
│   
│
├── utils/
│   ├── validation.py          # sklearn compliant input checks (X, y)
│   ├── decision.py            # Iterative threshold refinement logic
│
├── examples/
│   ├── demo_notebook.ipynb    # End-to-end example on sentiment dataset
│
└── README.md
```


##  Quick Start

### Installation

Clone and install locally:

```bash
git clone https://github.com/francescotiddia/tbnb.git
cd tbnb
pip install -e .
```

### Basic Usage

```python
import numpy as np
from scipy.sparse import csr_matrix
from models.tbnb import TbNB

# Example binary Bag-of-Words dataset
X = csr_matrix([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
y = np.array([1, 0, 1, 0])

# Train Tb-NB and optimize threshold
model = TbNB(alpha=1.0, iterative=False)
model.fit(X, y, criterion="accuracy", K=3)

# Predict on new samples
preds = model.predict(X)
scores = model.predict_scores(X)
```

---

##  Optimization Criteria

During training, the threshold ( \tau ) is optimized using one of the following criteria:

| Criterion            | Description                              |
| -------------------- | ---------------------------------------- |
| `accuracy`           | Maximizes mean classification accuracy   |
| `type_I`             | Minimizes false positive rate            |
| `type_II`            | Minimizes false negative rate            |
| `equal` / `balanced` | Minimizes total (Type I + Type II) error |

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

* **Romano, M., Zammarchi, G., & Conversano, C. (2024)**.
  *Iterative Threshold-Based Naïve Bayes Classifier*.
  *Statistical Methods & Applications, 33*, 235–265.
  [https://doi.org/10.1007/s10260-023-00721-1](https://doi.org/10.1007/s10260-023-00721-1)

---
