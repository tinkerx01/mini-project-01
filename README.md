This is a readme for mini-project-01 by tinkerx01.
-
## 1. Introduction
-
```
problem description: we aim to design a model to detect fraudulent activity from normal in creditcard transactions.

In this project following models are trained:
a) Logisitic Regression model
b) Knn classifier model with k = 1 , 5 , 20
c) Decision tree model with max depth = 10
on a 80/20 test-train split of Kaggle's creditcardfraud dataset.
```

## 2. Data Analysis
* Dataset statistics
```
Number of Samples: 284807
Number of features: 30
Normal (Class 0): 284315  (99.827%)
Fraud (Class 1): 492  (0.173%)
```
* Feature information

feature | type | range |
| -: | -: | -: |
V1 - 19V | float | -113 - 120 |
Time | positive ascending digit | 1 - 30|
Amount | positive digit | 1 - 172792 |

* Missing value analysis
```
no missing value from time, amount, class
duplicate values detected.
```
## 3. Initial Hypothesis

Explain your expected model behavior.
```
The model is expected to have low FN(low recall) and high TN(imbalance) detection. which with the given threshold (0.7) becomes possible.
```
## 4. Model Comparison

Include:

* Models
```
knn n = 1, 5, 20
decision tree max dept 10
logistice regression
```
* Metrics
see derived JSON files.

* Cross-validation results

| Model | Mean Precision | Mean Recall | Mean F1 |
|-: |-: |-: |-: |
| Logistic Regression | 0.8752 | 0.6421 | 0.7396 |
| KNN | 0.9269 | 0.7589 | 0.834 |
| Decision Tree | 0.8666 | 0.7639 | 0.8112 |

## 5. Scaling Experiment

Explain the effect of scaling.
```
scaling helps the models that use distance for training and prediction like LR and KNN.
```

## 6. Hyperparameter Experiment

Explain the impact of parameter changes.
```
applying threshold on knn help it better detect TN and TP. Increaning precision and reducing recall.
```

## 7. Final Model Selection

Explain why your final model was selected.
because it performed as expected (see experiments)

## 8. Running Instructions

Explain:

* Installation
you will need a seperate module for orchestration that operates as main. I used  a jupyter notebook. call functions and use variables from within each module. 
* Training
use trainModel()
* Prediction
train and use reportTest or reportTrain to see the results and save them in a JSON.

# 18. Reflection Questions

### Question 1

Why is Accuracy a misleading metric for this dataset?
because it has imbalance in classes.
---

### Question 2

What is the trade-off between detecting more fraudulent transactions and generating more false alarms?
the first is less costly than the second one. so we try to get as less recall as we can.

---

### Question 3

If you had one additional week, what would you improve in your fraud detection system?
- I would clean the code further
- I would like to have better input parameters for each function
- I would try to increase cohesion and reduce coupling within modules.
- I would like to achieve easier orchestration for running each operation.

---