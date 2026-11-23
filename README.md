This is a readme for mini-project-01 by tinkerx01.
-
Section.1: Introduction
-
```
In this project I will train:
a) Logisitic Regression model
b) Knn classifier model with k = 1 , 5 , 20
c) Decision tree model with max depth = 10
on a 80/20 test-train split of Kaggle's creditcardfraud dataset.
```

Section.2: Hypothesis
-
```
h1. I expect knn to be better than other models because it will capture the non-linearity between features.
h2. I think recall is very important here, considering the classes are imbalanced: 
(Class 0): 284315  (99.827%)
Fraud (Class 1): 492  (0.173%)
and missing frauds is more costly than detecting normal samples as fraud. f1 can also help us better see the model's performance on this dataset.
h3. If the model predicts all inputs as legitimate, we expect to get high accuracy, because the imbalance is not applied or detectable in that metric.
h4. If features like time and amount are not scaled, they might dominate the other metrics because the others contain negative values and amount can get pretty huge
compred to the other ranges. 
h5. If decision tree is applied without limit it might overfit. And because of the imbalance, it can isolate the fraudulent data and thus, memorize the training
data for class 1 instead of finding a generalizable pattern.
```


