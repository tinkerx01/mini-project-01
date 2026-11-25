
# Section.2: Hypothesis
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


# Section 3: Getting the reports on one split
-
```
- Was initial hyposesis correct? yes
- Knn with default neighbors (5) was the most reasonable.
- f1, recall, precision
- accuracy was very high and not possible to know if it's due to overfitting or learning
- I still get more FN than FP which is not good and should be fixed.
``` 


# 9. Cross Validation
| Model | Mean Precision | Mean Recall | Mean F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.8752 | 0.6421 | 0.7396 |
| KNN | 0.9269 | 0.7589 | 0.834 |
| Decision Tree | 0.8666 | 0.7639 | 0.8112 |

# 10. Mandatory Experiment 1: Effect of Scaling
| Model | Scaling | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| KNN | Without Scaling | 0 | 0 | 0 |
| KNN | With Scaling | 0.9186 | 0.8061 | 0.8587 |

- Why is KNN sensitive to scaling?
because it calculates distance between a and b. so the range of a and b can greatly matter in the product.

- Why is Decision Tree less sensitive?
decision tree does not care about distance. the question decision tree asks is whether the data has a certain condition or not. if a's value in feature-1 is 500 and b's is 234 it doesn't affect a being 1 in feature-2 and b being 0.78. they are checked seperatly.


# 11. Mandatory Experiment 2: Hyperparameter Analysis

for knn
| Parameter Value | Precision | Recall | F1 |
|---|---|---:|---:|
|1|0.8454|0.8367|0.841|
|5|0.9186|0.8061|0.8587|
|20|0.8409|0.7551|0.7957|

- Did overfitting occur?
yes it did with k=1
- Which value provides the best balance?
k=5. reasonable precision and recall, high f1.


# 12. Mandatory Experiment 3: Impact of Classification Threshold
knn 5 neighbors
| Threshold | Precision | Recall | F1 |
| --------: | --------: | -----: | -:      |
|       0.3 |   0.919   | 0.835  |  0.875  |
|       0.5 |   0.9571  | 0.7919 |  0.8667 |
|       0.7 |   0.9866  | 0.7487 |  0.8514 |

- What happens to Recall when the threshold decreases?
recall is increased.
- What happens to Precision?
it decreases
- Which threshold would you recommend for a fraud detection system?
0.7 because we need smaller recall.
- What trade-off does your chosen threshold create?
less f1 but also less recall.

# 13. Final Model Selection

1. Select one final model.
knn k=5
2. Select an appropriate classification threshold.
0.7
3. Explain your decisions.
```
based on resulting metrics knn is the obvious choice with reasonable performance and better generalization compared the other models.
we should keep in mind that the thresholds have noticable impact on the predictions. 
so we will keep threshold 0.7 to reduce recall which is optimal for the fraud detection system's concern of lesser recall.
```