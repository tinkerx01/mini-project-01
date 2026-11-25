# 9. Cross Validation
-
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

```
- Why is KNN sensitive to scaling?

because it calculates distance between a and b. so the range of a and b can greatly matter in the product.

- Why is Decision Tree less sensitive?

decision tree does not care about distance. the question decision tree asks is whether the data has a certain condition or not. if a's value in feature-1 is 500 and b's is 234 it doesn't affect a being 1 in feature-2 and b being 0.78. they are checked seperatly.
```

# 11. Mandatory Experiment 2: Hyperparameter Analysis
- for knn
| Parameter Value | Precision | Recall | F1 |
|---|---|---:|---:|---:|
|1|0.8454|0.8367|0.841|
|5|0.9186|0.8061|0.8587|
|20|0.8409|0.7551|0.7957|