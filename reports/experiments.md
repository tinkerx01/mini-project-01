After 5 fold cv
-
| Model | Mean Precision | Mean Recall | Mean F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.8752 | 0.6421 | 0.7396 |
| KNN | 0.9269 | 0.7589 | 0.834 |
| Decision Tree | 0.8666 | 0.7639 | 0.8112 |

-

| Model | Scaling | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| KNN | Without Scaling | 0 | 0 | 0 |
| KNN | With Scaling | 0.9186 | 0.8061 | 0.8587 |

-
- Why is KNN sensitive to scaling?
### because it calculates distance between a and b. so the range of a and b can greatly matter in the product.
- Why is Decision Tree less sensitive?
### decision tree does not care about distance. the question decision tree asks is whether the data has a certain condition or not. if a's value in feature-1 is 500 and b's is 234 it doesn't affect a being 1 in feature-2 and b being 0.78. they are checked seperatly.
