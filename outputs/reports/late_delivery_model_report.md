# Late Delivery Prediction Model Report

## Model Objective

This model predicts whether a synthetic defense production work order is likely to finish later than its planned completion date.

## Model Type

Random Forest Classifier with preprocessing for categorical and numeric production features.

## Training Summary

- Training rows: 262
- Testing rows: 88
- Baseline late-delivery rate: 50.00%
- Accuracy: 54.55%

## Confusion Matrix

|  | Predicted On Time | Predicted Late |
|---|---:|---:|
| Actual On Time | 25 | 19 |
| Actual Late | 21 | 23 |

## Model Notes

The model is trained entirely on synthetic data and is intended for portfolio demonstration. It is not designed for real production, quality, engineering, safety, or defense decision-making.
