# Experiment 1 results and error analysis

The fixed, time-blocked split contains 54,900 full-training images, 10,005 subset
training images, 9,225 validation images, and 8,875 test images. The same test
set is used in each row. No hyperparameters were tuned.

| Model | Training images | Validation accuracy | Test accuracy | Test macro F1 | Fit time (s) | Mean inference (ms/image) |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 10,005 | 62.31% | 59.48% | 59.20% | 2.0 | 1.91 |
| Random forest | 10,005 | 57.79% | 60.14% | 59.74% | 9.6 | 29.95 |
| Linear SVM | 10,005 | 56.66% | 57.01% | 56.79% | 42.7 | 3.81 |
| Logistic regression | 54,900 | 63.09% | 61.51% | 61.15% | 20.3 | 2.60 |
| Random forest | 54,900 | 59.59% | 65.15% | 65.01% | 176.2 | 32.45 |
| Linear SVM | 54,900 | 59.12% | 60.43% | 59.87% | 195.2 | 1.13 |

Full training increased test accuracy by 2.03, 5.01 and 3.41 percentage
points for logistic regression, random forest and SVM respectively. The full
logistic model has the highest validation accuracy; the full random forest has
the highest test accuracy in this comparison. Model selection should use the
validation set, not whichever model happens to score highest on the test set.
These are single runs, not uncertainty estimates.

Latency is a CPU batch-of-one mean over 200 test images; it includes feature
extraction and prediction but not image loading or resizing. The forest used
four workers. Timings depend on machine load and are not webcam latencies.
See [comparison.csv](comparison.csv) and [comparison.png](comparison.png).

## What gets confused

For the full random forest, the most frequent directed test errors were:

| True | Predicted | Images |
|---|---|---:|
| D | O | 159 |
| M | A | 81 |
| G | H | 75 |
| U | V | 66 |
| V | U | 65 |

These are counts of frames, not independent signers: consecutive test frames
can represent very similar poses. See [all confusion pairs](random_forest_full_confusions.csv),
the [normalized matrix](random_forest_full_confusion.png), and
[prediction paths](random_forest_full_predictions.csv).

Inspection of representative errors suggests hypotheses, not established causes:

- `D/D2580.jpg` has a prominent circular finger shape as well as an extended
  finger; that shared circle may make D resemble O.
- In `M/M2328.jpg`, the dimly lit, compact fist and occluded fingers make
  individual finger positions hard to distinguish from other fist shapes. All
  81 M-to-A errors occurred in the dim blocks.
- `G/G1608.jpg` shows a sideways hand; small differences in extended-finger
  placement may be difficult to retain at 64x64. Similarly, the distinction
  between U and V in `U/U1136.jpg` and `V/V1292.jpg` depends on finger spacing.
  The coarse HOG cells and global color histograms may lose such details.

Lighting is associated with errors: the full random forest's error rate was
56.1% in dim frames (2,975 images), compared with 25.2% in normal and 23.1%
in bright frames (2,950 each). Lighting and recording blocks are intertwined,
so this does not prove lighting caused the difference. Only 11 test frames had
the blur flag, too few for a useful conclusion. Future work should inspect
additional examples and test lighting augmentation and finer features on the
validation set, then evaluate on photos from new signers and rooms.
