Build a Bayesian Image Classifier Using Class-wise Mean Vectors and Full Covariance Matrices, and Analyze Classification Decisions
Project Overview
This project implements a Gaussian Bayesian image classifier from scratch using Python, NumPy, and SciPy. It classifies handwritten digits by modelling each digit class with a class-wise mean vector and covariance matrix.

The main model uses a separate full covariance matrix for each class, corresponding to Quadratic Discriminant Analysis (QDA). The classifier also supports shared and diagonal covariance structures so their performance can be compared.

In addition to predicting digit labels, the project analyzes how classification decisions are made by examining the contributions of class priors, covariance log-determinants, and Mahalanobis distances to the model's log-joint scores.
Objectives
- Build a Gaussian Bayesian classifier using class-wise mean vectors.
- Estimate a full covariance matrix for each digit class.
- Apply ridge regularization to handle singular covariance matrices.
- Compare full, shared, and diagonal covariance structures.
- Select the regularization parameter using stratified cross-validation.
- Evaluate classification performance on a held-out test set.
- Analyze posterior confidence, calibration, and misclassification decisions.
- Validate the custom Gaussian log-joint calculations against SciPy.
Dataset
The project uses the handwritten digits dataset provided by sklearn.datasets.load_digits.
Property	Description
Dataset	scikit-learn handwritten digits
Number of samples	1,797
Image dimensions	8 × 8 pixels
Features per image	64
Classes	10 (digits 0–9)
Pixel scaling	Divided by 16, resulting in values from 0 to 1
Training/test split	70% / 30%
Split method	Stratified
Random state	0


The dataset is loaded directly through scikit-learn, so a separate dataset download is not required.
Methodology
1. Gaussian Bayesian Classification
For each class \(k\), the model estimates:
- Mean vector (\(\mu_k\)): the average 64-dimensional pixel vector for that digit.
- Covariance matrix (\(\Sigma_k\)): describes the variance of pixel features and their relationships.
- Class prior (\(\pi_k\)): estimated from the proportion of training samples belonging to that class.
The class-conditional distribution is modelled as a multivariate Gaussian:
\[
p(x \mid k) =
\frac{1}{(2\pi)^{d/2}|\Sigma_k|^{1/2}}
\exp\left(
-\frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)
\right)
\]
Using Bayes' theorem, the classifier computes the posterior probability of each class. The prediction follows the Maximum A Posteriori (MAP) decision rule:
\[
\hat{k} = \arg\max_k p(k \mid x)
\]
For numerical computation, the implementation works with log probabilities. The class-wise log-joint score is:
\[
\log p(x,k) =
\log \pi_k
-\frac{1}{2}\log|\Sigma_k|
-\frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)
-\frac{d}{2}\log(2\pi)
\]
The common evidence term \(\log p(x)\) is not required when choosing the class with the largest score.
2. Covariance Structures
The GaussianBayes implementation supports three covariance configurations:
Configuration	Description	Classifier interpretation
full	A separate full covariance matrix is estimated for each class.	QDA
shared	A common covariance matrix is formed from the class covariance matrices using class-prior weights.	Shared-covariance Gaussian classifier (LDA-style)
diag	Only the diagonal elements of each class covariance matrix are retained.	Gaussian Naive Bayes-style model


The full covariance configuration is the primary model for this project. The other configurations provide comparison points for understanding the effect of covariance assumptions.
3. Regularization
Covariance matrices can become singular when some features have zero variance or when the available samples do not provide enough information to estimate a stable matrix. In this dataset, some pixel features may be constant in the training set.

The notebook applies ridge regularization:
\[
\Sigma_k^{(\mathrm{reg})} = \Sigma_k + \lambda I
\]
where:
- \(\lambda\) is the regularization parameter.
- \(I\) is the identity matrix.
The notebook first demonstrates the failure of an unregularized covariance fit, then evaluates nine candidate values of \(\lambda\) using five-fold stratified cross-validation:
\[
\lambda \in \{10^{-4}, \ldots, 10^0\}
\]
The best regularization value is selected independently for each covariance configuration based on mean cross-validation accuracy.
4. Numerical Stability
Instead of explicitly calculating the inverse of each covariance matrix, the classifier uses:
- Cholesky decomposition to factorize regularized covariance matrices.
- Triangular-system solving to calculate the Mahalanobis-distance terms.
- Log-determinants calculated from the Cholesky factors.
- logsumexp to normalize log-joint scores into log-posterior probabilities.
These techniques improve numerical stability when working with multivariate Gaussian distributions.
5. Classification Evaluation
After selecting the regularization parameter, the notebook fits each covariance model on the training data and evaluates it on the held-out test data.

The evaluation includes:
- Test accuracy for each covariance configuration.
- Confusion matrix for the full-covariance model.
- Per-class precision, recall, and F1-score.
- Correct and incorrect prediction confidence.
- Expected Calibration Error (ECE).
- Test Negative Log-Likelihood (NLL).
6. Analysis of Classification Decisions
The classifier's log-joint score is decomposed into additive terms:
- Prior term: contribution from the class probability.
- Log-determinant term: contribution from the covariance matrix volume.
- Mahalanobis term: contribution from the distance between the sample and the class mean, accounting for covariance.
- Constant term: Gaussian normalization constant.
For misclassified samples, the notebook compares the predicted class with the true class by measuring the difference in each score contribution. It also checks how often the Mahalanobis-distance term alone would select the true class and identifies cases where the log-determinant term favors the predicted class.

The notebook displays the most confident misclassifications, including the original digit image, predicted label, true label, confidence, and posterior probabilities.
Implementation
The main classifier is implemented in the GaussianBayes class.
Method	Purpose
fit(X, y)	Estimates class means, priors, covariance matrices, Cholesky factors, and log-determinants.
terms(X)	Calculates the individual additive terms of the class-wise log-joint scores.
log_joint(X)	Computes the class-wise log-joint scores.
predict_log_proba(X)	Returns normalized log-posterior probabilities.
predict_proba(X)	Returns posterior probabilities.
predict(X)	Predicts the class with the highest log-joint score.


Supporting functions:
- cv_accuracy(...): computes mean stratified cross-validation accuracy for a given regularization value and covariance type.
- confusion(...): constructs a confusion matrix from true and predicted labels.
Technologies Used
- Python — primary programming language.
- NumPy — numerical computation and matrix operations.
- SciPy — Cholesky decomposition, triangular solving, log-sum-exp, and multivariate normal reference calculations.
- Matplotlib — data and evaluation visualizations.
- scikit-learn — handwritten digits dataset and data-splitting utilities.
- Jupyter Notebook — interactive development and execution.
Repository Structure
.
├── bayes_classifier_pattern_matching (1).ipynb
└── README.md
The notebook contains the complete implementation, dataset preparation, regularization search, model evaluation, visualizations, classification-decision analysis, and SciPy validation.
Installation
Python 3.9 or later is recommended.

Install the required libraries:
pip install numpy scipy matplotlib scikit-learn jupyter
How to Run the Project
1. Clone or download the repository.
2. Open a terminal in the project directory.
3. Launch Jupyter Notebook:
   jupyter notebook
4. Open bayes_classifier_pattern_matching (1).ipynb.
5. Run all cells from top to bottom.
The notebook loads the digits dataset directly from scikit-learn.
Outputs and Visualizations
When the notebook is executed, it produces:
1. Training and test dataset dimensions and the count of constant training pixels.
2. A demonstration of the singular covariance issue without regularization.
3. Five-fold cross-validation scores across regularization values for all three covariance configurations.
4. A plot comparing cross-validation accuracy against the regularization parameter.
5. Test accuracy for the full, shared, and diagonal covariance models.
6. A confusion matrix and per-digit precision, recall, and F1-score for the full-covariance model.
7. Visualizations of class-wise mean digit images and pixel variances.
8. Confidence distributions and a reliability diagram.
9. Expected Calibration Error and test Negative Log-Likelihood.
10. A decomposition of classification decisions for misclassified samples.
11. Visualizations of high-confidence misclassifications and their posterior probabilities.
12. A numerical sanity check against SciPy's multivariate normal log-density implementation, along with a posterior probability normalization check.
Note: Exact accuracy, selected regularization values, calibration metrics, and other numerical results are generated by executing the notebook. They are not hard-coded in this README.

Limitations

- The model assumes that each digit class follows a multivariate Gaussian distribution.
- Full covariance estimation requires more parameters than diagonal covariance modelling and can be sensitive to limited training data.
- Ridge regularization helps address singularity and numerical instability, but the selected value can affect model performance.
- The dataset contains small 8×8 grayscale images; results may not generalize directly to larger or more complex image datasets.
- The notebook is an experimental implementation and does not include a deployed prediction service or graphical user interface.
- 
Future Scope

Possible extensions include:
- Testing the classifier on larger handwritten-character datasets.
- Comparing the approach with additional generative and discriminative classifiers.
- Exploring alternative covariance estimation and regularization techniques.
- Applying dimensionality reduction before covariance estimation.
- Investigating additional probability-calibration methods.
- Building an interface for classifying user-uploaded handwritten digit images.
License
No license is specified in the supplied notebook. Add an appropriate LICENSE file if you plan to distribute the project publicly.
