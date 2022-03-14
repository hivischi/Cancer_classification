# Cancer_classification
This is a classification work about medical data based on decision tree and support vector machine (data from kaggle.com) using Google colab for Python language. 
Data are based on 569 observations whered the target variable is "diagnosis" with two possible outputs: Malignant (M) or Benign (B).
These outputs have been then changed in a binary encoding (1 for "M" and 0 for "B") in order to compute the correlation between this variable and all the predictors.

Missing data have ben checked: variable "Unnamed: 32" had only missing data, so it has been completely removed. Furthermore, the variable "id" has been removed because it is considered useless.
A correlation matrix has been computed in order to make feature selection:
- variables related to perimeter and area have been removed because they are highly correlated with the variable "radius";
- variables that end with "worst" have been removed because they are highly correlated with the variables that end with "mean";
- variables related to "concave points" and "compactness" have ben removed because they are highly correlated with "concavity". 
Subsequently, the variables have been standardized because they have different units of measure. Feature selection has been then carried out through feature importance score 
computation.
Then data have been splitted in training set (70%) vs. test set (30%). In order to identify the best hyper-parameter configuration for the decision tree, gridsearch 
cross-validation has been carried out. The obtained tree had 8 terminal nodes, which have low impurity values. In addition, the model has been evaluated both on training set and 
test set, through F-1 score because the classes are not balanced. The model has high scores on both sets, therefore it has good classification performances and does not show 
overfitting problems.

Even for the support vector machine (SVM) a grid search has been used to identify the best hyper-parameter configuration (in this case for the parameter C and the kernel function).
Classification results were good both on training set and test set. In order to see the obtained results, a dimensionalty reduction through Principal Component Analysis (PCA) 
has been applied. Two principal components are selected expalining about 50% of the total inertia. 
