# Module 12 Report 

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any other algorithms).

The analysis is to build model that can identify the creditworthiness of borrowers. Using a dataset of historical lending activity from a peer-to-peer lending services company. The dataset including each loan size, interest rate, borrower's income, debt-to-income	ratio, number of accounts, derogatory mark, total debts and loan status with label 0 healthy loan and 1 high-risk loan. The model is to predict the loan status with other remaining variables. Before building the model, the dataset is split into training and testing datasets by using `train_test_split`. Then, build a logistic regression model and fit the model by using the training data. Next, save the predictions on the testing data labels by using the testing feature data and the fitted model. Finally, evaluate the model’s performance by generating confusion matrix and classification reports.


## Results

Using bulleted lists, describe the accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
    * Description of Model 1 Accuracy, Precision, and Recall scores.
                  precision    recall  f1-score   support

           0       1.00      1.00      1.00     18759
           1       0.87      0.95      0.91       625

    accuracy                           0.99     19384
    macro avg       0.94      0.97      0.95     19384
    weighted avg       0.99     0.99    0.99     19384

    Precision:
    For Class 0 (Healthy Loan): Precision is 1.00, meaning that all loans predicted as healthy were indeed healthy.
    For Class 1 (Risky Loan): Precision is 0.87, indicating that 87% of the loans predicted as risky were actually risky. This means there were some false positives (loans incorrectly classified as risky).
    
    Recall:
    For Class 0 (Healthy Loan): Recall is 1.00, meaning that all healthy loans were correctly identified.
    For Class 1 (Risky Loan): Recall is 0.95, indicating that 95% of the actual risky loans were correctly predicted by the model. This means there were some false negatives (risky loans that were incorrectly classified as healthy).

    Accuracy: the accuracy of the model is 0.99, meaning that 99% of the total predictions (both classes) were correct.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

Overall, the model predicts both healthy loan and risky loan with 99% accuracy. The macro and weighted averages are over 90% correction. The model performs exceptionally well in predicting healthy loans (class 0) with perfect precision and recall. However, for high-risk loans it may misclassify loans with less risk with high risk. This is need some improvement. As the dataset used in the model including limited financial information, and creditworthiness of borrowers is also affected by over factors like family income, asset value, work experience, ages etc, we should consider those factors to improve the model performance. With all other features included, it is better to use decision tree like random forest algorithm. It can map out non-linear relationships and reduces the risk of overfitting by averaging the results of multiple decision trees, which helps to generalize better on unseen data. Also it provides insights into feature importance, allowing you to identify which variables have the most impact on the predictions. It can efficiently handle large datasets and a high number of features without significant performance degradation. The algorithm is less sensitive to outliers compared to other algorithms, as it uses multiple trees to make predictions. Therefore, if we consider enlarge our study dataset, I recommend to use random forest model.

