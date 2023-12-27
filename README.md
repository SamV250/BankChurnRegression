# BankChurnRegression
This project aimed to do a logistic regression to predict bank churn
for the ABC Multistate bank. Bank churn is a term to define if the 
client has left the bank during some period (1) or not (0).

The project did a basic logistic regression, and here is a quick
analysis of the results:

Classification Report:

Precision: Precision measures the accuracy of the positive predictions. For class 1 (churn), the precision is 0.55, indicating that about 55% of the predicted churn cases are correct.

Recall (Sensitivity): Recall measures the ability of the model to capture all the positive instances. For class 1, the recall is 0.20, meaning that the model is capturing only 20% of the actual churn cases.

F1-Score: The F1-score is the harmonic mean of precision and recall. For class 1, the F1-score is 0.29.

Accuracy: The overall accuracy of the model is 0.81, indicating that the model correctly predicts the target variable in 81% of cases.

AUC-ROC Score: The AUC-ROC score is 0.7789. This score provides a measure of how well the model distinguishes between classes. A higher AUC-ROC score (closer to 1) indicates better performance.

Here is the dataset: https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset

Summary:
The model has a decent overall accuracy, but the performance metrics for predicting churn (class 1) are not as high. The recall for class 1 is relatively low, suggesting that the model may not be identifying all the customers who actually churned. Depending on the specific goals and requirements, you might consider further tuning the model or exploring other machine learning algorithms to improve performance.
