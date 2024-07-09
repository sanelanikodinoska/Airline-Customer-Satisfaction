# Airline-Customer-Satisfaction

The code is created as a foundation of understanding and comparing data science/machine learning solutions made in Python notebooks locally and on Azure cloud, as a part of Course DP-100 - Designing and Implementing a Data Science Solution on Azure.

The data for this project is given by an airline organization. The actual name of the company is not given.
The main purpose of this dataset is to predict whether a future customer would be satisfied with their service given the details of the other parameters values.
The dataset consists of the details of customers who have already flown with them. The feedback of the customers on various context and their flight data has been consolidated.
This data shows whether a customer is satisfied with the airlines or not after travelling with them. There are several other measurement or to say feedback taken from the customers as well as their demographic data is also recorded.
Also the airlines need to know on which aspect of the services offered by them have to be emphasized more to generate more satisfied customers.
In order to select the best classifier, several models are built upon supervised machine learning algorithms and their performace is displayed. Since satisfaction (or label 1) is preferred, accuracy, but recall as well, are calculated for thier comparision.

On the other half, a resource group is created on Azure portal using free subscription, created Azure Machine Learning (studio). Automated ML found the best estimator/classifier, the model was registered and deployed to a real-time endpoint. Designer was used to build several pipelines for data processing, training, scoring and evaluating the models – best model is registered and deployed using blue/green deployment. All of this was possible with previously created (from UI) data assets, compute targets and environments. Finally, with code-implementation approach - Python SDK, a pipeline through notebook, scripts and .yml files was created to fine tune hyperparameters with sweep job for a model. All of the gained metrics were compared using mlflow methods.
