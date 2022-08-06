Current project shows the usage of the linear regression to predict Boston's house pricing based on previous
research data.  
Linear regression has been chosen as the optimal algorythm to predict numeric values due to the analysis.
Research data stored on /datasets/boston_house_pricing.csv file.  
System actions:
 - reads the data
 - splits it onto 2 sets: training and test one (80%/20%)
 - prepares data using ScikitLearn pipelines
 - feeds prepared data to the linear regression algorythm
 - shows predictions/errors
 - saves the model to the /models/model.pkl file