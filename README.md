# Neural-Network-for-Cancer-Prediction
From my previous project, It was observed that the accuracy of the logistic regression model peaks at 86% accuracy. Instead, I plan to build more non linear hypothesis with complex function using a neural network. 
Relevant research paper : https://bmccancer.biomedcentral.com/articles/10.1186/s12885-017-3877-1

Master branch of Neural Network;
The neural netork has 1 output layer, 1 input layer and 1 hidden layer. The input layer consists of 9 input variables(including bias unit there will be 10 variables). The hidden layer consists of 3 activation nodes. The output layer has just one output as the output is binary(Cancer or no Cancer).
Pricipal componenent analysis shows that we need only 2 principal components to capture all the 9 variables with 99% variance or more, hence the PCA images are also plotted and presented in the Master branch. 
The neural network is trained using a Training set which is 60% of the total dataset. It is seen that with a lambda value of 0.1, 
