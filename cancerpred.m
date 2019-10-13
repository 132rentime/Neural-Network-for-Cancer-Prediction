data = csvread('Cancerdata.csv');

X = data([2:end],[1:end-1]);
Y = data([2:end],end);

%As Y contains no random sorting within it, this can be a deterent to predictive ability
%hence we randomize the Y data. 

dat = load('randorder.mat');
randorder = dat.A;
X = X(randorder,:);
Y = Y(randorder,:);

%We clasify having cancer as 1 and without cancer as 0
%However the dataset clasifies having cancer as 2 and without cancer as 1
%This code does the required change
idxwi = find(Y == 2);
idxwo = find(Y == 1);
Y(idxwi) = 1;
Y(idxwo) = 0;

%Carrying out visualizing of data and Principal Component Analysis on the data
PcA(X,Y,idxwi,idxwo)

Xmu = mean(X);
Xmu = repmat(Xmu,size(X,1),1);
%Mean normalization
Xadjust = X - Xmu;
sd = std(X);
Xadjust = Xadjust./sd;

%_____ split is used to separate the Training
%set, Validation set and the test set
Xtrain = Xadjust([1:80],:);
Ytrain = Y([1:80],:);

Xval = Xadjust([81:100],:);
Yval = Y([81:100],:);


Xtest = Xadjust([101:end],:);
Ytest = Y([101:end],:);

input_layer = 9;
hidden_layer = 3;
output_layer = 1;

A = load('thetavector.mat');
initial_theta = A.thetavector;

lambda = 0.1;
%creating a function handle for the cost function
costfunc = @(t)nncost(t,input_layer,hidden_layer,output_layer,Xtrain,Ytrain,lambda);

options = optimset('MaxIter',150);

[nn_weights, cost] = fmincg(costfunc,initial_theta,options);
%reshaping the neural network weights to produce Theta1 and Theta2
Theta1 = reshape(nn_weights([1:(hidden_layer * (input_layer + 1))],:),hidden_layer,(input_layer + 1));
Theta2 = reshape(nn_weights([(((hidden_layer)*(input_layer + 1)) + 1):end],:),output_layer, (hidden_layer + 1));
%testing the Theta1 and Theta2 on Xtrain and Ytrain
[pred] = results(Xtrain,Ytrain,Theta1,Theta2);
%Training set accuracy
printf('\nTraining Set Accuracy: %f\n', mean(double(pred == Ytrain)) * 100);
%testing the Theta1 and Theta2 on Xtest and Ytest
[pred] = results(Xtest,Ytest,Theta1,Theta2);
%Testing set accuracy
printf('\nTesting Set Accuracy: %f\n', mean(double(pred == Ytest)) * 100);












