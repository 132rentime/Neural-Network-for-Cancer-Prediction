function [] = learningcurve(Xtrain,Ytrain,Xval,Yval,lambda, initial_theta, input_layer, hidden_layer, output_layer)
  m = size(Xtrain,1);
  costlist = zeros(m,3)
  for i = 1:m
    
  xtemp = Xtrain([1:m],:);
  ytemp = Ytrain([1:m],:);
  
  costfunc = @(t)nncost(t,input_layer,hidden_layer,output_layer,x,y,lambda);
  options = optimset('MaxIter',50);
  [nn_weights, cost] = fmincg(costfunc,initial_theta,options);
  
  thet1 = reshape(nn_weights([1:((hidden_layer)*(input_layer + 1))],1),hidden_layer,input_layer + 1);
  thet2 = reshape(nn_weights([(((hidden_layer)*(input_layer + 1)) + 1):end],1),input_layer,hidden_layer + 1);
  
  [trainingcost] = resultscost(xtemp,ytemp,thet1,thet2);
  [validationcost] = resultscost(Xval,Yval,thet1,thet2);
  costlist(i,1) = i;
  costlist(i,2) = trainingcost;
  costlist(i,3) = validationcost;
  
  figure(3) 
  plot(costlist(:,1),costlist(:,2),'b-');
  hold on
  plot(costlist(:,1),costlist(:,3),'r-');
  
  endfor
    
  
endfunction
