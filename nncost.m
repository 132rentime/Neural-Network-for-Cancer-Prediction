function [J,grad] = nncost(theta,input_layer,hidden_layer,output_layer,Xtrain,Ytrain,lambda)
  %extracting Theta1 and Theta2
  Theta1 = reshape(theta([1:((input_layer + 1)*hidden_layer)]),hidden_layer,(input_layer + 1));
  Theta2 = reshape(theta([((input_layer + 1)*hidden_layer + 1):end]),output_layer,(hidden_layer + 1));
  
  m = size(Xtrain,1);
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));
  
  %starting forward propagation
  Xtrain = [ones(size(Xtrain,1),1),Xtrain];
  z2 = Theta1*Xtrain'; %gives a 5 x 69 vector with each column responding to 5 activation units for that example
  a2 = sigmoid(z2);
  a2 = [ones(1,size(a2,2));a2];
  z3 = Theta2*a2;
  a3 = sigmoid(z3);
  
  a3 = a3';
  
  J1 = (1./m)*sum(Ytrain.*log(a3));
  J2 = (1./m)*sum((1 - Ytrain).*log(1 - a3));
  
  
  J3 = (lambda./(2*m))*sum(sum(Theta1(:,[2:end]).^2));
  J4 = (lambda./(2*m))*sum(sum(Theta2(:,[2:end]).^2));
  
  J = -J1-J2+J3+J4;
  
  %end of forward propagation and finding the cost
  
  delta1 = 0;
  delta2 = 0;
  %back propagation is done for each example at one time.
  for t = 1:m
    z_1 = Xtrain(t,:);
    z_2 = Theta1*z_1';
    a_2 = sigmoid(z_2);
    a_2 = [ones(1,size(a_2,2));a_2];
    z_3 = Theta2*a_2;
    a_3 = sigmoid(z_3);
    d_3 = a_3 - Ytrain(t,:);
    d_2 = (Theta2)'*(d_3).*((a_2).*(1 - a_2));
    delta2 = delta2 + (d_3*a_2');
    delta1 = delta1 + (d_2([2:end],:)*z_1);
  endfor
  
  Theta1_grad = (1./m)*delta1;
  Theta2_grad = (1./m)*delta2;
  %regularization of gradient
  Theta1_grad = [Theta1_grad(:,1),Theta1_grad(:,[2:end]) + (lambda./m).*Theta1(:,[2:end])];
  Theta2_grad = [Theta2_grad(:,1),Theta2_grad(:,[2:end]) + (lambda./m).*Theta2(:,[2:end])];
  %unroll gradients
  grad = [Theta1_grad(:);Theta2_grad(:)];
endfunction
