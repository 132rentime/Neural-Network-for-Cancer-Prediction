function [cost] = resultscost(x,y,Theta1,Theta2)
  m = size(x,1);
  
  %carrying out forward propagation for testset
  z2 = Theta1*[ones(m,1),x]';
  a2 = sigmoid(z2);
  z3 = Theta2*[ones(1,m);a2];
  a3 = sigmoid(z3);
  a3 = a3';
 
 cost = (1./m).*((a3 - y).^2);
  
endfunction
