function [thetavector] = initializeTheta(input_layer,hidden_layer,output_layer)
  
  Theta1 = zeros(hidden_layer,input_layer + 1);
  Theta2 = zeros(output_layer,hidden_layer + 1);
  
  epsilon_init1 = sqrt(6)./(sqrt(input_layer + hidden_layer));
  epsilon_init2 = sqrt(6)./(sqrt(hidden_layer + output_layer));
  
  Theta1 = rand(size(Theta1,1),size(Theta1,2)) * 2 * epsilon_init1 - epsilon_init1;
  Theta2 = rand(size(Theta2,1),size(Theta2,2)) * 2 * epsilon_init2 - epsilon_init2;
  
  thetavector = [Theta1(:) ; Theta2(:)];
endfunction
