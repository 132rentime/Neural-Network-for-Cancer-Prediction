function [] = PcA(X,Y,idxwi,idxwo)
 %carrying out singular value decomposition on the variables 
 %to figure out which variables contain most variance and 
 %which principal components are required to represent the data 
 
 Xmu = mean(X);
 Xmu = repmat(Xmu,size(X,1),1);
 %Mean normalization
 Xadjust = X - Xmu;
 
 %Obtaining the covariance matrix
 covX = cov(Xadjust);
 [U,S,V] = svd(covX);
 
 
 tracetotal = sum(sum(S));
 
 for i = 1:size(S,2)
   %calculates the percentage of variance contributed by
   %a set of principal components from 1 to size(S,1)
   tracepercent = sum(sum(S(:,[1:i])))./tracetotal;
   if i>=2 && tracepercent>=0.99
     k = i;
     break
   end
 endfor 
 
 %We found the variables which contribute most to the total variance
 %Now we visualize these variables if it can be shown in 2D
 %Note that there is no reduction in dimensionality using PCA in this code as 
 %of yet
 if isequal(i,2)
 figure(1)
 %without cancer
 plot(X(idxwo,1),X(idxwo,i),'bx');
 hold on
 %with cancer
 plot(X(idxwi,1),X(idxwi,i),'rx');
else 
 %Assuming one principal component is not enough
 print('Data cannot be visualized as Principal component number is above 2')
endif 

%Now we move to principal component analysis
reducedX = Xadjust*U(:,[1:i]);
 if isequal(i,2)
 figure(2)
 %without cancer
 plot(reducedX(idxwo,1),reducedX(idxwo,i),'bx');
 hold on
 %with cancer
 plot(reducedX(idxwi,1),reducedX(idxwi,i),'rx');
 xlabel('Principal Component 1');
 ylabel('Principal Component 2');
else 
 %Assuming one principal component is not enough
 print('Data cannot be visualized as Principal component number is above 2')
endif 

endfunction
