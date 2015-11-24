function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
%optimal parameters C = 1.000000 sigma = 0.100000
#{
TempC = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
TempSigma = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
CSigma = size(length(TempC),length(TempSigma));
for i=1:length(TempC)
	for j=1:length(TempSigma)
		printf("Computing for C = %f sigma = %f \n",TempC(i),TempSigma(j));
		model= svmTrain(X, y, TempC(i), @(x1, x2) gaussianKernel(x1, x2, TempSigma(j) )); 
		PredictY = svmPredict(model, Xval);	
		CSigma(i,j) = mean(double(PredictY ~= yval)); 
	end
end

minm = size(yval,1);
for i=1:length(TempC)
	for j=1:length(TempSigma)
		if( minm > CSigma(i,j) )
			minm = CSigma(i,j);
			C = TempC(i);
			sigma = TempSigma(j);
		endif
	end
end
printf("optimal parameters C = %f sigma = %f\n",C,sigma);
#}
C = 1;
sigma = 0.1;
% =========================================================================
end
