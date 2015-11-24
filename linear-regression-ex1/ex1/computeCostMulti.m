function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = size(y,1); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% theta matrix 			(n+1) X 1
% theta transpose matrix 	1 X (n+1)
% X matrix 			m X (n+1)
thetatranspose=theta.';	      				% 1 X (n+1)
mthetatranspose=repmat(thetatranspose,m,1);  % all m rows are same
HthetaX=mthetatranspose.*X;    				% m X (n+1)
HthetaX=sum(HthetaX.').';      				% m X 1		row wise sum
squareThis=HthetaX-y;         				% m X 1
squaredTerm=squareThis.*squareThis;	% squaring it
J=sum(squaredTerm)/(2*m);
% =========================================================================

end
