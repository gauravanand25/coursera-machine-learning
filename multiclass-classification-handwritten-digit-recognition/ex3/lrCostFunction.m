function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = size(y,1); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
%	theta						(n+1) X 1
%	X							m X (n+1)
thetatranspose=theta';								% 1 X (n+1)
mthetatranspose=repmat(thetatranspose,m,1);			% m X (n+1)
thetatransposeX=mthetatranspose.*X;					% m X (n+1)
thetatransposeX=sum(thetatransposeX.').';  			% m X 1		row wise sum
HthetaX=sigmoid(thetatransposeX);					% m X 1
logHthetaX=log(HthetaX);							% m X 1
log1minusthetaX=log(1-HthetaX);						% m X 1
J=(sum(y.*logHthetaX) + sum((1-y).*log1minusthetaX));
J=(-J)/m;
J=J+(lambda/(2*m))*(sum(theta.*theta)-theta(1)^2);


HthetaXminusy=HthetaX-y;
grad=((HthetaXminusy')*X)';
grad=grad/m;
grad=grad+(lambda/m).*theta;
grad(1)=grad(1)-(lambda/m)*theta(1);


% =============================================================

grad = grad(:);

end
