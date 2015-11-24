function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
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

HthetaXminusy=HthetaX-y;
grad=((HthetaXminusy')*X)';
grad=grad/m;
% =============================================================

end
