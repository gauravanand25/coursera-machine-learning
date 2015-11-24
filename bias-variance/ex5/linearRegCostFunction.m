function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = size(y,1); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%	theta		(n+1) X 1 		
%	X			m x (n+1)
%
J = sum(( ( sum( ( (repmat(theta',m,1)).*X )' ) )'-y ).^2)/(2*m);
J += sum(theta(2:end).^2)*lambda/(2*m);		% regularization term

grad = sum( repmat( ( sum( ( (repmat(theta',m,1)).*X )' ) )'-y,1,size(theta,1) ).*X)'; 
grad += [zeros(1,size(theta,2)); lambda.*theta(2:end,:)];
grad /= m;
% =========================================================================
grad = grad(:);

end
