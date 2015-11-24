function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

	% theta matrix 			(n+1) X 1
	% theta transpose matrix 	1 X (n+1)
	% X matrix 			m X (n+1)
	thetatranspose=theta.';	      				% 1 X (n+1)
	mthetatranspose=repmat(thetatranspose,m,1);  % all m rows are same
	HthetaX=mthetatranspose.*X;    				% m X (n+1)
	HthetaX=sum(HthetaX.').';       				% m X 1		row wise sum
	HthetaXminusy=HthetaX-y;         				% m X 1	[ h(xi)-yi ]
	temp=(alpha/m).*((HthetaXminusy.')*X);			% 1 X (n+1)
	theta=theta-(temp.');
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
