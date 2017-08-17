function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%thetas = theta(2:end);
thetas = theta(2:end,:);   % theta: (size(X,2),size(y,1))
J = (1/(2*m)).*(X*theta - y)'*(X*theta - y) + (lambda/(2*m)).*(sum(thetas.^2));
h = X*theta;
%theta = theta - (1/m).*(X)'*(X*theta - y);
%X = X(:,2:end);
%thetas = theta(2:end);

%grad = (1/m).*(X'*(h - y)) + (lambda/m).*(thetas);

X1 = X(:,1);
grad1 = (1/m).*(X1'*(h - y));
X2 = X(:,2:end);
grad2 = (1/m).*(X2'*(h - y)) + (lambda/m).*(thetas);

%grad = (1/m).*(X'*(h - y)) + [0 (lambda/m).*(thetas)]

grad = [grad1 ; grad2];

% =========================================================================

grad = grad(:);

end
