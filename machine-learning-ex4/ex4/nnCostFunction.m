function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
        
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%printf('\nAt the start of nncostfunc\n');
X = [ones(m,1) X];
a = sigmoid(X*Theta1');
n = size(a,1);
a = [ones(n,1) a];
h = sigmoid(a*Theta2');
init = 1;

Y = zeros(m, num_labels); 
for j = 1:m,
	Y(j,y(j,1)) = 1;
end;       	

for t = 1:m,
	for k = 1:num_labels, 
		J = J + (1/m)*((-1)*log(h(t,k))*Y(t,k) - (1-Y(t,k))*log(1-h(t,k)));
	end;	
end;


n = size(Theta1,2);
theta1 = Theta1(:,2:n);

for j = 1:size(Theta1,1),
	for t = 1:n-1,
		J = J + (lambda/(2*m)).*theta1(j,t)^2;
	end;
end;


n = size(Theta2,2);
theta2 = Theta2(:,2:n);

for j = 1:size(Theta2,1),
	for t = 1:n-1,
		J = J + (lambda/(2*m)).*theta2(j,t)^2;
	end;
end;

#{
l = (-Y).*(log(h))- (1-Y).*(log(1-h));
J = (1/m).*sum(sum(l)) + (lambda/(2*m)).*(sum(sum((theta1.^2))) + sum(sum((theta2.^2))));
#}

% -------------------------------------------------------------
% Theta1 is 5x4, Theta2 is 3x6
% =========================================================================
%#{
del3 = zeros(1,num_labels);
del2 = zeros(1,hidden_layer_size);
bigDel2 = zeros(size(Theta1));
bigDel3 = zeros(size(Theta2));
for t = 1:m,
	a = X(t, :); % 1x4
	a1 = sigmoid(a*Theta1'); %1x5
	a1 = [1 a1];  % 1x6
%	fprintf('\na1\n');	
	a2 = sigmoid(a1*Theta2'); % 1x3
	del3 = a2 - Y(t,:); % 1x3 
	del2 = (del3*Theta2).*[1 sigmoidGradient(a*Theta1')]; % 1x3 3x6 1x6
	del2 = del2(2:end);% 1x5	
	bigDel2 = bigDel2 + del2'*(a);% 5x4
	bigDel3 = bigDel3 + del3'*(a1); % 3x6
%	fprintf('\nAt the end of t = %d\n',t);
end;
Theta1 = Theta1(:,2:end);
zer = zeros(size(Theta1,1),1);
Theta1 = [zer Theta1];
Theta2 = Theta2(:,2:end);
zer = zeros(size(Theta2,1),1);
Theta2 = [zer Theta2];
Theta1_grad = (1/m).*bigDel2 + (lambda/m).*Theta1 ;
Theta2_grad = (1/m).*bigDel3 + (lambda/m).*Theta2;

%#}
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
