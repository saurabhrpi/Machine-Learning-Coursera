function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.01;

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
predictions = 0;
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
predictions = svmPredict(model,Xval);
err = mean(double(predictions ~= yval));
prevError = err;	
C_old = 0.01;
sigma_old = 0.01;
for i = 1:8,
	for j = 1:8,
		model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
		predictions = svmPredict(model,Xval);
		err = mean(double(predictions ~= yval));
		if(err < prevError),
				prevError = err;
				C_old = C;
				sigma_old = sigma;	
		end;	
		sigma = sigma*3;
	end;
	C = C*3;
	sigma = 0.01;
end;

C = C_old;
sigma = sigma_old;



% =========================================================================

end
