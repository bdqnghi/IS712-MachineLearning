function [loss, weights] = analytical_train(X, Y, weights)
	
	m = length(Y); % number of training examples

	weights = pinv(X'*X)*X'*Y;
	loss    = (1/(2*m))*sum(power((X*weights - Y),2));
end
