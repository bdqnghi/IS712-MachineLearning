function loss = analytical_test(X, Y, weights)

	m = length(Y); % number of training examples
	loss = (1/(2*m))*sum(power((X*weights - Y),2));
end
