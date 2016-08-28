function [loss, weights, bias] = linearR_train(X, Y, weights, alpha)
	
	m = length(Y); %number of elements
	loss    = (1/(2*m))*sum(power((X*weights - Y),2));
	delta = (1/(2*m))*sum(X.*repmat((X*weights - Y), 1, size(X,2)));
	weights = (weights' - (alpha * delta))';
	
end
