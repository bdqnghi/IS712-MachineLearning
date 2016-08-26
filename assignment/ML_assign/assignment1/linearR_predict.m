function loss = linearR_predict(X, Y, weights)

	m = length(Y); %number of elements
	loss = (1/(2*m))*sum(power((X*weights - Y),2));
	
end
