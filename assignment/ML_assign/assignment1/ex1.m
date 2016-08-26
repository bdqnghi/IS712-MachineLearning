%
%
%This is just a sample code (uncompleted) and you can implement the assignment from scrach.
%But please only print out the information listed (line 50, 63, 64, 65) in this sample code
%when submitting your code.
%
%
%If you use this sample structure, you need to implement the functions
%"analytical_train", "analytical_test", "linearR_train", "linearR_predict" 
%in the corresponding files by yourself. 
%The function "data_split" is provided as an example. You can rewrite it.
%
%

%Command line parameters
arg_list = argv ();

%Loading data
%file_path = strcat("../data/", arg_list{1});
file_path = strcat("../data/", "winequality-white.csv");

data = importdata(file_path);

%Identify the model name
%model_name = arg_list{2};

%Split the data into training, validation and test data sets, X is feature, Y is label
%Some data may need normalization on the features
[X_train, Y_train, X_val, Y_val, X_test, Y_test] = data_split(data);


%[X_train mu sigma] = feature_normalize(X_train);
% Add intercept term to X
%X_train = [ones(length(X_train), 1) X_train];
%X_train

%Weight initialization
feature_size = size(X_train, 2);
weights = randn(feature_size, 1) * 0.5;

bias = randn(1);

alpha = 0.01;
iterations = 1000;

[weights, loss_train] = gradient_descent_multi(X_train, Y_train, weights, alpha, iterations);

%fprintf('Cost J: \n');
%fprintf(' %f \n', J_history);
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', weights);
fprintf('\n');


	
