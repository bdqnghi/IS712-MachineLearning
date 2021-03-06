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
file_path = strcat("../data/", arg_list{1});
%file_path = strcat("../data/", "winequality-white.csv");

data = importdata(file_path);

%Identify the model name
model_name = arg_list{2};
%model_name ="iterative";

%Split the data into training, validation and test data sets, X is feature, Y is label
%Some data may need normalization on the features
[X_train, Y_train, X_val, Y_val, X_test, Y_test] = data_split(data);


[X_train] = feature_norm(X_train);
[X_val] = feature_norm(X_val);
[X_test] = feature_norm(X_test);

m_train = length(Y_train);
m_val = length(Y_val);
m_test = length(Y_test);

X_train = [ones(m_train, 1) X_train];
X_val = [ones(m_val, 1) X_val];
X_test = [ones(m_test, 1) X_test];

%Weight initialization
feature_size = size(X_train, 2);
weights = randn(feature_size, 1) * 0.5;

alpha = 0.01; %which is the learning rate
iterations = 1000;
total_cost = zeros(iterations, 1);

if strcmp(model_name, "analytical")
    %Training
    [loss_train, weights] = analytical_train(X_train, Y_train, weights);
    %Evauate on validation data set
    loss_val   = analytical_test(X_val, Y_val, weights);
    %Evaluate on testing data set
    loss_test  = analytical_test(X_test, Y_test, weights);

elseif strcmp(model_name, "iterative")
   
    for i = 1:1:iterations
    	%printf("weights :%f\n", i, weights);
        %Training
        [loss_train, weights] = linearR_train(X_train, Y_train, weights, alpha);
        total_cost(i) = loss_train;

        if i <= 10
            printf("Iteration %d loss: %f\n", i, loss_train);
        end
        %printf("Iteration %d loss: %f\n", i, loss_train);

        %Evaluate on validation data set
        loss_val = linearR_predict(X_val, Y_val, weights);
    end
        %Evaluate on testing data set
    loss_test = linearR_predict(X_test, Y_test, weights);
else
    printf("Training model should be provided !!!!! \n")
    return
end

figure;
plot(1:numel(total_cost), total_cost, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Total cost');

%printf("weights for training data: %f\n", weights);
printf("Final loss for training data: %f\n", loss_train);
printf("Final loss for validation data: %f\n", loss_val);
printf("Final loss for test data: %f\n", loss_test);
