% Testing for perceptron algorithm
% Input: Perceptron coefficients W; testing data X; testing data labels Y
% Output: Error rate for W
% The first column of W is the bias term b

function Err = perceptron_test_err(W,X,Y)
E = zeros(length(X),1);
for(i = 1:length(X))
    E(i) = (sign(W*([1,X(i,:)])') ~= Y(i));
end
Err = mean(E);
end




