% Preceptron with averaged weights implementation
% Input: training data X; training data labels Y
% Output: Weight parameters at each iteration W_iterations
% The first column of W is the bias term b
function W_iterations = Avg_perceptron(X,Y)
W_iterations=[];
Wt= zeros(1,length(X(1,:))+1);
W_update = zeros(1,length(X(1,:))+1);
N = length(X);
update_flag = 1;
t = 1;

while(update_flag)
    %temp = Wt;
    update_flag = 0;
    for (i = 1: N)
        if(dot(Wt,[1,X(i,:)])*Y(i) <= 0)
            update_flag = 1;
            temp_update = Y(i)*[1,X(i,:)];
            W_update = ((t-1)*W_update +Y(i)*[1,X(i,:)])/t;
            Wt = Wt + W_update;
            t = t+1;
            W_iterations = [W_iterations;Wt];
        end
    end
    
end
end
