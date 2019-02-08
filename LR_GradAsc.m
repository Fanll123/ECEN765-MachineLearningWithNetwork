% Gradient Ascend Algorithm for MLE logistic regression

function W_iterations = LR_GradAsc(train_X, train_Y,learningRate)
[nSamples, nFeature] = size(train_X);
W_iterations=[];
Wt= zeros(1,nFeature+1);
obj_last = -Inf;
t = 1;
while (t)
    temp = zeros(1,nFeature + 1);
    for (i = 1:nSamples)
        temp = temp + (train_Y(i)-sigmoid(dot([1,train_X(i,:)],Wt)))...
            * [1,train_X(i,:)];
    end
    Wt = Wt + learningRate * temp;
    obj_value = likelihood_LR(Wt, train_X, train_Y);
    
    if (abs((obj_value-obj_last)/obj_value) <= 0.0001)
        break;
    end
    W_iterations=[W_iterations;Wt];
    obj_last = obj_value;
end
end
