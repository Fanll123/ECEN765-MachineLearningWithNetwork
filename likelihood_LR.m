function L = likelihood_LR(Wt, train_X, train_Y)
temp=0.0;
[nSamples, nFeature] = size(train_X);
for(i = 1:nSamples)
    
            hx = sigmoid(dot([1,train_X(i,:)] , Wt));
        if train_Y(i) == 1
            temp = temp + log(hx);
        else
            temp = temp + log(1 - hx);
        end
   % temp = temp +(train_Y(i)*log(sigmoid(dot([1,train_X(i,:)],Wt)))+...
    %    (1-train_Y(i))*log(1-sigmoid(dot([1,train_X(i,:)],Wt))))
end
L = temp;
end
