function L = likelihood_WLR(Weight, Beta, train_X, train_Y)
temp=0.0;
[nSamples, nFeature] = size(train_X);
for(i = 1:nSamples)
    
            hx = sigmoid(dot([1,train_X(i,:)] , Beta));
        if train_Y(i) == 1
            temp = temp + Weight(i)*log(hx);
        else
            temp = temp + Weight(i)*log(1 - hx);
        end
end
L = temp-0.001 * dot(Beta, Beta);
end
