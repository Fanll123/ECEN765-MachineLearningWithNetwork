function Beta_L = LR_LocalWt(Weight, test_X, train_X, train_Y,learningRate)

[nSamples, nFeature] = size(train_X);
Beta_L= zeros(nFeature+1,1);
obj_last = 0.0;
%obj = [];
t = 1;
while (t)
    Grad_L = zeros(nFeature + 1,1);
    Hess_L = zeros(nFeature+1);
    for (i = 1:nSamples)
        Grad_L = Grad_L + Weight(i)*(train_Y(i)-sigmoid(dot([1,train_X(i,:)],Beta_L)))...
            * [1,train_X(i,:)]';
        Hess_L = Hess_L -( Weight(i) * sigmoid(dot([1,train_X(i,:)],Beta_L))...
            *(1-sigmoid(dot([1,train_X(i,:)],Beta_L)))* [1,train_X(i,:)]' * [1,train_X(i,:)]);
    end
    Grad_L = Grad_L-2*Beta_L*0.001;
    Hess_L = Hess_L-2*0.001*eye(nFeature+1);
    learningRate =1;
    Beta_L = Beta_L - inv(Hess_L) * (Grad_L) * learningRate;
    
    obj_value = likelihood_WLR(Weight, Beta_L',train_X, train_Y);

    if (abs(obj_value-obj_last) <= 0.0001*abs(obj_value))
    %if (norm(Grad_L)<=0.001)
        Beta_L = Beta_L';
        break;
    end
    obj_last = obj_value;
end
end
