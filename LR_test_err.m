function Err = LR_test_err(W,X,Y )

    nTest = size(X,1);
    res = zeros(nTest,1);
    E = zeros(nTest,1);
    for (i = 1:nTest)
        sigm = sigmoid(dot([1, X(i,:)],W));
        if (sigm >= 0.5)
            res(i) = 1;
        else
            res(i) = 0;
        end
        E(i)= res(i)~= Y(i);
    end
Err = mean(E);
end