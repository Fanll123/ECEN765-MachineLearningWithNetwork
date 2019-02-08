%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Support Vector Machines
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Import the datasets
% Dataset files are copied to default working directory
TrainingData = importdata('bclass/bclass-train');
TestingData = importdata('bclass/bclass-test');
train_X = TrainingData(:,2:end);
train_Y = TrainingData(:,1);
test_X = TestingData(:,2:end);
test_Y = TestingData(:,1);
% Linear SVM
C = [0.25, 0.5, 1, 2, 4];
for (i=1:5)
    SVM_lin = fitcsvm(train_X,train_Y,'KernelFunction','linear',...
        'BoxConstraint',C(i));
    numSV_lin(i) = length(SVM_lin.SupportVectors);
    errTrain_lin(i) = mean(train_Y ~= predict(SVM_lin,train_X));
    errTest_lin(i) = mean(test_Y ~= predict(SVM_lin,test_X));   
end

plot(C, errTrain_lin,'-o')
hold on
plot(C, errTest_lin,'-o')
xlabel('C')
ylabel('Error')
title('Training and testing error for linear SVM')
legend('Training error','Testing error')
hold off

% RBF Kernel SVM
tau = [0.25, 0.5, 1, 2, 4];
for (j=1:5)
    for (i = 1:5)
        SVM_RBF = fitcsvm(train_X,train_Y,'KernelFunction','rbf',...
            'BoxConstraint',C(i),'KernelScale', tau(j));
        numSV_RBF(j,i) = length(SVM_RBF.SupportVectors);
        errTrain_RBF(j,i) = mean(train_Y ~= predict(SVM_RBF,train_X));
        errTest_RBF(j,i) = mean(test_Y ~= predict(SVM_RBF,test_X));
    end
    
end
% Plot the training error
for (j = 1:5)
    plot(C, errTrain_RBF(j,:),'-o')
    hold on
end
xlabel('C')
ylabel('Error')
title('Training error for RBF SVM')
legend('Tau = 0.25','Tau = 0.5','Tau = 1','Tau = 2','Tau = 4')
hold off
% Plot the testing error
for (j = 1:5)
    plot(C, errTest_RBF(j,:),'-o')
    hold on
end
xlabel('C')
ylabel('Error')
title('Testing error for RBF SVM')
legend('Tau = 0.25','Tau = 0.5','Tau = 1','Tau = 2','Tau = 4')
hold off

% Plot the support vector numbers
for (j = 1:5)
    plot(C, numSV_RBF(j,:),'-o')
    hold on
end
plot(C, numSV_lin,'-o')
xlabel('C')
ylabel('Suppoer vector numbers')
title('Support vector numbers for linear and RBF SVM')
legend('RBF SVM Tau = 0.25','RBF SVM Tau = 0.5','RBF SVM Tau = 1','RBF SVM Tau = 2','RBF SVM Tau = 4','Linear SVM')
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Gaussian Mixture Models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task 1: using the bclass dataset

CLLT = [];       % trace of complete log likelihoods for best run
C    = [];       % actual clusters
BIC = [];
AIC = [];
for (k = 2:10)
    % try 10 different iterations 
    gmm = fitgmdist(train_X, k,'Replicates',10,'RegularizationValue',0.1);
    C(k-1,:) = cluster(gmm, train_X);
    CLLT(k-1) = -gmm.NegativeLogLikelihood;
    BIC(k-1) = gmm.BIC;
    AIC(k-1) = gmm.AIC;
end
plot([2:1:10],CLLT,'-o')
xlabel('K')
ylabel('Log Likelihood')
title('Log Likelihood of bclass data GMM')

plot([2:1:10],BIC,'-o')
hold on
xlabel('K')
plot([2:1:10],AIC,'-o')
title('BIC and AIC of bclass data GMM')
legend('BIC','AIC')
hold off;


% Using the hw3-data
CLLT = [];       % trace of complete log likelihoods for best run
C    = [];       % actual clusters
BIC = [];
AIC = [];
X = load('hw3-data');
for (k = 2:10)
    % try 10 different iterations 
    gmm = fitgmdist(X, k,'Replicates',10,'RegularizationValue',0.1);
    C(k-1,:) = cluster(gmm, X);
    CLLT(k-1) = -gmm.NegativeLogLikelihood;
    BIC(k-1) = gmm.BIC;
    AIC(k-1) = gmm.AIC;
end
plot([2:1:10],CLLT,'-o')
xlabel('K')
ylabel('Log Likelihood')
title('Log Likelihood of hw3 data GMM')

plot([2:1:10],BIC,'-o')
hold on
xlabel('K')
plot([2:1:10],AIC,'-o')
title('BIC and AIC of hw3 data GMM')
legend('BIC','AIC')
hold off;


