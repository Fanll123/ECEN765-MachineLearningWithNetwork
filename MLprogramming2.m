%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Implement the perceptron algorithm for logistic regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Import the datasets
% Dataset files are copied to default working directory
TrainingData = importdata('bclass/bclass-train');
TestingData = importdata('bclass/bclass-test');

% Perceptron with raw data
train_X_raw = TrainingData(:,2:end);
train_Y_raw = TrainingData(:,1);
test_X_raw = TestingData(:,2:end);
test_Y_raw = TestingData(:,1);

W_pcpt_reg_raw = Reg_perceptron(train_X_raw,train_Y_raw); 

% Perceptron with L1 normalized data
for(i = 1:length(train_X_raw))
    train_X_l1nrm(i,:) = train_X_raw(i,:)/norm(train_X_raw(i,:),1);
end
for(i = 1:length(test_X_raw))
    test_X_l1nrm(i,:) = test_X_raw(i,:)/norm(test_X_raw(i,:),1);
end

W_pcpt_reg_l1nrm = Reg_perceptron(train_X_l1nrm,train_Y_raw);

% Perceptron with L2 normalized data
for(i = 1:length(train_X_raw))
    train_X_l2nrm(i,:) = train_X_raw(i,:)/norm(train_X_raw(i,:),2);
end
for(i = 1:length(test_X_raw))
    test_X_l2nrm(i,:) = test_X_raw(i,:)/norm(test_X_raw(i,:),2);
end

W_pcpt_reg_l2nrm = Reg_perceptron(train_X_l2nrm,train_Y_raw); 

% Plot the testing error of each iteration for training data and test data 
%%% training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for(i = 1:length(W_pcpt_reg_raw))
err_reg_raw_train(i)=perceptron_test_err(W_pcpt_reg_raw(i,:),train_X_raw,train_Y_raw);
end
for(i = 1:length(W_pcpt_reg_l1nrm))
err_reg_l1nrm_train(i)=perceptron_test_err(W_pcpt_reg_l1nrm(i,:),train_X_l1nrm,train_Y_raw);
end
for(i = 1:length(W_pcpt_reg_l2nrm))
err_reg_l2nrm_train(i)=perceptron_test_err(W_pcpt_reg_l2nrm(i,:),train_X_l2nrm,train_Y_raw);
end
plot([1:length(W_pcpt_reg_raw)],err_reg_raw_train)
hold on;
plot([1:length(W_pcpt_reg_l1nrm)],err_reg_l1nrm_train)
hold on;
plot([1:length(W_pcpt_reg_l2nrm)],err_reg_l2nrm_train)
xlabel('Iterations')
ylabel('Error rate')
title('Perceptron training error per iteration')
legend('Raw data','L1 normalized','L2 normalized')
hold off;


%%% testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for(i = 1:length(W_pcpt_reg_raw))
err_reg_raw_test(i)=perceptron_test_err(W_pcpt_reg_raw(i,:),test_X_raw,test_Y_raw);
end
for(i = 1:length(W_pcpt_reg_l1nrm))
err_reg_l1nrm_test(i)=perceptron_test_err(W_pcpt_reg_l1nrm(i,:),test_X_l1nrm,test_Y_raw);
end
for(i = 1:length(W_pcpt_reg_l2nrm))
err_reg_l2nrm_test(i)=perceptron_test_err(W_pcpt_reg_l2nrm(i,:),test_X_l2nrm,test_Y_raw);
end
plot([1:length(W_pcpt_reg_raw)],err_reg_raw_test);
hold on;
plot([1:length(W_pcpt_reg_l1nrm)],err_reg_l1nrm_test);
hold on;
plot([1:length(W_pcpt_reg_l2nrm)],err_reg_l2nrm_test);
xlabel('Iterations')
ylabel('Error rate')
title('Perceptron testing error per iteration')
legend('Raw data','L1 normalized','L2 normalized')
hold off;

% Repeat the task by using average perceptron
W_pcpt_avg_raw = Avg_perceptron(train_X_raw,train_Y_raw); 
W_pcpt_avg_l1nrm = Avg_perceptron(train_X_l1nrm,train_Y_raw);
W_pcpt_avg_l2nrm = Avg_perceptron(train_X_l2nrm,train_Y_raw);
% training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for(i = 1:length(W_pcpt_avg_raw))
err_avg_raw_train(i)=perceptron_test_err(W_pcpt_avg_raw(i,:),train_X_raw,train_Y_raw);
end
for(i = 1:length(W_pcpt_avg_l1nrm))
err_avg_l1nrm_train(i)=perceptron_test_err(W_pcpt_avg_l1nrm(i,:),train_X_l1nrm,train_Y_raw);
end
for(i = 1:length(W_pcpt_avg_l2nrm))
err_avg_l2nrm_train(i)=perceptron_test_err(W_pcpt_avg_l2nrm(i,:),train_X_l2nrm,train_Y_raw);
end
plot([1:length(W_pcpt_avg_raw)],err_avg_raw_train)
hold on;
plot([1:length(W_pcpt_avg_l1nrm)],err_avg_l1nrm_train)
hold on;
plot([1:length(W_pcpt_avg_l2nrm)],err_avg_l2nrm_train)
xlabel('Iterations')
ylabel('Error rate')
title('Averaged perceptron training error per iteration')
legend('Raw data','L1 normalized','L2 normalized')
hold off;

% testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for(i = 1:length(W_pcpt_avg_raw))
err_avg_raw_test(i)=perceptron_test_err(W_pcpt_avg_raw(i,:),test_X_raw,test_Y_raw);
end
for(i = 1:length(W_pcpt_avg_l1nrm))
err_avg_l1nrm_test(i)=perceptron_test_err(W_pcpt_avg_l1nrm(i,:),test_X_l1nrm,test_Y_raw);
end
for(i = 1:length(W_pcpt_avg_l2nrm))
err_avg_l2nrm_test(i)=perceptron_test_err(W_pcpt_avg_l2nrm(i,:),test_X_l2nrm,test_Y_raw);
end
plot([1:length(W_pcpt_avg_raw)],err_avg_raw_test);
hold on;
plot([1:length(W_pcpt_avg_l1nrm)],err_avg_l1nrm_test);
hold on;
plot([1:length(W_pcpt_avg_l2nrm)],err_avg_l2nrm_test);
xlabel('Iterations')
ylabel('Error rate')
title('Averaged perceptron testing error per iteration')
legend('Raw data','L1 normalized','L2 normalized')
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Apply gradient ascend logistic regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_Y_01 = train_Y_raw;
train_Y_01(train_Y_01 < 0)=0;
test_Y_01 = test_Y_raw;
test_Y_01(test_Y_01 <0)=0;

% Using raw data
W_LRGA_raw = LR_GradAsc(train_X_raw, train_Y_01, 0.003);
% Using l1 normalized data
W_LRGA_l1nrm = LR_GradAsc(train_X_l1nrm, train_Y_01, 0.003);
% Using l2 normalized data
W_LRGA_l2nrm = LR_GradAsc(train_X_l2nrm, train_Y_01, 0.003);
% Training error of each iteration
for (i = 1:length(W_LRGA_raw))
    err_LRGA_raw_train(i)=LR_test_err(W_LRGA_raw(i,:),train_X_raw,train_Y_01);
end
for (i = 1:length(W_LRGA_l1nrm))
    err_LRGA_l1nrm_train(i)=LR_test_err(W_LRGA_l1nrm(i,:),train_X_l1nrm,train_Y_01);
end
for (i = 1:length(W_LRGA_l2nrm))
    err_LRGA_l2nrm_train(i)=LR_test_err(W_LRGA_l2nrm(i,:),train_X_l2nrm,train_Y_01);
end
plot([1:length(W_LRGA_raw)],err_LRGA_raw_train);
hold on;
plot([1:length(W_LRGA_l1nrm)],err_LRGA_l1nrm_train);
hold on;
plot([1:length(W_LRGA_l2nrm)],err_LRGA_l2nrm_train);
xlabel('Iterations')
ylabel('Error rate')
title('Gradient decent logistic regression training error per iteration')
legend('Raw data','L1 normalized','L2 normalized')
hold off;

% Testing error of each iteration
for (i = 1:length(W_LRGA_raw))
    err_LRGA_raw_test(i)=LR_test_err(W_LRGA_raw(i,:),test_X_raw,test_Y_01);
end
for (i = 1:length(W_LRGA_l1nrm))
    err_LRGA_l1nrm_test(i)=LR_test_err(W_LRGA_l1nrm(i,:),test_X_l1nrm,test_Y_01);
end 
for (i = 1:length(W_LRGA_l2nrm))
    err_LRGA_l2nrm_test(i)=LR_test_err(W_LRGA_l2nrm(i,:),test_X_l2nrm,test_Y_01);
end
plot([1:length(W_LRGA_raw)],err_LRGA_raw_test);
hold on;
plot([1:length(W_LRGA_l1nrm)],err_LRGA_l1nrm_test);
hold on;
plot([1:length(W_LRGA_l2nrm)],err_LRGA_l2nrm_test);
xlabel('Iterations')
ylabel('Error rate')
title('Gradient decent logistic regression testing error per iteration')
legend('Raw data','L1 normalized','L2 normalized')
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. Implement Local Weighted 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 tau = [0.01, 0.05, 0.1, 0.5, 1.0, 5.00];
 Err_WLR = zeros(1,6);
for (l = 1:6)
for (i = 1:length(test_X_raw))
    for(j = 1:length(train_X_raw))
        W(i,j,l)= exp(-norm(test_X_raw(i,:)-train_X_raw(j,:),2)^2/(2*tau(l)^2));
    end
    Beta_t(i,:,l)=LR_LocalWt(W(i,:,l), test_X_raw(i), train_X_raw, train_Y_01, 0.5);
    Err_WLR(l)= Err_WLR(l)+ LR_test_err(Beta_t(i,:,l), test_X_raw(i,:), test_Y_01(i,:));
end            
end
plot(tau,Err_WLR/length(test_X_raw),'o-')
hold on;
xlabel('tau')
ylabel('Error rate')
title('Error rate and bandwidth')
hold off;
