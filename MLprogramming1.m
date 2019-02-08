% 1.  Display training data as binary images and convert all datas to
% vectors

% Training and testing data are copied to default working directory
trainFiles = dir('digits/trainingDigits/*.txt');
trainSize = size(trainFiles);
trainSize = trainSize(1);
trainData = zeros(trainSize, 1024);
for (i = 1: trainSize)
    trainLabel(i) = trainFiles(i).name(1);
    FID0 = fopen(fullfile('digits','trainingDigits',trainFiles(i).name),'r');
    A = fscanf(FID0,'%s');
    for (j = 1: 1024)
    trainData(i,j) = str2num(A(j));
    end
end
trainLabel = transpose(trainLabel);

testFiles = dir('digits/testDigits/*.txt');
testSize = size(testFiles);
testSize = testSize(1);
testData = zeros(testSize, 1024);
for (i = 1: testSize)
    FID0 = fopen(fullfile('digits','testDigits',testFiles(i).name),'r');
    A = fscanf(FID0,'%s');
    for (j = 1: 1024)
    testData(i,j) = str2num(A(j));
    end
    testLabel(i) = testFiles(i).name(1);
end
testLabel = transpose(testLabel);

% Visualize the training datas
% take the No. 100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900 sample
imgmtx = transpose(reshape(trainData(1900,:),32,32));
imshow(imgmtx)

% 2. Implement Naive Bayes classifier
% The distribution is bernouli distribution, so we use multinomial model
Mdl_NB = fitcnb(trainData, trainLabel,'Distribution','mn');
trainNB = predict(Mdl_NB, trainData);
testNB = predict(Mdl_NB, testData);
err_NB_train = 1 - sum(trainNB == trainLabel)/trainSize;
err_NB_test = 1 - sum(testNB == testLabel)/testSize;


% Model conditional probabilities
l_train=zeros(1,10);
for i=1:10
    l_train(i)=length(find((trainLabel)==i-1));
end
Mat={}
for k=1:10
    if k==1
    Mat{1}=trainData(1:sum(l_train(1:k)),:);
    else
    Mat{k}=trainData(sum(l_train(1:k-1))+1:sum(l_train(1:k)),:);
    end
end

Non=ones(10,size(Xtrain,2));
Noff=ones(10,size(Xtrain,2));
P0n=ones(10,size(Xtrain,2));

for i=1:10
    p=i-1;
    trMat=cell2mat(Mat(i));
    Non(i,:)=sum(trMat==1,1);
    Noff(i,:)=sum(trMat==0,1);
    a=1; b=1;
    p0n(i,:)=(Non(i,:)+a)./(Non(i,:)+Noff(i,:)+a+b);
end
figure,
imagesc(Cond_prob);
title('the posterior mean');
Cond_prob = Mdl_NB.DistributionParameters


% 3. Implement K-nearest neighborhood
for (k = 1:10)
    Mdl_KNN = fitcknn(trainData,trainLabel,'NumNeighbors',k);
    trainKNN = predict(Mdl_KNN,trainData);
    testKNN = predict(Mdl_KNN,testData);
    testKNN_all(:,k) = testKNN;
    err_KNN_test(k) = 1 - sum(testKNN == testLabel)/testSize;
    err_KNN_train(k) = 1 - sum(trainKNN == trainLabel)/trainSize;
end


% Use majority voting method to do model averaging and plot the model averaging error
for (i = 1:10)
    for(j =1: testSize)
    testKNN_avg(j,i) = mode(testKNN_all(j,1:i));
    end
    err_KNN_avg(i) = 1 - sum(testKNN_avg(:,i) == testLabel)/testSize;
end
% Plot the training error, test error and model average test error with respect of k
K = linspace(1,10,10);
plot(K, err_KNN_test,'o-', 'linewidth', 3)
hold on;
plot(K, err_KNN_train,'o-', 'linewidth', 3)
hold on;
plot(K, err_KNN_avg,'o-', 'linewidth', 3)
xlabel('k')
ylabel('error rate')
title('K nearest neighborhood training/test/model average error rate')
legend('test error','training error','model average test error')
hold off;

% 4. Implement PCA on both datasets 
alldata = [trainData;testData];
[coeff,score,latent,tsquared,explained] = pca(alldata);

% Find approporiate components
x = 0;
while (sum(explained(1:x)) < 97)
    x = x+1;
end
% The first 442 components could explain more than 97% of the variance
% So we take 440 components
% The projected data sets are:
trainData_pca = trainData * coeff(:,1:440);
testData_pca = testData * coeff(:,1:440);

% After PCA, the distribution is a Gaussian model
Mdl_NB_pca = fitcnb(trainData_pca, trainLabel,'Distribution','normal');
testNB_pca = predict(Mdl_NB_pca, testData_pca);
err_NB_pca = 1 - sum(testNB_pca == testLabel)/testSize;
trainNB_pca = predict(Mdl_NB_pca, trainData_pca);
err_NB_train_pca = 1 - sum(trainNB_pca == trainLabel)/trainSize;


for (k = 1:10)
    Mdl_KNN_pca = fitcknn(trainData_pca,trainLabel,'NumNeighbors',k);
    trainKNN_pca = predict(Mdl_KNN_pca,trainData_pca);
    testKNN_pca = predict(Mdl_KNN_pca,testData_pca);
    testKNN_all_pca(:,k) = testKNN_pca;
    err_KNN_test_pca(k) = 1 - sum(testKNN_pca == testLabel)/testSize;
    err_KNN_train_pca(k) = 1 - sum(trainKNN_pca == trainLabel)/trainSize;
end
for (i = 1:10)
    for(j =1: testSize)
    testKNN_avg_pca(j,i) = mode(testKNN_all_pca(j,1:i));
    end
    err_KNN_avg_pca(i) = 1 - sum(testKNN_avg_pca(:,i) == testLabel)/testSize;
end
% Plot the training error, test error with respect of k
K = linspace(1,10,10);
plot(K, err_KNN_test,'o-', 'linewidth', 3)
hold on;
plot(K, err_KNN_test_pca,'o-', 'linewidth', 3)
xlabel('k')
ylabel('error rate')
title('KNN testing error rate')
legend('No-PCA','PCA')
hold off;

plot(K, err_KNN_train,'o-', 'linewidth', 3)
hold on;
plot(K, err_KNN_train_pca,'o-', 'linewidth', 3)
xlabel('k')
ylabel('error rate')
title('KNN training error rate')
legend('No-PCA','PCA')
hold off;

plot(K, err_KNN_avg,'o-', 'linewidth', 3)
hold on;
plot(K, err_KNN_avg_pca,'o-', 'linewidth', 3)
xlabel('k')
ylabel('error rate')
title('KNN model averge testing error rate')
legend('No-PCA','PCA')
hold off;

