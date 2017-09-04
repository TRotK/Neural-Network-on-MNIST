clear all; close all;

%% extracting data
tic;
addpath('hw2data');
addpath('hw2data/training set');
[trainImgs,trainLabels] = readMNIST('train-images-idx3-ubyte','train-labels-idx1-ubyte',60000,0);
addpath('hw2data/test set');
[testImgs,testLabels] = readMNIST('t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte',10000,0);
toc;

%% initialization
eta = 10^-5;
c = 10;
d = size(trainImgs,2); %d = 784
N = size(trainImgs,1); %N = 60000
N_test = size(testImgs,1); %N_test = 10000

X = trainImgs'; %784x60000 
X_test = testImgs'; %784x10000
X0 = ones(1,N); %bias: 1x60000
X0_test = ones(1,N_test); %bias: 1x10000

T = zeros(c,N);
for n = 1:N
    T(trainLabels(n)+1,n) = 1;
end
T_test = zeros(c,N_test);
for n = 1:N_test
    T_test(testLabels(n)+1,n) = 1;
end

%% a) ii)
tic;
W = randn(d,c); %784x10
W0 = randn(1,c); %1x10

numIter = 2000;
E_train = zeros(1,numIter); err_train = zeros(1,numIter);
E_test = zeros(1,numIter); err_test = zeros(1,numIter);
for t = 1:numIter
    t
    % feed forward
    A = [W0;W]'*[X0;X]; %10x60000 = 10x785 * 785x60000
    [Y,pred] = softmax(A);
    A_test = [W0;W]'*[X0_test;X_test];
    [Y_test,pred_test] = softmax(A_test);

    % compute error
    err_train(t) = 1 - sum(pred==trainLabels')/N;
    err_test(t) = 1 - sum(pred_test==testLabels')/N_test;
    
    % compute cost
%     E_train(t) = -sum(sum(T.*log(Y)));
%     E_test(t) = -sum(sum(T_test.*log(Y_test)));
    
    % gradient
    Grad = [X0;X] * (T-Y)';
    %    785x60000  10x60000

    % update weights
    Wu = [W0;W] + eta*Grad;
    W = Wu(2:end,:);
    W0 = Wu(1,:);
end
toc;

finalErr_1 = [err_train(end);err_test(end)];

figure(51);
plot(1:numIter,err_train,'r'); hold on;
plot(1:numIter,err_test,'b'); hold off;
axis([1 numIter 0 1]);
xlabel('#Iteration t'); ylabel('Error');
legend('Training','Test');
title('Training and test error vs. #iteration with single layer neural network');

%% b) ii)
tic;

eta = 1e-5;
HHH = [10,20,50];

Y0 = ones(1,N); %2nd-layer bias: 1x60000  
Y0_test = ones(1,N_test);

finalErr_2 = zeros(2,3);
numIter = 2000;
figure(52);
for p = 1:3
    H = HHH(p);
    % 1st-layer weights
    W_ij = randn(d,H); %784xH
    W0_ij = randn(1,H); %1xH
    % 2nd-layer weights
    W_jk = randn(H,c); %Hxc
    W0_jk = randn(1,c); %1xc

    err_train = zeros(1,numIter);
    err_test = zeros(1,numIter);
    for t=1:numIter
        t
        % feed forward
        G = [W0_ij;W_ij]'*[X0;X]; %Hx60000 = Hx785 * 785x60000
        [Y,Sig_drev] = sigmoid(G);
        U = [W0_jk;W_jk]'*[Y0;Y]; %10x60000 = 10x(H+1) * (H+1)x60000
        [Z,pred] = softmax(U); 

        G_test = [W0_ij;W_ij]'*[X0_test;X_test]; %Hx10000 = (H+1)x785 * 785x10000
        [Y_test,~] = sigmoid(G_test); %Hx10000
        U_test = [W0_jk;W_jk]'*[Y0_test;Y_test]; %10x10000 = 10x(H+1) * (H+1)x10000
        [Z_test,pred_test] = softmax(U_test);

        % compute error
        err_train(t) = 1 - sum(pred==trainLabels')/N;
        err_test(t) = 1 - sum(pred_test==testLabels')/N_test;

        % gradient
        Delta_k = -(T-Z); %10x60000
        Grad_jk = [Y0;Y] * (-Delta_k)'; %(H+1)x10
        %       (H+1)x60000   10x60000
        Grad_ij = -[X0;X] * (Sig_drev.* (W_jk*Delta_k))'; %785xH
        %        785x60000    Hx60000    Hx10 10x60000    

        % update weights
        Wu_jk = [W0_jk;W_jk] + eta*Grad_jk;
        W_jk = Wu_jk(2:end,:);
        W0_jk = Wu_jk(1,:);

        Wu_ij = [W0_ij;W_ij] + eta*Grad_ij;
        W_ij = Wu_ij(2:end,:);
        W0_ij = Wu_ij(1,:);
    end
        
    finalErr_2(:,p) = [err_train(end);err_test(end)];
    
    subplot(1,3,p);
    plot(1:numIter,err_train,'r'); hold on;
    plot(1:numIter,err_test,'b'); hold off;
    axis([1 numIter 0 1]);
    xlabel('#Iteration t'); ylabel('Error');
    legend('Training','Test');
    title(['Training and test error vs. #iteration with two-layer '...
        'neural network (H = ' num2str(H) ')']);    
end
toc;

%% b) iii)
tic;

lbd = 0.001;
eta = 2e-6;
HHH = [10,20,50];

Y0 = ones(1,N); %2nd-layer bias: 1x60000  
Y0_test = ones(1,N_test);

finalErr_3 = zeros(2,3,2);
numIter = 2000;
figure(53);
for act = 1:2
    for p = 1:3
        H = HHH(p);
        % 1st-layer weights
        W_ij = randn(d,H); %784xH
        W0_ij = randn(1,H); %1xH
        % 2nd-layer weights
        W_jk = randn(H,c); %Hxc
        W0_jk = randn(1,c); %1xc

        err_train = zeros(1,numIter);
        err_test = zeros(1,numIter);
        for t=1:numIter
            t
            % feed forward
            G = [W0_ij;W_ij]'*[X0;X]; %Hx60000 = Hx785 * 785x60000
            if act > 1
                [Y,Act_drev] = ReLU(G); %Hx60000 
            else
                [Y,Act_drev] = sigmoid(G); %Hx60000
            end
            U = [W0_jk;W_jk]'*[Y0;Y]; %10x60000 = 10x(H+1) * (H+1)x60000
            [Z,pred] = softmax(U); 

            G_test = [W0_ij;W_ij]'*[X0_test;X_test]; %Hx10000 = (H+1)x785 * 785x10000
            if act > 1
                [Y_test,~] = ReLU(G_test); %Hx10000
            else
                [Y_test,~] = sigmoid(G_test); %Hx10000
            end
            U_test = [W0_jk;W_jk]'*[Y0_test;Y_test]; %10x10000 = 10x(H+1) * (H+1)x10000
            [Z_test,pred_test] = softmax(U_test);

            % compute error
            err_train(t) = 1 - sum(pred==trainLabels')/N;
            err_test(t) = 1 - sum(pred_test==testLabels')/N_test;

            % gradient
            Delta_k = -(T-Z); %10x60000
            Grad_jk = [Y0;Y] * (-Delta_k)' - 2*lbd/eta*[W0_jk;W_jk]; %(H+1)x10
            %       (H+1)x60000   10x60000
            Grad_ij = -[X0;X] * (Act_drev.* (W_jk*Delta_k))' - 2*lbd/eta*[W0_ij;W_ij]; %785xH
            %        785x60000    Hx60000    Hx10 10x60000    

            % update weights
            Wu_jk = [W0_jk;W_jk] + eta*Grad_jk;
            W_jk = Wu_jk(2:end,:);
            W0_jk = Wu_jk(1,:);

            Wu_ij = [W0_ij;W_ij] + eta*Grad_ij;
            W_ij = Wu_ij(2:end,:);
            W0_ij = Wu_ij(1,:);
        end

        finalErr_3(:,p,act) = [err_train(end);err_test(end)];

        if act > 1
            subplot(2,3,p+3);
            plot(1:numIter,err_train,'r'); hold on;
            plot(1:numIter,err_test,'b'); hold off;
            axis([1 numIter 0 1]);
            xlabel('#Iteration t'); ylabel('Error');
            legend('Training','Test');
            title(['Errors vs. #iteration for two-layer '...
            'neural network (H = ' num2str(H) ') with ReLU']); 
        else
            subplot(2,3,p); 
            plot(1:numIter,err_train,'r'); hold on;
            plot(1:numIter,err_test,'b'); hold off;
            axis([1 numIter 0 1]);
            xlabel('#Iteration t'); ylabel('Error');
            legend('Training','Test');
            title(['Error vs. #iteration for two-layer '...
            'neural network (H = ' num2str(H) ') with sigmoid']);
        end   
    end
end    
toc;

%% b) iv)
tic;

lbd = 0.0001;
eta = 2e-6;
HHH = [10,20,50];

Y0 = ones(1,N); %2nd-layer bias: 1x60000  
Y0_test = ones(1,N_test);

finalErr_4 = zeros(2,3,2);
numIter = 2000;
figure(54);
for act = 1:2
    for p = 1:3
        H = HHH(p);
        % 1st-layer weights
        W_ij = randn(d,H); %784xH
        W0_ij = randn(1,H); %1xH
        % 2nd-layer weights
        W_jk = randn(H,c); %Hxc
        W0_jk = randn(1,c); %1xc

        err_train = zeros(1,numIter);
        err_test = zeros(1,numIter);
        for t=1:numIter
            t
            % feed forward
            G = [W0_ij;W_ij]'*[X0;X]; %Hx60000 = Hx785 * 785x60000
            if act > 1
                [Y,Act_drev] = ReLU(G); %Hx60000 
            else
                [Y,Act_drev] = sigmoid(G); %Hx60000
            end
            U = [W0_jk;W_jk]'*[Y0;Y]; %10x60000 = 10x(H+1) * (H+1)x60000
            [Z,pred] = softmax(U); 

            G_test = [W0_ij;W_ij]'*[X0_test;X_test]; %Hx10000 = (H+1)x785 * 785x10000
            if act > 1
                [Y_test,~] = ReLU(G_test); %Hx10000
            else
                [Y_test,~] = sigmoid(G_test); %Hx10000
            end
            U_test = [W0_jk;W_jk]'*[Y0_test;Y_test]; %10x10000 = 10x(H+1) * (H+1)x10000
            [Z_test,pred_test] = softmax(U_test);

            % compute error
            err_train(t) = 1 - sum(pred==trainLabels')/N;
            err_test(t) = 1 - sum(pred_test==testLabels')/N_test;

            % gradient
            Delta_k = -(T-Z); %10x60000
            Grad_jk = [Y0;Y] * (-Delta_k)' - 2*lbd/eta*[W0_jk;W_jk]; %(H+1)x10
            %       (H+1)x60000   10x60000
            Grad_ij = -[X0;X] * (Act_drev.* (W_jk*Delta_k))' - 2*lbd/eta*[W0_ij;W_ij]; %785xH
            %        785x60000    Hx60000    Hx10 10x60000    

            % update weights
            Wu_jk = [W0_jk;W_jk] + eta*Grad_jk;
            W_jk = Wu_jk(2:end,:);
            W0_jk = Wu_jk(1,:);

            Wu_ij = [W0_ij;W_ij] + eta*Grad_ij;
            W_ij = Wu_ij(2:end,:);
            W0_ij = Wu_ij(1,:);
        end

        finalErr_4(:,p,act) = [err_train(end);err_test(end)];

        if act > 1
            subplot(2,3,p+3);
            plot(1:numIter,err_train,'r'); hold on;
            plot(1:numIter,err_test,'b'); hold off;
            axis([1 numIter 0 1]);
            xlabel('#Iteration t'); ylabel('Error');
            legend('Training','Test');
            title(['Errors vs. #iteration for two-layer '...
            'neural network (H = ' num2str(H) ') with ReLU']); 
        else
            subplot(2,3,p); 
            plot(1:numIter,err_train,'r'); hold on;
            plot(1:numIter,err_test,'b'); hold off;
            axis([1 numIter 0 1]);
            xlabel('#Iteration t'); ylabel('Error');
            legend('Training','Test');
            title(['Error vs. #iteration for two-layer '...
            'neural network (H = ' num2str(H) ') with sigmoid']);
        end   
    end
end    
toc;

%% c) Stochastic Gradient Descent
%% Repeat a) ii) SpeedUp
tic;

eta = 1e-2;

W = randn(d,c); %784x10
W0 = randn(1,c); %1x10
Wu = [W0;W]; Xu = [X0;X]; Xu_test = [X0_test;X_test];

s = randperm(N);

numIter = 60000;
err1_train = zeros(1,numIter);
err1_test = zeros(1,numIter);
for t = 1:numIter
    t
    % feed forward
    %A = [W0;W]'*[X0;X]; %10x60000 = 10x785 * 785x60000
    A = Wu'*Xu; 
    [Y,pred] = softmax(A);
    A_test = Wu'*Xu_test; %A_test = [W0;W]'*[X0_test;X_test];
    [Y_test,pred_test] = softmax(A_test);

    % compute error
    err1_train(t) = sum(pred~=trainLabels')/N;
    err1_test(t) = sum(pred_test~=testLabels')/N_test;
    
    % gradient
    %Grad = [X0(:,s(t));X(:,s(t))] * (T(:,s(t))-Y(:,s(t)))';
    %             785x1                     10x1
    Grad = Xu(:,s(t)) * (T(:,s(t))-Y(:,s(t)))';
    
    % update weights
    Wu = Wu + eta*Grad;
%     Wu = [W0;W] + eta*Grad;
%     W = Wu(2:end,:);
%     W0 = Wu(1,:);
end
toc;

finalErr_5 = [err1_train(end);err1_test(end)];

figure(55);
plot(1:numIter,err1_train,'r'); hold on;
plot(1:numIter,err1_test,'b'); hold off;
axis([1 numIter 0 1]);
xlabel('#Iteration t'); ylabel('Error');
legend('Training','Test');
title('Training and test error vs. #iteration using SGD with single layer neural network');

%% Repeat b iii) SpeedUp
tic;

eta = [1e-2,2e-3];
HHH = [10,20,50];

Y0 = ones(1,N); %2nd-layer bias: 1x60000  
Y0_test = ones(1,N_test);
Xu = [X0;X]; 
Xu_test = [X0_test;X_test];

numIter = 60000;
err2_train = zeros(6,numIter);
err2_test = zeros(6,numIter);

s = randperm(N);

for act = 1:2
    for p = 1:3
        H = HHH(p);
        % 1st-layer weights
        W_ij = randn(d,H); %784xH
        W0_ij = randn(1,H); %1xH
        Wu_ij = [W0_ij;W_ij];
        % 2nd-layer weights
        W_jk = randn(H,c); %Hxc
        W0_jk = randn(1,c); %1xc
        Wu_jk = [W0_jk;W_jk];
        
        for t=1:numIter
            t
            % feed forward
            G = Wu_ij'*Xu; %Hx60000 = Hx785 * 785x60000
            G_test = Wu_ij'*Xu_test; %Hx10000 = (H+1)x785 * 785x10000
            if act > 1
                [Y,Act_drev] = ReLU(G); %Hx60000 
                [Y_test,~] = ReLU(G_test); %Hx10000
            else
                [Y,Act_drev] = sigmoid(G); %Hx60000
                [Y_test,~] = sigmoid(G_test); %Hx10000
            end
            Yu = [Y0;Y];
            U = Wu_jk'*Yu; %10x60000 = 10x(H+1) * (H+1)x60000
            [Z,pred] = softmax(U); 

            Yu_test = [Y0_test;Y_test];
            U_test = Wu_jk'*Yu_test; %10x10000 = 10x(H+1) * (H+1)x10000
            [Z_test,pred_test] = softmax(U_test);

            % compute error
            if act > 1
                err2_train(p+3,t) = sum(pred~=trainLabels')/N;
                err2_test(p+3,t) = sum(pred_test~=testLabels')/N_test;
            else
                err2_train(p,t) = sum(pred~=trainLabels')/N;
                err2_test(p,t) = sum(pred_test~=testLabels')/N_test;
            end
            
            % gradient
            delta_k = -(T(:,s(t))-Z(:,s(t))); %10x1
            grad_jk = Yu(:,s(t)) * (-delta_k)'; %(H+1)x10
            %           (H+1)x1     10x1
            grad_ij = -Xu(:,s(t)) * (Act_drev(:,s(t)).* (W_jk*delta_k))'; %785xH
            %            785x1           Hx1             Hx1    10x1    

            % update weights
            Wu_jk = Wu_jk + eta(act)*grad_jk;
            Wu_ij = Wu_ij + eta(act)*grad_ij;
        end    
    end
end
toc;

figure(56);
for sp = 1:3
    subplot(2,3,sp);
    plot(1:numIter,err2_train(sp,:),'r'); hold on;
    plot(1:numIter,err2_test(sp,:),'b'); hold off;
    axis([1 numIter 0 1]);
    xlabel('#Iteration t'); ylabel('Error');
    legend('Training','Test');
    title(['Error vs. #iteration for two-layer '...
    'neural network (H = ' num2str(H) ') using SGD with sigmoid']);

    subplot(2,3,sp+3);
    plot(1:numIter,err2_train(sp+3,:),'r'); hold on;
    plot(1:numIter,err2_test(sp+3,:),'b'); hold off;
    axis([1 numIter 0 1]);
    xlabel('#Iteration t'); ylabel('Error');
    legend('Training','Test');
    title(['Error vs. #iteration for two-layer '...
    'neural network (H = ' num2str(H) ') using SGD with ReLU']);
end

finalErr_6 = [err2_train(:,end)';err2_test(:,end)'];

%% save errors
save('finalErrs.mat','finalErr_1','finalErr_2','finalErr_3','finalErr_4','finalErr_5','finalErr_6');
