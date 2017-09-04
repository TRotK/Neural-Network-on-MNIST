function [ Y , y_pred ] = softmax( A )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    %N = size(A,2);
    c = size(A,1);
%     Y = A;
%     y_pred = zeros(1,N);
    exp_A = exp(A);
    sum_A = sum(exp_A,1);
%     for n = 1:N
%         Y(:,n) = exp_A(:,n)/sum_A(n);
%         y_pred(n) = find(Y(:,n)==max(Y(:,n)),1) - 1;
%     end
    Y = exp_A./repmat(sum_A,c,1);
    %[~,y_pred] = max(Y,[],1);
    [~,y_pred] = max(Y);
    y_pred = y_pred - 1;
    
    %Y = exp(A) * (diag(1./sum(exp(A),1)));
end

