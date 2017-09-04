function [ Y, Y_drev ] = sigmoid( A )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    Y = 1./(1 + exp(-A));
    Y_drev = exp(-A)./((1+exp(-A)).^2);
end

