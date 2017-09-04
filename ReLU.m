function [ Y, Y_drev ] = ReLU( A )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    Y = (A>0) .* A;
    Y_drev = A>0;
end

