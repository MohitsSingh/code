function [ A ] = normalize( A )
%NORMALIZE Summary of this function goes here
%   Detailed explanation goes here
%     A = bsxfun(@minus,A,mean(A,2));
    A = bsxfun(@rdivide,A,sum(abs(A),2));
    A = A.^.5;
end

