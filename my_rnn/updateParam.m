function [ param,dparam,mparam] = updateParam( param,dparam,mparam,learningRate)
%UPDATEPARAM Summary of this function goes here
%   Detailed explanation goes here
mparam = mparam + dparam.^2;
param = param - learningRate*dparam ./ ((mparam+1e-8).^.5);

end

