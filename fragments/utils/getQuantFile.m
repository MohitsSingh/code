function [ quantPath ] = getQuantFile( globalOpts,currentID )
%GETQUANTFILE Summary of this function goes here
%   Detailed explanation goes here
if isempty(globalOpts.vocab_name)
    quantPath = sprintf(globalOpts.featPath,[currentID '_quant']);
else
    quantPath = sprintf(globalOpts.featPath,[currentID '_' globalOpts.vocab_name '_quant']);
end



