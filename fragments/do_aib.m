function [parents] = do_aib(trainingInstances,globalOpts)
%DO_AIB Summary of this function goes here
%   Detailed explanation goes here
    nPos = size(trainingInstances.posFeatureVecs,2);
    nNeg = size(trainingInstances.negFeatureVecs,2);

    p_pos = sum(trainingInstances.posFeatureVecs,2)';    
    p_neg = sum(trainingInstances.negFeatureVecs,2)';
    % weight the negative weights to be with the same mass as positive\
    
    p_neg = p_neg*nPos/nNeg;
    
	p = [p_pos;p_neg];
    
    p = p(:,1:globalOpts.numWords);
    
    p = p/sum(p(:));
    p = double(p);
    
    [parents] = vl_aib(p);
end

