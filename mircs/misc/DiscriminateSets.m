function [feats] = DiscriminateSets(X1,X2)
%DISCRIMINATESETS Summary of this function goes here
%   Detailed explanation goes here

    % try to learn a classifier for each of X1...    
    q = X1{1};    
    for iSample = 1:size(q,2);
        pos_sample = q(:,iSample);        
        for jj = 1:5:length(X2)            
            neg_samples = X2{jj};
                [w,b,sv] = train_classifier(pos_sample,neg_samples);
    end
end