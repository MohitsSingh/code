function vocab = train_vocabulary(model,image_ids)
%TRAIN_VOCABULARY train a vocabulary from pre-calculated descriptors.
%   Detailed explanation goes here

if (exit('vocab.mat'))
    load('vocab.mat');
    return;
end
descrs = {} ;
for ii = 1:length(image_ids)
    currentID = image_ids{ii};
    [F,dd] = vl_phow(imread(currentID),'Verbose',false, 'Sizes', 7, 'Step', 5);
    descrs{ii} = dd;
end

descrs = vl_colsubset(cat(2, descrs{:}), 10e4); % 10000 for 4096 descs

% Quantize the descriptors to get the visual words
vocab = vl_kmeans(descrs, model.,...
    'verbose','verbose','verbose', 'algorithm', 'elkan');
save('vocab.mat', 'vocab') ;

end
