function [im, labels] = getBatch_simple(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
if length(size(imdb.images.labels)) > 2    
    labels = imdb.images.labels(:,:,:,batch) ;
else
    labels = imdb.images.labels(batch) ;
end
% if rand > 0.5, im=fliplr(im) ; end