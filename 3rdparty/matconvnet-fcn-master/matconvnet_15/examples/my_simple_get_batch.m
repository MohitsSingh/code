function [im,labels] = my_simple_get_batch(imdb, batch)
% -------------------------------------------------------------------------
im =  cat(4,imdb.images.data{batch});
labels = imdb.images.label(batch) ;
