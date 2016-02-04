function imdb_sub = restrict_to_sub(imdb,wanted_class,other_class)
imdb_sub = imdb;
imdb_sub.images.labels(~ismember(imdb.images.labels,wanted_class)) = other_class;
end
