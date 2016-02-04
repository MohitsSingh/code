function res = flipAll(images,dim)
res = {};
if nargin < 2
    dim = 2;
end
    
for k = 1:length(images)
    res{k} = flipdim(images{k},dim);
end
end