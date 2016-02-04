function m = multiCrop2(I,rects)

if ~iscell(I)
    I = {I};
end
    
m = {};
% multiple images, multiple rects
if length(I) == size(rects,1)
    II = I;
    for t = 1:size(rects,1)
        I = II{t};
        s = size(I,3);
        m{t} = my_arrayCrop(I,[rects(t,[2 1]) 1],[rects(t,[4 3]) s]);
    end
    return;
end
s = size(I,3);
% one image, multiple rects
if (length(I) == 1 && size(rects,1)>1)
    I = I{1};
    s = size(I,3);
    for t = 1:size(rects,1)
        m{t} = my_arrayCrop(I,[rects(t,[2 1]) 1],[rects(t,[4 3]) s]);
    end
    return;
end

% multiple images, one rect

for t = 1:length(I)
    m{t} = my_arrayCrop(I{t},[rects([2 1]) 1],[rects([4 3]) s]);
end
