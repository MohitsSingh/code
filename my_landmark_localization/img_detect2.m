function res = img_detect2(param,w,curImg)
if (length(size(curImg))==2)
    curImg = cat(3,curImg,curImg,curImg);
end

[boxes] = get_boxes_single_scale(param.windowSize, param.cellSize,single(curImg));
z = 31*(param.windowSize/param.cellSize)^2;
U = (fhog2(single(curImg),param.cellSize));
w = w(:,1:z)';

filters = reshape(w,param.windowSize/param.cellSize,param.windowSize/param.cellSize,31,[]);
filters = mat2cell2(single(filters),[1 1 1 size(filters,4)]);
dd = fconv_var_dim(U, filters, 1, length(filters));
dd = cellfun2(@(x) x(:), dd);
dd = cat(2,dd{:});

res = [boxes dd];
