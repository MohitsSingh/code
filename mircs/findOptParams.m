function curParams = findOptParams(conf,newImageData,params,sel_train)

% grid search over parameters...
[img_h,winSize,nIters,min_scale,useSaliency,nn] = ...
    ndgrid(params.img_h,params.wSize,params.nIter,params.min_scale,...
    params.useSaliency,params.nn);
curParams = [];
f_train = find(sel_train);
%f_val = f_train(1:2:end);
f_val = vl_colsubset(f_train,round(.3*length(f_train)),'Uniform');
f_train = setdiff(f_train,f_val);

bestPerf = 0;
iBest = 0;

trainImageData = newImageData(f_train);
valImageData = newImageData(f_val);
for iParam = 1:numel(img_h)
    curParams(iParam).img_h = img_h(iParam);
    curParams(iParam).wSize = [winSize(iParam) winSize(iParam)];
    curParams(iParam).nIter = nIters(iParam);
    curParams(iParam).min_scale = min_scale(iParam);
    curParams(iParam).scaleToPerson = true;%TODO
    curParams(iParam).nn = nn(iParam);
    curParams(iParam).useSaliency = useSaliency(iParam);
    % perpare data...
    [XX,offsets,all_scales,imgInds,subInds,values,kdtree,all_boxes,imgs] = preparePredictionData(conf,trainImageData,curParams(iParam));
    % test
%     profile on;    
    
    res = zeros(length(f_val),2);
    for iVal = 1:length(f_val)
        iVal/length(f_val)        
        pMap = ...
            predictBoxes(conf,valImageData(iVal),XX,curParams(iParam),offsets,all_scales,...
            imgInds,subInds,values,imgs,all_boxes,kdtree,false);
        %measure overlap of output probability map with action object
        roi_action = computeHeatMap(pMap,valImageData(iVal).obj_bbox) > 0;
        pMap = normalise(pMap);
        res(iVal,1) = sum(pMap(roi_action));
        res(iVal,2) = sum(pMap(:))-res(iVal,1);
    end
%         profile viewer;
    curParams(iParam).res = res;
end
