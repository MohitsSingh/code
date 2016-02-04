function [rois,scores,thetas,curFeats] = scoreCoarseRois(conf,imgData,params_coarse,featureExtractor,w_int,b_int)
I = getImage(conf,imgData);
gt_graph = get_gt_graph(imgData,params_coarse.nodes,params_coarse,I);
faceBox = imgData.faceBox;
h = faceBox(4)-faceBox(2);
rois = sampleAround(gt_graph{1},inf,h,params_coarse,I,false);
thetas = [rois.theta];
rois = arrayfun3(@(x) x.bbox, rois,1);
roiPatches = multiCrop2(I,round(rois));
curFeats = featureExtractor.extractFeaturesMulti(roiPatches,true);
%     profile off
scores = w_int'*curFeats+b_int;
