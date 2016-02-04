
function [w_int, b_int] = train_interaction_detector_coarse(conf,fra_db,f_train_pos,params,featureExtractor)
params.cand_mode = 'boxes';
cur_set = f_train_pos;
roi_pos_patches = {};
roi_neg_patches = {};
nodes = params.nodes;
for it = 1:length(cur_set)
    it
    %     profile on
    t = cur_set(it);
    imgData = fra_db(t);
    I = getImage(conf,imgData);            
    gt_graph = get_gt_graph(imgData,nodes,params,I);
    faceBox = imgData.faceBox;
    h = faceBox(4)-faceBox(2);    
    rois = sampleAround(gt_graph{1},inf,h,params,I,false);
    rois = arrayfun3(@(x) x.bbox, rois,1);
    g = poly2mask2(cellfun2(@(x) x.poly,gt_graph),size2(I));
    roiMasks = box2Region(rois,size2(I));
    [~,ints,uns] = regionsOverlap(roiMasks,g);
    ints = ints./cellfun(@nnz,roiMasks)';
    roiPatches = multiCrop2(I,round(rois));
    roi_pos_patches{end+1} = roiPatches(ints > .2);
    roi_neg_patches{end+1} = roiPatches(ints < .05);
    %     profile viewer
end
% z = getLogPolarMask(50,10,2);
%%
roi_pos_feats = featureExtractor.extractFeaturesMulti(cat(2,roi_pos_patches{:}));
roi_neg_feats = featureExtractor.extractFeaturesMulti(cat(2,roi_neg_patches{:}));
%%
[w_int, b_int] = concat_and_learn(gather(roi_pos_feats),gather(roi_neg_feats));