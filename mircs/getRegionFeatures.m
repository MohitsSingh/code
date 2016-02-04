function [feats,regions,regionOvp] = getRegionFeatures(conf,imageID)
conf.get_full_image = true;
[fullImage] = getImage(conf,imageID);
load('~/code/kmeans_4000.mat','codebook');
model.numSpatialX = [2];
model.numSpatialY = [2];
model.vocab = codebook;
model.w = [];
model.b = [];

bowFile = fullfile(conf.bowDir,strrep(imageID,'.jpg','.mat'));
regionFile = fullfile(conf.gpbDir,strrep(imageID,'.jpg','_regions.mat'));
% fill masks...
load(bowFile);
load(regionFile);
regions = fillRegionGaps(regions);
feats = struct('frames',[],'descrs','binsa');
feats.frames = F;
feats.binsa = bins';
feats.descrs = [];
feats = getBOWFeatures(conf,model,{fullImage},{regions},feats);
end