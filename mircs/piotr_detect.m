function [all_bbs,all_ids] = piotr_detect(conf,detectors,ids)
all_bbs = {};
all_ids = {};
for k = 1:length(ids)
    k
    I = getImage(conf,ids{k});
    all_bbs{k} = acfDetect(I,detector);
    all_ids{k} = ones(size(all_bbs{k},1),1)*k;
end
end
