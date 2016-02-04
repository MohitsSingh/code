function boxes = gt_boxes(globalOpts,set_,cls)

all_boxes = {};
[ids,t] = textread(sprintf(globalOpts.VOCopts.imgsetpath,[cls '_' set_]),'%s %d');
ids = ids(t==1);

for iID = 1:length(ids)
    
    currentID = ids{iID};
    rec = PASreadrecord(getRecFile(globalOpts,currentID));
    clsinds=strmatch(cls,{rec.objects(:).class},'exact');
    isDifficult = false(size([rec.objects(clsinds).difficult]));
    isTruncated = false(size([rec.objects(clsinds).truncated]));%this effectively
    % toggles usage of truncated examples. 'false' means they are used.
    % get bounding boxes
    c = cat(1,rec.objects(clsinds(~isDifficult & ~isTruncated)).bbox);
    % normalize by image size...
    c(:,[1 3]) = c(:,[1 3])/rec.imgsize(1);
    c(:,[2 4]) = c(:,[2 4])/rec.imgsize(2);
    
    all_boxes{iID} = c;
    %cat(1,rec.objects(clsinds(~isDifficult & ~isTruncated)).bbox);
end

boxes = cat(1,all_boxes{:});

