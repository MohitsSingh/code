function all_overlaps = get_boxes_recall(globalOpts,set_,cls)

[ids,t] = textread(sprintf(globalOpts.VOCopts.imgsetpath,[cls '_' set_]),'%s %d');
ids = ids(t==1);

all_overlaps={};

cc = 1;
for iID = 1:1:length(ids)    
    iID
    currentID = ids{iID};
    rec = PASreadrecord(getRecFile(globalOpts,currentID));
    clsinds=strmatch(cls,{rec.objects(:).class},'exact');
%         
%     % don't ignore difficult / truncated. 
    isDifficult = false(size([rec.objects(clsinds).difficult]));
    isTruncated = false(size([rec.objects(clsinds).truncated]));                    
    c = cat(1,rec.objects(clsinds(~isDifficult & ~isTruncated)).bbox);
%     
%     c = cat(1,rec.objects.bbox);

    bf = getBoxesFile(globalOpts,ids{iID});    
    b = load(bf);
    boxes = b.boxes(:,[2 1 4 3]);
    
    all_overlaps{cc} = max(boxesOverlap(boxes,c))';
    cc = cc+1;
            
    %cat(1,rec.objects(clsinds(~isDifficult & ~isTruncated)).bbox);
end

all_overlaps = cat(1,all_overlaps{:});
% boxes = cat(1,all_boxes{:});

