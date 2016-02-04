function [ gt_struct] = loadGT(VOCopts,imageset)
%GETGTBOXES read all ground-truth boxes from data for given image set
%(train , val, test)

clear gt_struct;
%gt_struct = struct('imageset',{},'imageID',{},'cls',{},'bbox',{});

% load given image set
ids=textread(sprintf(VOCopts.imgsetpath,imageset),'%s');

detector.bbox={};
detector.gt=[];
tic;
tic_id = ticStatus(sprintf('loading ground truth for %s',imageset),1,0);
n = 0;
for iID=1:length(ids)
    % display progress
    % read annotation
    
    gt_struct(iID) = PASreadrecord(sprintf(VOCopts.annopath,ids{iID}));
    tocStatus(tic_id,iID/length(ids));
    %     for u = 1:length(rec.objects)
    %         n = n+1;
    %         gt_struct(n).imageset = imageset;
    %         gt_struct(n).imageID = ids{iID};
    % %         gtBoxes(n).
    %         gt_struct(n).cls
    %     end
    %     % find objects of class and extract difficult flags for these objects
    %     %clsinds=strmatch(cls,{rec.objects(:).class},'exact');
    %     diff=[rec.objects(clsinds).difficult];
    %     % assign ground truth class to image
    %
    %     detector.bbox{end+1}=cat(1,rec.objects(clsinds(~diff)).bbox)';
    %     tocStatus(tic_id);
end
