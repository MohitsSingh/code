function detBB = main_verify(filename,categId,images,base_dir,detBB)
% function detBB = main_verify(filename,categId,images,base_dir,detBB)
%
% Main function for verification classifier
% Input:  filename:     filename of the test image
%         categId:      category label (1-5)
%         images:       exemplar image data (computed in training phase)
%         base_dir:     base directory of exemplar images
%         detBB:        initial bounding box hypotheses
% Output: detBB:        refined bounding box hypotheses
%         detBB.rect:   bounding box rects
%         detBB.score:  bounding box scores
%         detBB.vscore: bounding box verification scores
%
% Related functions: combine_regions, assign_rflags, compute_hists, 
% dist_bwo, compute_min_dist
%
% Copyright @ Chunhui Gu April 2009

% add paths
addpath util/

%%% Parameter Setting
frac = 0.8; % fraction of overlap (region flag assignment)
%%% End Parameter Setting

rects = detBB.rect;
nrects = size(rects,1);
if nrects == 0,
    detBB.vscore = detBB.score;
    return;
end;

regions = combine_regions(filename);
% assign region flags based on overlapping criterion
regionflags = assign_rflags(rects,regions,frac);
[hists,rdata] = compute_hists(filename,regions);

count = 0; N = length(images);
app = zeros(nrects,N);
for id = 1:length(images),

    if images(id).categId ~= categId,
        continue;
    end;

    % compute exemplar data
    efname = images(id).filename;
    efilename = [base_dir efname];
    ebboxes = images(id).bboxes;
    %eregions = combine_regions(efilename);
    eregions = images(id).regions;
    eweights = images(id).weights;
    ebeta = images(id).beta;
    [ehists,erdata] = compute_hists(efilename,eregions);
    
    rflags = (eweights>0);
    eregions = eregions(rflags);
    eweights = eweights(rflags);
    ehists = ehists(rflags,:);
    erdata = erdata(rflags,:);
    
    distmat = dist_bwo(ehists,hists,erdata,rdata,'chi-square',true,false);
    
    dd = zeros(nrects,1);
    for rectId = 1:nrects,
        if any(regionflags(rectId,:)),
            rflags = regionflags(rectId,:);
            dist_min = compute_min_dist(distmat(:,rflags),ebboxes,rects(rectId,:),eregions,regions(rflags));
            dd(rectId) = eweights'*min(dist_min,[],2);
        else
            dd(rectId) = Inf;
        end;
    end;
    
    if ~isempty(ebeta),
        count = count + 1;
        app(:,count) = 1 ./ (1 + exp( -[dd ones(size(dd))]*ebeta ));
    end;
    
    fprintf('done verification from exemplar %d in %g seconds.\n',id,toc);
end;

detBB.vscore = mean(app(:,1:count),2);