function detBB = main_voting(filename,categId,thres,images,base_dir)
% function detBB = main_voting(filename,categId,thres,images,base_dir)
%
% Main function for Hough-like voting based on region matching
% Input:  filename:     filename of the test image
%         categId:      category label (1-5)
%         thres:        threshold for voting score
%         images:       exemplar image data (computed in training phase)
%         base_dir:     base directory of exemplar images
% Output: detBB:        bounding box hypotheses
%         detBB.rect:   bounding box rects
%         detBB.score:  bounding box scores
%         detBB.id:     exemplar id where bounding box is transformed
%
% Related functions: combine_regions, compute_hists, dist_bwo,
% gen_houghvoting, cat_det, mean_shift
%
% Copyright @ Chunhui Gu April 2009

% add paths
addpath util/

%%% Parameter Setting
param.th = 0; param.sw = 0.2; param.sh = 0.2; param.ss = 1.2;
isvisual = false;
%%% End Parameter Setting

% compute query image data
regions = combine_regions(filename);
[hists,rdata] = compute_hists(filename,regions);

tic;
detBB = [];
numexemplars = length(images);
for id = 1:numexemplars,
    
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
    [ehists,erdata] = compute_hists(efilename,eregions);
    
    % view exemplar region weights
    if false,
        view_weights(efilename,ebboxes,eregions,eweights);
    end;
    
    rflags = (eweights>0);
    eregions = eregions(rflags);
    eweights = eweights(rflags);
    ehists = ehists(rflags,:);
    erdata = erdata(rflags,:);
    
    % generate hough votes
    distmat = dist_bwo(ehists,hists,erdata,rdata,'chi-square',true,false);
    det = gen_houghvoting(eregions,regions,distmat,eweights,ebboxes,efilename,filename);
    
    % view votes per exemplar
    if false,
        subplot(1,2,1); imshow(efilename);
        subplot(1,2,2); view_votes(filename,det);
        keyboard;
    end;
    
    % collect bounding box votes
    detBB = cat_det(detBB,det,id);
    
    fprintf('done voting from exemplar %d in %g seconds.\n',id,toc);
end;

% clustering bounding boxes in feature space
[detBB.rect,dscores,dprob] = mean_shift(detBB.rect,detBB.score,param);
if ~isempty(dscores),
    detBB.score = dprob'*detBB.score;
end;
detBB = rmfield(detBB,'id');
% remove candidates with scores lower than threshold
detBB.rect = detBB.rect(detBB.score>=thres,:);
detBB.score = detBB.score(detBB.score>=thres,:);

if isvisual,
    view_votes(filename,detBB); keyboard;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% View learned region weights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function view_weights(efilename,ebboxes,eregions,eweights)

subplot(1,2,1); imshow(efilename);
rectangle('Position',[ebboxes(:,1:2),ebboxes(:,3:4)-ebboxes(:,1:2)],'EdgeColor','r','LineWidth',3);
ucm2 = imread([efilename '_ucm2.bmp']);
ucm = ucm2(3:2:end,3:2:end);
for rr = 1:length(eregions),
    if eweights(rr) > 0,
        subplot(1,2,2); imshow(ucm.*uint8(ucm>=40) + 255*uint8(eregions{rr}));
        rectangle('Position',[ebboxes(:,1:2),ebboxes(:,3:4)-ebboxes(:,1:2)],'EdgeColor','r','LineWidth',3);
        title(['weight = ' num2str(eweights(rr),'%.2d')]);
        pause;
    end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% View hough votes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function view_votes(filename,det)

imshow(filename);
for rr = 1:size(det.rect,1),
    rectangle('Position',det.rect(rr,:),'EdgeColor','r');
    text(det.rect(rr,1),det.rect(rr,2),num2str(det.score(rr),'%.2f'),'color','k','backgroundcolor','r',...
        'verticalalignment','top','horizontalalignment','left','fontsize',8);
end;