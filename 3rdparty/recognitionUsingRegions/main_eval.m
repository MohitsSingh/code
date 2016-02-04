function [det, fppi, prec] = main_eval(det_bboxes, test_bound, test_class)
% function [det, fppi, prec] = main_eval(det_bboxes, test_bound, test_class)
%
% Main function for evaluating detection performance.
%
% Related functions: compute_det_fppi
%
% Copyright @ Chunhui Gu, April 2009

clr = {'r--','g--','b--','c--','k--'};
categs = {'AppleLogos','Bottles','Giraffes','Mugs','Swans'};
ncategs = length(categs);

scores = det_bboxes.score;
% for ii = 1:size(scores,1),
%     for jj = 1:size(scores,2),
%         if ~isempty(scores{ii,jj}),
%             scores{ii,jj} = scores{ii,jj} .* det_bboxes.vscore{ii,jj};
%         end;
%     end;
% end;

N = 200;
max_score = getmaxscore(scores);
thres_all = zeros(ncategs,N);
for catId = 1:ncategs,
    %thres_all(catId,:) = 0:max_score(catId)/(N-1):max_score(catId);
    logmax = log10(max_score(catId));
    thres_all(catId,:) = 10.^(-4:(logmax+4)/(N-1):logmax);
end;

det = zeros(ncategs,length(thres_all));
fppi = zeros(ncategs,length(thres_all));
prec = zeros(ncategs,length(thres_all));

for ii = 1:N,
    for catId = 1:ncategs,
        thres = thres_all(catId,ii);
        [det(catId,ii),fppi(catId,ii),prec(catId,ii)] = ...
            compute_det_fppi(det_bboxes.rect(catId,:),scores(catId,:),thres,test_bound,test_class==catId,0.5);
    end;    
    fprintf('done thres = %.2f.\n',thres);
end;

if nargout == 0,
    
    % plot detection-FPPI curve
    figure; clf; hold on;
    for catId = 1:ncategs,
        plot(fppi(catId,:),det(catId,:),clr{catId},'LineWidth',2);
    end;
    hold off;
    xlim([0 1]);
    xlabel('FPPI');
    ylabel('Detection Rate');
    title('Detection Accuracy with Bounding Boxes');
    legend(categs);
    grid on;

    % plot precision-recall curve
    figure; clf; hold on;
    for catId = 1:ncategs,
        plot(det(catId,:),prec(catId,:),clr{catId},'LineWidth',2);
    end;
    hold off;
    xlabel('Recall Rate');
    ylabel('Precision Rate');
    title('Detection Precision-Recall Curve');
    legend(categs);
    grid on;
    
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function maxscore = getmaxscore(scores)

maxscore = zeros(size(scores,1),1);
for ii = 1:size(scores,1),
    for jj = 1:size(scores,2),
        if ~isempty(scores{ii,jj}),
            maxscore(ii) = max(maxscore(ii), max(scores{ii,jj}(:)));
        end;
    end;
end;