function plot_average_detection()
% function plot_average_detection()
%
% This function plots average detection rate over false positive per image
% (FPPI) curve, as well as average precision-recall curve.
%
% Copyright @ Chunhui Gu, April 2009

addpath util/

linespec = {'r--','g--','b--','m--','c--'};
categs = {'AppleLogos','Bottles','Giraffes','Mugs','Swans'};
ncategs = 5;
nruns = 5;

for ii = 1:nruns,
    load(['mat/run' num2str(ii) '/det_bboxes_detection.mat']);
    load(['mat/run' num2str(ii) '/gt_bound_mask.mat']);
    load(['mat/run' num2str(ii) '/filename.mat']);
    for jj = 1:size(det_bboxes.score,1),
        for kk = 1:size(det_bboxes.score,2),
            if ~isempty(det_bboxes.score{jj,kk}),
                det_bboxes.score{jj,kk} = det_bboxes.score{jj,kk} .* det_bboxes.vscore{jj,kk};
            end;
        end;
    end;
    [r(ii).det, r(ii).fppi, r(ii).prec] = main_eval(det_bboxes, test_bound, test_class);
end;

% plot detection-FPPI curve
fppi_std = 0:0.002:2;
figure(1); clf; hold on;
for catId = 1:ncategs,
    for ii = 1:nruns,
        fppi = r(ii).fppi(catId,:);
        det = r(ii).det(catId,:);
        [fppi,I] = unique(fppi);
        det = det(I);
        det_std(ii,:) = interp1(fppi,det,fppi_std,'linear');
    end;
    det_mean = mean(det_std);
    plot(fppi_std,det_mean,linespec{catId},'LineWidth',3);
end;

hold off;
xlim([0,1]);
xlabel('False-positives per image','FontSize',15);
ylabel('Detection rate','FontSize',15);
title('Average detection accuracy','FontSize',15);
legend(categs);
set(gca,'FontSize',13);
grid on;

% plot precision-recall curve
% det_std = 0:0.002:1;
% figure(3); clf; hold on;
% for catId = 1:ncategs,
%     for ii = 1:nruns,
%         prec = r(ii).prec(catId,:);
%         det = r(ii).det(catId,:);
%         [det,I] = unique(det);
%         prec = prec(I);
%         prec_std(ii,:) = interp1(det,prec,det_std,'linear');
%     end;
%     prec_mean = mean(prec_std);
%     plot(det_std,prec_mean,linespec{catId},'LineWidth',3);
% end;
% 
% hold off;
% xlabel('Recall (Detection) rate','FontSize',15);
% ylabel('Precision rate','FontSize',15);
% title('Average precision-recall curve','FontSize',15);
% legend(categs);
% set(gca,'FontSize',13);
% grid on;