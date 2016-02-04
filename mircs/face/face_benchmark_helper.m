
ovps = [results.ovp];
nnz([results.zhu_bad])

faces_detected = ovps > .5;
ratioFacesDetected = sum(faces_detected)/length(results)

delta_zhu = zeros(6,length(results));
delta_ours = zeros(6,length(results));
faceLengths = zeros(size(results));
cares = zeros(6,length(results));

goods = false(size(results));
for t = 1:length(results)    
    t;
    if (results(t).zhu_bad),continue,end
    if (~results(t).face_detected ),continue,end
    goods(t) = true;
end

mean_error_zhu = [results(goods).zhu_error]./[results(goods).curFaceSize];
mean_error_ours = [results(goods).our_error]./[results(goods).curFaceSize];
clf
bins = linspace(0,1,200);
f = @(x) hist(x,bins);
cumulative_error = @(x) cumsum(f(x))/sum(f(x));
h_ours = cumulative_error(mean_error_ours);
h_zhu = cumulative_error(mean_error_zhu);
u = find(bins <= .3);

h  = figure(1); plot(bins(u),h_ours(u),'r-','LineWidth',2);
hold on; plot(bins(u),h_zhu(u),'g-','LineWidth',2);

xlabel('Average localization error as fraction of face size');
ylabel('Fraction of the num. of testing faces');

legend('ours', 'zhu et al');
grid on;
fontSize = 15;
lineWidth = 2;


backgroundColor = [1 1 1];
pubgraph(h,fontSize,lineWidth,backgroundColor);
xlim([0 .3])
ylim([0 1]);

%     faceLengths(t) = norm(results(t).gt_pts(1,:)-results(t).gt_pts(5));              
%     delta_zhu(:,t) = sum((results(t).zhu_res-results(t).gt_pts).^2,2).^.5;
%     delta_ours(:,t) = sum((results(t).our_res-results(t).gt_pts).^2,2).^.5;
%     cares(:,t) = results(t).gt_care;
% end

% % 
% % 
% % % goods = ~[results.zhu_bad] & faces_detected;
% % delta_zhu = delta_zhu(:,goods);
% % delta_ours = delta_ours(:,goods);
% % faceLengths = faceLengths(goods);
% % cares = cares(:,goods);
% % 
% % mean_error_zhu = delta_zhu./repmat(faceLengths,size(delta_ours,1),1);
% % mean_error_zhu = mean(mean_error_zhu);
% % % mean_error_zhu = sum(mean_error_zhu.*cares)./sum(cares);
% % mean_error_ours = delta_ours./repmat(faceLengths,size(delta_ours,1),1);
% % mean_error_ours = mean(mean_error_ours);
% % % mean_error_ours = sum(mean_error_ours.*cares)./sum(cares);
% % 
% figure,hist(mean(delta_ours)./mean(delta_zhu),100)
% hold on; plot(mean(delta_zhu),'r-')


% sort(delta_ours./delta_zhu,2)
% plot(sort(delta_ours./delta_zhu,2)')
% mean(delta_zhu-delta_ours,2)

%% now on the COFW dataset...
load cofw_my_landmarks;% cofw_res
R = load('/home/amirro/code/3rdparty/rcpr_v1/data/COFW_test.mat');
% x2(R.IsT{1});plotPolygons(U,'g.');
%%
my_mean_diffs = zeros(size(R.IsT));

%%
% hold on;showCoords(U)
%cofw_to_my_coords = [14 16 26 23 24 29 21];
cofw_to_my_coords = [14 16 23 24 29 21];
[u,iu] = sort(my_mean_diffs,'descend');
% iu = 1:length(R.IsT)
for k = 1:length(R.IsT)
    ii = iu(k)
    u(k)
    U = R.phisT(ii,1:2*end/3);
    U =reshape(U,[],2);
    U = U(cofw_to_my_coords,:);
    u_pupils = U(1:2,:);
    u_pupilDist = sum((u_pupils(1,:)-u_pupils(2,:)).^2,2).^.5;
    my_local_predictions = boxCenters(cofw_res(ii).kp_local([1 2 4:end],1:4));
    my_global_predictions = boxCenters(cofw_res(ii).kp_global([1 2 4:end],1:4));
    my_predictions = (my_global_predictions+my_local_predictions)/2;
    curDiff = sum((my_predictions-U).^2,2).^.5;
    curDiff = curDiff/u_pupilDist;
    my_mean_diffs(ii) = mean(curDiff);
%     mean(curDiff)
    clf; imagesc2(R.IsT{ii});plotPolygons(my_predictions,'g.');
    plotPolygons(U,'r.');
    pause
end
mean(my_mean_diffs(my_mean_diffs<.5))
%%
clf;shapeGt('draw',regModel.model,R.IsT{1},R.phisT(1,:)); drawnow;pause(.01);

%%
h_ours_cofw = cumulative_error(my_mean_diffs);

% h_zhu = cumulative_error(mean_error_zhu);
u = find(bins <= .3);
h  = figure(1); plot(bins(u),h_ours(u),'r-','LineWidth',2);
