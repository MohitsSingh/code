function showPredsHelper2(fra_db,imdb,k)
outPath = '~/storage/fra_action_fcn';
labels_full = {'none','face','hand','drink','smoke','bubbles','brush','phone'};
p = j2m(outPath,fra_db(k));
L = load(p);
coarse_probs = L.scores_full_image;
fine_probs = L.scores_hires;
coarse_probs = bsxfun(@rdivide,exp(coarse_probs),sum(exp(coarse_probs),3));
[~,coarse_pred] = max(coarse_probs,[],3);
[~,fine_pred] = max(fine_probs,[],3);
%     zoomBox = inflatebbox(region2Box(imdb.labels{k}>2),4,'both');
zoomBox = inflatebbox(fra_db(k).faceBox,1.5,'both');
%     showPredictions(single(imdb.images_data{k}),coarse_pred,coarse_probs,labels_full,1,zoomBox);
%     showPredictions(single(imdb.images_data{k}),fine_pred,fine_probs,labels_full,2,zoomBox);
% % [h1_coarse,h2_coarse,I1] = showPredictions(single(imdb.images_data{k}),coarse_pred,coarse_probs,labels_full,1,zoomBox);
% % [h1_fine,h2_fine,I2] = showPredictions(single(imdb.images_data{k}),fine_pred,fine_probs,labels_full,2,zoomBox);
[h1_coarse,h2_coarse,I1] = showPredictions(single(imdb.images_data{k}),coarse_pred,coarse_probs,labels_full,1);
[h1_fine,h2_fine,I2] = showPredictions(single(imdb.images_data{k}),fine_pred,fine_probs,labels_full,2);
return;


ff ='figures/coarse_to_fine';


x2(I1)
x2(I2)

h=figure();imagesc([I1;I2]);axis image;axis off;
text_start = [10,15];

t = text(text_start(1),text_start(2),'coarse prediction');
t.Color = [1 0 0];
t.FontSize = 15;
t.FontWeight = 'bold';
t = text(text_start(1),text_start(2)+size(I1,1),'fine prediction');
t.Color = [1 0 0];
t.FontSize = 15;
t.FontWeight = 'bold';

nLabels = length(labels_full);
R = colormap(jet(nLabels));
colormap(R);
lcolorbar(labels_full,'Location','horizontal')
% h = gcf;
saveTightFigure(h,fullfile(ff,[fra_db(k).imageID(1:end-4) '_pred_coarse_and_fine.pdf']));


figure(h1_coarse);
saveTightFigure(h1_coarse,fullfile(ff,[fra_db(k).imageID(1:end-4) '_pred_coarse_12.pdf']));
figure(h2_coarse);
saveTightFigure(h2_coarse,fullfile(ff,[fra_db(k).imageID(1:end-4) '_prob_coarse_z.pdf']));
figure(h1_fine);
saveTightFigure(h1_fine,fullfile(ff,[fra_db(k).imageID(1:end-4) '_pred_fine_.pdf']));
figure(h2_fine);
saveTightFigure(h2_fine,fullfile(ff,[fra_db(k).imageID(1:end-4) '_prob_fine_.pdf']));

%%
I = imdb.images_data{k};
U = sum(coarse_probs(:,:,2:end),3);
