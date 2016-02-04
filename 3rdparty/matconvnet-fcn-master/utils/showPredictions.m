function [h1,h2,I1] = showPredictions(rgb,pred,scores_,labels,n,zoomBox)
nLabels = size(scores_,3);

if nargin == 6
    zoomToBox(zoomBox);
end

mm = floor(sqrt(nLabels));
nn = ceil(nLabels/mm);
h2 = figure(2*n); clf;
text_start = [10,15];
if nargin == 6
    text_start = [zoomBox(1)+2,zoomBox(2)+5];
end
for t = 1:nLabels
    %tight_subplot
    vl_tightsubplot(mm,nn,t,'Margin',0);
    %     continue
    imagesc(sc(cat(3,scores_(:,:,t),rgb/255),'prob_jet',[0 1]));axis off;
    %imagesc(sc(cat(3,scores_(:,:,t),rgb/255),'prob_jet'));axis off;
    if nargin == 6
        zoomToBox(zoomBox);
    end
    %     imagesc2(sc(cat(3,scores_(:,:,t),rgb/255),'prob_jet'));%,[0 1]));
    if (~isempty(labels))
        
        t = text(text_start(1),text_start(2),labels{t});
        t.Color = [1 1 0];
        t.FontSize = 15;
        t.FontWeight = 'bold';
        %         title(labels{t});
    end
end
%%
clf
% ha = tight_subplot(mm,nn,[0 0],[0 0],[0 0]);%;,[.01 .03],[.1 .01],[.01 .01])
for t = 1:nLabels
    %tight_subplot
    %     axes(ha(t));
    subplot_tight(mm,nn,t);
    %imagesc2(sc(cat(3,scores_(:,:,t),rgb/255),'prob_jet',[0 1]));
    imagesc(sc(cat(3,scores_(:,:,t),rgb/255),'prob_jet',[0 1]));axis off;
    if nargin == 6
        zoomToBox(zoomBox);
    end
    %     imagesc2(sc(cat(3,scores_(:,:,t),rgb/255),'prob_jet'));%,[0 1]));
    %     continue;
    if (~isempty(labels))
        
        t = text(text_start(1),text_start(2),labels{t});
        t.Color = [1 1 0];
        t.FontSize = 15;
        t.FontWeight = 'bold';
        %         title(labels{t});
    end
end
%%
% %%

saturationValue =.6;
doExperimentalStuff = false;
if (~doExperimentalStuff )
    h1 = figure(2*n-1) ;
    I1 = rgb/255;
    clf;
    subplot(2,1,2);
    imagesc2(I1);
    subplot(2,1,1);
    
    pred(end)=nLabels;
    
    cmap_hsv = rgb2hsv(jet(nLabels));
    I1_hsv = rgb2hsv(I1);
    h = I1_hsv(:,:,1);
    s = I1_hsv(:,:,2);
    v = I1_hsv(:,:,3);
    for iPred = 2:length(labels)
        %             iPred=2;
        p = pred==iPred;
        h(p) = cmap_hsv(iPred,1);
        s(p) = saturationValue;
    end
    s(pred==1)=0;
    I1_hsv = cat(3,h,s,v);
    I1 = hsv2rgb(I1_hsv);
    clf;imagesc2(I1);
    title('predicted') ;
    if ~isempty(labels)
        R = colormap(jet(nLabels));
        colormap(R);
        lcolorbar(labels,'Location','vertical')
    end
    if nargin == 6
        zoomToBox(zoomBox);
    end    
else
    pred(pred<4 & pred>=1) = 1;
    zoomBox = round(zoomBox);
    pred = cropper(pred,zoomBox);
    rgb = cropper(rgb,zoomBox);
    h1 = figure(2*n-1) ;
%     pred = pred(1:end/2,:);
%     rgb = rgb(1:end/2,:,:);
    clf;
    vl_tightsubplot(2,1,n);
    
    if n==2
        t = text(text_start(1),text_start(2),'fine')
        t.Color = [1 1 0];
        t.FontSize = 15;
        t.FontWeight = 'bold';
    else
        t = text(text_start(1),text_start(2),'coarse')
        t.Color = [1 1 0];
        t.FontSize = 15;
        t.FontWeight = 'bold';
    end
    pred(end)=nLabels;
    I1 = rgb/255;
    cmap_hsv = rgb2hsv(jet(nLabels));
    I1_hsv = rgb2hsv(I1);
    h = I1_hsv(:,:,1);
    s = I1_hsv(:,:,2);
    v = I1_hsv(:,:,3);
    for iPred = 2:length(labels)
        %             iPred=2;
        p = pred==iPred;
        h(p) = cmap_hsv(iPred,1);
        s(p) = saturationValue;
    end
        s(pred==1)=0;
    I1_hsv = cat(3,h,s,v);
    I1 = hsv2rgb(I1_hsv);
    imagesc2(I1);
    title('predicted') ;
    if ~isempty(labels)
        R = colormap(jet(nLabels));
        colormap(R);
        lcolorbar(labels,'Location','vertical')
    end
    if nargin == 6
        zoomToBox(zoomBox);
    end
% % %     %--%--%--%--
% % %     %--%--%--%--
% % %     %--%--%--%--
% % %     %--%--%--%--
% % %     %--%--%--%--
% % %     vl_tightsubplot(2,1,2);
% % %     pred(end)=nLabels;
% % %     I1 = rgb/255;
% % %     cmap_hsv = rgb2hsv(jet(nLabels));
% % %     I1_hsv = rgb2hsv(I1);
% % %     h = I1_hsv(:,:,1);
% % %     s = I1_hsv(:,:,2);
% % %     v = I1_hsv(:,:,3);
% % %     for iPred = 2:length(labels)
% % %         %             iPred=2;
% % %         p = pred==iPred;
% % %         h(p) = cmap_hsv(iPred,1);
% % %         s(p) = saturationValue;
% % %     end
% % %     s(pred==1)=0;
% % %     I1_hsv = cat(3,h,s,v); 
% % %     I1 = hsv2rgb(I1_hsv);
% % %     imagesc2(I1);
% % %     title('predicted') ;
% % %     if ~isempty(labels)
% % %         R = colormap(jet(nLabels));
% % %         colormap(R);
% % %         lcolorbar(labels,'Location','horizontal')
% % %     end
% % %     if nargin == 6
% % %         zoomToBox(zoomBox);
% % %     end
end
