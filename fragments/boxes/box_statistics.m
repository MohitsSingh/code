
% % % 
% % % nn = [];
% % % choice = 1:20
% % % for k = choice
% % %     
% % %     cls = globalOpts.VOCopts.classes{k}
% % %     set_ = 'train';
% % %     ovp = get_boxes_recall(globalOpts,set_,cls);
% % %     % possible recall graph...
% % %     
% % %     xout = 0:.01:1;
% % %     [n] = hist(ovp,xout);
% % %     nn = [nn,n'];
% % % end
% % % 
% % % ccc = [1:14 16:20];
% % % plot(xout,1-bsxfun(@rdivide,cumsum(nn(:,ccc)),sum(nn(:,ccc))));
% % % legend(globalOpts.VOCopts.classes(choice(ccc)));
% % % 
% % % plot(xout,1-cumsum(sum(nn,2))/sum(nn(:)));
% % % 
% % % nn_ = mean(nn(:,ccc),2);
% % % plot(xout,1-bsxfun(@rdivide,cumsum(nn_),sum(nn_)));
% % % 
% % % % plot(xout,1-cumsum(n)/sum(n))

cls = 'aeroplane'
train_boxes = gt_boxes(globalOpts,'train',cls);
test_boxes = gt_boxes(globalOpts,'val',cls);
[a0,b0,c0] = BoxSize(train_boxes);
a0 = a0-1;
b0 = b0-1;

% create 
[a1,b1,c1] = BoxSize(test_boxes);
a1 = a1-1;
b1 = b1-1;

random_boxes = {};
% sample other boxes...
t = 0;
for iID = 1:20:length(train_images)
    iID
    bf = getBoxesFile(globalOpts,train_images{iID});
    rec = PASreadrecord(getRecFile(globalOpts,train_images{iID}));
    b = load(bf);
    t = t+1;
    b.boxes(:,[1 3]) = b.boxes(:,[1 3])/rec.imgsize(2);
    b.boxes(:,[2 4]) = b.boxes(:,[2 4])/rec.imgsize(1);
    random_boxes{t} = b.boxes;
end
rr = cat(1,random_boxes{:});
[b2,a2,c2] = BoxSize(rr);
a2 = a2-1;
b2 = b2-1;
%%
figure,
choice = 1;
plot(a2(1:choice:end),b2(1:choice:end),'b.');
hold on;
plot(a0,b0,'r.');


%%
h = fspecial('gaussian',151,3);
r_factor = 20;

sum_pos = accumarray(round([a0,b0]*r_factor)+1,1,[r_factor r_factor]+2);
sum_pos = imfilter(sum_pos,h);
sum_pos = sum_pos/sum(sum_pos(:));
figure,imagesc((sum_pos));

a_0 = round(a0*r_factor)+1;
b_0 = round(b0*r_factor)+1;
pos_probs = sum_pos(sub2ind(size(sum_pos),a_0,b_0));

figure,hist(pos_probs,100)
% 
sum_neg = accumarray(round([a2,b2]*r_factor)+1,1,[r_factor r_factor]+2);
sum_neg = imfilter(sum_neg,h);
sum_neg = sum_neg/sum(sum_neg(:));
figure,imagesc((sum_neg));

figure,imagesc((log(sum_pos)-log(sum_neg)));

LLL = (log(sum_pos)-log(sum_neg));
save box_size_stats LLL

a_2 = round(a2*r_factor)+1;
b_2 = round(b2*r_factor)+1;
neg_probs = LLL(sub2ind(size(sum_pos),a_2,b_2));

figure,hist((neg_probs),100);

% check boxes of test set...
a_1 = round(a1*r_factor)+1;
b_1 = round(b1*r_factor)+1;

test_probs = sum_pos(sub2ind(size(sum_pos),a_1,b_1));

Z = zeros(size(sum_pos));
Z(sub2ind(size(sum_pos),a_1(test_probs~=0),b_1(test_probs~=0))) = 1;
figure,imagesc(Z);

figure,imagesc((sum_pos));



figure,hist(test_probs,100)

% now find 

% hold on;plot(a1,b1,'g.');
% title(cls);




% %
% % for cls = 1:20
% % close all;
% % VOCopts.classes{cls}
% %
% % f = (bboxesXclass(:,21)==1 &...
% %     bboxesXclass(:,23)~=1 &...
% %     bboxesXclass(:,cls)==1 & bboxesXclass(:,22) <= length(train_images));
% %
% % bboxes_ = bboxes(f,:);
% %
% % A = getBboxArea(bboxes_);
% %
% % lengths = bboxes_(:,4)-bboxes_(:,2)+1;
% % heights = bboxes_(:,3)-bboxes_(:,1)+1;
% %
% % [nn,xxout] = hist(lengths./heights,10);
% % figure,bar(xxout,nn/sum(nn));
% %
% % [n,xout] = hist(A.^.5,10);
% % figure,bar(xout,(n/sum(n)))
% %
% % bboxes_ = bboxes(bboxesXclass(:,21)==0,:);
% % % & bboxesXclass(:,cls)>0 & ...
% % %     bboxesXclass(:,cls) < 1,:);
% % A = getBboxArea(bboxes_);
% %
% % [n1,xout_] = hist(A.^.5,xout);
% % figure,bar(xout,cumsum(n1/sum(n1)))
% %
% % lengths= bboxes_(:,4)-bboxes_(:,2);
% % heights = bboxes_(:,3)-bboxes_(:,1);
% %
% % % sampleRatio = 100;
% % % lengths = lengths(1:sampleRatio:end);
% % % heights = heights (1:sampleRatio:end);
% % % % cl = colormap('jet');
% % % % % figure,plot(lengths,heights,'r+');
% % % % KK = 500;
% % % % [C, A] = vl_ikmeans(uint8([lengths,heights]'), KK);
% % % %
% % % close all
% %
% % aspect_ratios = sort(heights./lengths);
% % % length(aspect_ratios)
% % aspect_ratios(isinf(aspect_ratios)) = [];
% % % length(aspect_ratios)
% % aspect_ratios(isnan(aspect_ratios)) = [];
% % % length(aspect_ratios)
% % aspect_ratios = aspect_ratios(1:end);
% % % search for large aspect
% % [n,xout] = hist(aspect_ratios,[.001:.001:10]);
% %
% % % search for small aspect
% % % [n,xout] = hist(aspect_ratios,[.0001:.01:1]);
% %
% % figure,plot(xout,100*cumsum(n)/sum(n));
% % title(VOCopts.classes(cls))
% %
% % pause;
% % end
% %
% % % bar(xout,n);
% % % set(gca,'YScale','log')
% % %%
% % %%
% % figure; hold on;
% % % cl = get(gca,'ColorOrder');
% %
% % AT = vl_ikmeanspush(uint8([lengths,heights]'),C);
% %
% % ncl = size(cl,1);
% %
% % cl = cl(randperm(size(cl,1)),:);
% % for k = 1:KK
% %     sel = find(A == k);
% %     selt = find(AT == k);
% %     plot(lengths(sel),heights(sel),'.','Color',cl(mod(k,ncl)+1,:));
% %
% % end
% % %%
% %
% % [n,xout] = hist(double(A).^.25,10);
% % % plot(xout,n);
% % figure,
% % bar(xout,n);
% %
% % % find the mean overlap for each class....
% % %B = mean(bboxesXclass(:,1:20));
% % B = sum(bboxesXclass(:,1:20)>0);
% % figure,(hist(bboxesXclass(:,21)));
% % b = bboxesXclass(:,22);
% %
% % % find the maximal image of a category
% %
% % % plot the aspect ratios...
% %
% %
% % for k = 1:length(train_images)
% %
% % end
% %
% %
% % %%
% % s = []
% % for cls =2:20
% %
% % %     close all;
% % % cls = 2
% % VOCopts.classes(cls)
% % true_boxes = find(bboxesXclass(:,cls) == 1 & bboxesXclass(:,21) == 1 & ...
% %     bboxesXclass(:,23) ~= 1);
% % true_box_images = bboxesXclass(true_boxes,22);
% %
% % % for each image find the best box overlap
% %
% % nRatio = 1;
% % max_overlaps = zeros(size(true_box_images(1:nRatio:end)));
% % tic
% %
% % t_box = true_box_images(1:nRatio:end);
% % t_boxes = true_boxes(1:nRatio:end);
% % for iImage = 1:length(t_box)
% %     if (toc > 1)
% %         disp(num2str(100*iImage/length(t_box)));
% %         tic;
% %     end
% %     currentImage = t_box(iImage);
% %     other_boxes = bboxesXclass(:,22) == currentImage & ...
% %         bboxesXclass(:,21) ~= 1;
% %     other_boxes = bboxes(other_boxes,:);
% %
% %     % find the overlap of all boxes with the current
% %     currentBox = bboxes(t_boxes(iImage),:);
% % %     imagePath = getImageFile(globalOpts,all_images{currentImage});
% % %     clf,imshow(imagePath);
% % %     hold on;
% % %     plotBoxes2(currentBox,'Color','m','LineWidth',3);
% % %     plotBoxes2(other_boxes,'Color','g');
% % %     pause;
% %
% %
% % %     currentBox = repmat(currentBox,size(other_boxes,1),1);
% % %     [~, ~, bi] = BoxSize(BoxIntersection( currentBox,other_boxes));
% % %     [~, ~, bu] = BoxSize(BoxUnion( currentBox,other_boxes));
% % %     overlaps = bi./bu;
% %
% % %     max_overlaps(iImage) = max(overlaps);
% % end
% %
% % [n,xout] = hist(max_overlaps,0:.01:1);
% %
% % clf;plot(xout,1-(cumsum(n)/sum(n)));
% % title(VOCopts.classes(cls));
% % xlabel('overlap');
% % ylabel('% above overlap');
% % grid on;
% % pause;
% % end
% % %%
% % % now find all the rows fitting these images.
% % tf = ismember(b, a);
% %
% % q = bboxesXclass(tf,:);
% % q = q(q(:,21)==0,:);
% %
% % % find the avg. overlap score for the images....
% %
% % % find the max. overlap score for each image...
% % uniqueImages = unique(q(:,22));
% % pImage = zeros(size(uniqueImages));
% %
% % tic;
% % for iImage = 1:length(uniqueImages)
% %     if (toc > .3)
% %         disp(100*iImage/length(uniqueImages))
% %         drawnow
% %         tic;
% %     end
% %     isImage = q(:,22) == uniqueImages(iImage);
% %     pImage(iImage) = max(q(isImage,cls));
% % end
% %
% % [n,xout] = hist(pImage,.1:.03:1);
% % % figure,bar(xout,n/sum(n));
% % s = [s;n];
% % % pause
% % % end
% % % B =
% %
% %
% % % s_b = s;
% %
% % s_b = bsxfun(@rdivide,s_b, sum(s_b,2));
% %
% %
% %
% %
% %
% % %
% % % [phat,pci] = raylfit(n);
% % %
% % % p = raylpdf(xout,phat);
% % % figure,plot(xout,p)
% % % hold on;
% % % plot(xout,n,'r');
% % % set(gca,'YScale','log');
% % % set(gca,'XScale','log');
% %
% %
% % % find aspect ratio for classes....
% %
% % for cls = 1:20
% %     true_boxes = find(bboxesXclass(:,cls) == 1 & bboxesXclass(:,21) == 1 & ...
% %     bboxesXclass(:,23) ~= 1)';
% %
% %     lengths = bboxes(true_boxes,4)-bboxes(true_boxes,2)+1;
% %     heights = bboxes(true_boxes,3)-bboxes(true_boxes,1)+1;
% %
% %     figure,plot(heights,lengths,'r+');
% %
% %     LH0 = [lengths,heights];
% %
% %     not_true_boxes = find(bboxesXclass(:,cls) < .2 & bboxesXclass(:,21) ~= 1 & ...
% %     bboxesXclass(:,23) ~= 1)';
% %
% %     lengths = bboxes(not_true_boxes,4)-bboxes(not_true_boxes,2)+1;
% %     heights = bboxes(not_true_boxes,3)-bboxes(not_true_boxes,1)+1;
% %      hold on,plot(heights,lengths,'r+');
% %
% %     LH1 = [lengths,heights];
% %
% %
% %     posTrain = vl_colsubset(1:size(LH0,1),floor(size(LH0,1)/2));
% %     negTrain = vl_colsubset(1:size(LH1,1),floor(size(LH1,1)/2));
% %
% %     posTest = setdiff(1:size(LH0,1),posTrain);
% %     negTest = setdiff(1:size(LH1,1),negTrain);
% %
% %
% %     labels_train = [ones(size(posTrain)),...
% %         zeros(size(negTrain))];
% %
% %      labels_test = [ones(size(posTest)),...
% %         zeros(size(negTest))];
% %
% %     data_train = [LH0(posTrain,:);LH1(negTrain,:)];
% %
% %     data_test = [LH0(posTest,:);LH1(negTest,:)];
% %
% %
% %     svm_model = svmtrain(labels_train',data_train);
% %
% %
% %
% %
% %     figure,plot(heights,lengths,'r+');
% %
% %
% % end
% %
