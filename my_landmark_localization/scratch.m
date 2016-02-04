%% learn to predict given the location of each keypoint if it is 
%% occluded or not
% create windows around each keypoint, and classify if it is occluded or
% visible

sub_windows = createSubWindows(IsTr,phisTr);
all_subs = cutOutSubWindows(IsTr,sub_windows)
sub_windows_test = createSubWindows(IsT,phisT);
all_subs_test = cutOutSubWindows(IsT,sub_windows_test)

%%
curKP = 1;
curSubWindows = squeeze(sub_windows(:,curKP,:));
a = all_subs(:,curKP);
curX = getImageStackHOG(a);
curSubWindows_test = squeeze(sub_windows_test(:,curKP,:));
a_test = all_subs_test(:,curKP);
curX_test = getImageStackHOG(a_test);
% model_occ.w*curX_test
% curSubWindows_test(:,end)*2-1
% figure,plot(model_occ.w*curX_test,curSubWindows_test(:,end)*2-1,'r.')
%
clf; figure(1)
opts_string = '-s 2 -c .01 -B 1';
model_occ = train(curSubWindows(:,5)*2-1, sparse(double(curX)), opts_string, 'col');
vl_pr(curSubWindows_test(:,end)*2-1,model_occ.w(1:end-1)*curX_test)
%
% x2(a(curSubWindows(:,5)==1));
% 
% figure(2)
x2(a_test(model_occ.w*curX_test+model_occ.w(end) > 0))


%%
figure(1); clf;
for p = 1:length(IsT)
    clf; imagesc2(IsT{p});
    cur_xy = squeeze(phisT(p,:,1:2));
    showCoords(cur_xy);
    
    drawnow
    %plotPolygons(factors(p)*xy_t(p,:),'g.');
    pause
end

%%
mu1 = [1 2];
Sigma1 = [2 0; 0 0.5];
mu2 = [-3 -5];
Sigma2 = [1 0;0 1];
rng(1); % For reproducibility
X = [mvnrnd(mu1,Sigma1,1000);mvnrnd(mu2,Sigma2,1000)];

GMModel = fitgmdist(X,2);
figure
y = [zeros(1000,1);ones(1000,1)];
h = gscatter(X(:,1),X(:,2),y);
hold on
ezcontour(@(x1,x2)pdf(GMModel,[x1 x2]),get(gca,{'XLim','YLim'}))
title('{\bf Scatter Plot and Fitted Gaussian Mixture Contours}')
legend(h,'Model 0','Model1')
hold off


%%
%% make a simply graphical model just to understand what happens with the edges stuff...

%adj = [0 1;1 0];
adj = [0 1;0 0];
adj = adj+adj';
[xx,yy] = meshgrid(1:5,1:5);
nStates = int32(numel(xx(:)));
unary_scores = ones(numel(xx),2);
useMex = false;
[edgeStruct] = UGM_makeEdgeStruct(adj,nStates,useMex);
pairwise_scores = zeros(25,25,1); % a single edge.
xx = xx(:);
yy = yy(:);
% unary_scores(yy<4,:) = 0;
nodePot = unary_scores';

for t1 = 1:length(xx)
    x1 = xx(t1);
    for t2 = 1:length(xx)
        x2 = xx(t2);
        pairwise_scores(t1,t2,1) = x1 > x2;
    end
end

[best_configuration] = UGM_Decode_Tree(nodePot,pairwise_scores,edgeStruct)


%% generate two 1-d distributions, and model them the same way as you did above.
%


x1 = randn(100,1);
x2 = randn(100,1)+1;
p = fitgmdist(x1-x2,1);

%adj = [0 1;1 0];
adj = [0 1;0 0];
adj = adj+adj';
xx = 1:.1:5;
% [xx,yy] = meshgrid(1:5,1:5);
nStates = int32(numel(xx(:)));
unary_scores = ones(numel(xx),2);
useMex = false;
[edgeStruct] = UGM_makeEdgeStruct(adj,nStates,useMex);
pairwise_scores = zeros(length(xx)); % a single edge.
xx = xx(:);
% unary_scores(yy<4,:) = 0;
nodePot = unary_scores';

for t1 = 1:length(xx)
    x1 = xx(t1);
    for t2 = 1:length(xx)
        x2 = xx(t2);        
        pairwise_scores(t1,t2,1) = log_lh(p,x2-x1);
    end
end


% for t1 = 1:length(xx)
%     x1 = xx(t1);
%     for t2 = 1:length(xx)
%         x2 = xx(t2);
%         pairwise_scores(t1,t2,1) = x1 > x2;
%     end
% end

[best_configuration] = UGM_Decode_Tree(nodePot,pairwise_scores,edgeStruct)

%%
% otherwise: let's make a lookup table.
x1 = round(randn(10000,1)+15);
x2 = round(randn(10000,1)+20);
x1 = max(1,x1); x2 = max(1,x2);
nStates = int32(max(max(x1),max(x2)));
% counting
% figure,plot(x1,x2,'r.')
% p = fitgmdist(x1-x2,1);
%adj = [0 1;1 0];
adj = [0 1;0 0];
adj = adj+adj';
% [xx,yy] = meshgrid(1:5,1:5);
unary_scores = ones(nStates,2);
useMex = false;
[edgeStruct] = UGM_makeEdgeStruct(adj,nStates,useMex);
pairwise_scores = zeros(nStates);
nodePot = unary_scores';
pairwise_scores = accumarray([x1 x2],1,[nStates nStates]);
[best_configuration] = UGM_Decode_Tree(nodePot,pairwise_scores,edgeStruct)

%% 
zfun = @(u,v) v-u;
p = fitgmdist(zfun(x1,x2),1);
[xx2,xx1] = meshgrid(1:nStates,1:nStates);
pairwise_scores = pdf(p,double(zfun(xx1(:),xx2(:))));
pairwise_scores = reshape(pairwise_scores,size(xx1));
figure(1); imagesc((pairwise_scores));
[best_configuration] = UGM_Decode_Tree(nodePot,pairwise_scores,edgeStruct)

%% make it 2d
% otherwise: let's make a lookup table; 
% first: make a random arrangement of points in space around 
% a set of fixed centers
clc
close all
mus = [15 15;...
    15 20;...
    21 17;...
    24 20;...
    10 15];
curAdj = l2(mus,mus).^.5;
[adj,pred] = graphminspantree(sparse(curAdj),'Method','Kruskal');

adj=adj;
full(adj)
xs = {};
z = 0;

for t = 1:size(mus,1)   
%     t
    m = (bsxfun(@plus,randn(10000,2),mus(t,:)));
    m(m<=1) = 1;
%     plotPolygons(m,'
    z = max(z,max(m(:)));
    xs{t} = m;
end
z = ceil(z);

% x1 = round(randn(1000,2)+15);
% x2 = round(randn(1000,2)+20);
% x1 = max(1,x1); x2 = max(1,x2);
% z = int32(max(max(x1(:)),max(x2(:))));
[locs_y,locs_x] = meshgrid(1:z,1:z);

locs_x = locs_x(:); locs_y = locs_y(:);
my_shuffle = randperm(length(locs_x));

all_locs = [locs_x(:),locs_y(:)];
nStates = z^2;
[loc_j,loc_i] = meshgrid(1:nStates,1:nStates);
% counting
% figure,plot(x1,x2,'r.')
% p = fitgmdist(x1-x2,1);
%adj = [0 1;1 0];
[ii,jj] = find(adj);

figure(10); clf; gplot2(adj,mus,'r-+','LineWidth',2);

edges = [ii jj];
nEdges = length(ii);
adj = adj+adj';
% [xx,yy] = meshgrid(1:5,1:5);
nNodes = length(xs);
unary_scores = ones(nStates,nNodes);
useMex = false;
[edgeStruct] = UGM_makeEdgeStruct(adj,nStates,useMex);
% pairwise_scores = zeros(nStates);
nodePot = unary_scores';
pairwise_scores = zeros(nStates,nStates,nEdges);
for iEdge = 1:nEdges
    edge_i = edges(iEdge,1);
    edge_j = edges(iEdge,2);
    x1 = round(xs{edge_i});
    x2 = round(xs{edge_j});
    x1_1 = sub2ind([z z],x1(:,1),x1(:,2));
    x2_1 = sub2ind([z z],x2(:,1),x2(:,2));
    pairwise_scores(:,:,iEdge) = accumarray([x1_1 x2_1],1,[nStates nStates]);
end

[best_configuration] = UGM_Decode_Tree(nodePot,pairwise_scores,edgeStruct);
all_locs(best_configuration,:) -(mus)
p = pairwise_scores(:,:,1);
[m,im] = max(p(:));
[imax,jmax] = ind2sub(size(p),im);
[all_locs(imax,:);all_locs(jmax,:)]

figure(2); subplot(1,2,1);
gplot2(adj,mus,'r-+','LineWidth',2); title('orig');
showCoords(mus);
q = 25;
axis equal
xlim([0 q]);ylim([0 q]);

subplot(1,2,2); 
% hold on;
%showCoords(mus);title('decoded');
gplot2(adj,all_locs(best_configuration,:),'r-+','LineWidth',2); 
showCoords(all_locs(best_configuration,:));
axis equal
xlim([0 q]);ylim([0 q]);

% best_configuration
% all_locs(best_configuration(2),:)-all_locs(best_configuration(1),:)

%%
zfun = @(u,v) v-u;
pairwise_scores = zeros(nStates,nStates,nEdges);
[xx1,xx2] = meshgrid(1:nStates,1:nStates);
for iEdge = 1:nEdges
    edge_i = edges(iEdge,1);
    edge_j = edges(iEdge,2);
    x1 = xs{edge_i};
    x2 = xs{edge_j};
    p = fitgmdist(zfun(x1,x2),1);    
    pairwise_scores(:,:,iEdge) = reshape(pdf(p,double(zfun(all_locs(xx1,:),all_locs(xx2,:)))),...
        size(xx1));
end
%pairwise_scores = pdf(p,double(zfun(all_locs(xx1,:),all_locs(xx2,:))));
% pairwise_scores = reshape(pairwise_scores,size(xx1));
% figure(1); imagesc((pairwise_scores));
[best_configuration] = UGM_Decode_Tree(nodePot,pairwise_scores,edgeStruct)
all_locs(best_configuration,:) -(mus)

figure(2); subplot(1,2,1);
gplot2(adj,mus,'r-+','LineWidth',2); title('orig');
showCoords(mus);
q = 25;
axis equal
xlim([0 q]);ylim([0 q]);

subplot(1,2,2); 
% hold on;
%showCoords(mus);title('decoded');
gplot2(adj,all_locs(best_configuration,:),'r-+','LineWidth',2); 
showCoords(all_locs(best_configuration,:));
axis equal
xlim([0 q]);ylim([0 q]);







% all_locs(best_configuration(2),:)-all_locs(best_configuration(1),:)


