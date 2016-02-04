function [samples,locs,dists] = findDiscriminativePatches(conf,A,B,dists12,top_choice)
% findDiscriminativePatches Given the assymetric minimal distance matrix
% from A to B, finds the top k patches in image of A which sets
% it apart from the set of images B.
if (nargin < 5)
    top_choice =10;
end
close all;
samples = {};
locs = {};
dists = {};

for ii = 1:length(A)
    fprintf(1,'image %d out of %d (%02.2f%%)\n',ii,length(A),100*ii/length(A));
    if (ischar(A{ii}))
        I = toImage(conf,getImagePath(conf,A{ii}));
    else
        I = A{ii};
    end
    %     [X,locs_] = sampleHogs(conf,{I},'',inf,0);
    [X,uus,vvs,scales,t] = allFeatures(conf,I);
    
    bads = false(size(B));
    for k = 1:length(bads)
        if (length(dists12)<ii || isempty(dists12{ii}))
            bads(k) = true;
        else
            p = dists12{ii}{k};
            if (length(p)~=size(X,2))
                bads(k) = true;
            end
        end
    end
    
    if sum(bads)==length(bads)
        continue;
    end
    
    Q = cat(1,dists12{ii}{~bads}); % min dist to all images: length(A) x numel(
    %     Q(isinf(Q)) = inf;
    [minQ,iMinQ] = min(Q,[],1); % min dist to ANY image (global minimum)
    
    [maxMin,iMaxMin] = sort(minQ,'descend'); % find patch whos minimal distance is **maximal**
    
    %         close all;
    %         R = zeros(size(I,1),size(I,2));
    %         [ bbs ] = uv2boxes( conf,uus,vvs,scales,t);
    %         for k = iMaxMin(1)
    %             b = round(bbs(k,1:4));
    %             b(1) = max(b(1),1);
    %             b(2) = max(b(2),1);
    %             b(3) = min(b(3),size(I,2));
    %             b(4) = min(b(4),size(I,1));
    %             R(b(2):b(4),b(1):b(3)) = max(R(b(2):b(4),b(1):b(3)),minQ(k));
    %         end
    %
    %         R = R-min(R(:));
    %         R = R/max(R(:));
    %         R = cat(3,R,R,R);
    %         I = im2double(I);
    %         R = I.*R.^4;
    %         figure,imshow([I,R]);
    %
    %         pause;
    %         continue;
    % obtain at most top_choice patches...
    
    top = iMaxMin(1:min(10*top_choice,length(iMaxMin)));
    %     to_sample = minQ;
    %     top = double(weightedSample(minQ,minQ, 5*top_choice));
    %     bbs = locs_;
    [ bbs ] = uv2boxes( conf,uus(top),vvs(top),scales(top),t);
    %
    overlaps = boxesOverlap(bbs);
    
    [m,n] = find(overlaps>.2);
    
    removed = false(size(overlaps,1),1);
    for ki = 1:length(n)
        % only remove a box which overlaps with a box which
        % hasn't been removed yet.
        if (~removed(m((ki))))
            removed(n(ki)) = true;
        end
    end
    %
    %     window_inds(removed) = [];
    %     boxes(removed,:) = []; % just keep the boxes themselves for later visualization
    %%
    
    
    f = find(~removed);
    f = f(1:min(top_choice,length(f)));
    
    bbs = bbs(f,:);
    top = top(f);
    dists{ii} = minQ(top);%maxMin(top);
    bbs(:,11) = ii;
    samples{ii} = X(:,top);
    locs{ii} = bbs;
    %     close all;
    %     h = figure('visible','off');imshow(I);
    %     hold on; plotBoxes2(bbs(:,[2 1 4 3]),'Color','g','LineWidth',2);
    %
    %     frame = getframe(gcf);
    %     [img,~] = frame2im(frame);
    %
    %     imwrite(img,sprintf('mdf/%03.0f.jpg',ii));
    %     pause;
end
% dists = [dists{:}];
% samples = cat(2,samples{:});
% locs = cat(1,locs{:});