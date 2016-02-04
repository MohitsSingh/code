function theSals = showNeighborDifferences(f0,f1,imgs,INN,sizes,sal,ref_imgs)
theSals = {};
debug_ = true;
useSal = nargin >= 6 && ~isempty(sal);
for k = 1:length(imgs)
    k
    
    img = imgs{k};
    knn = 20;
    neighbors = f1(:,INN(k,1:knn));
    d = mean(neighbors,2);
    vars = var(neighbors ,0,2);
    vars = vars/max(vars(:));
    if (debug_)
        clf;
        %         figure(1);
        subplot(3,3,1);
        imagesc(img);axis image;title('img');
        subplot(3,3,2);
        imagesc(HOGpicture(reshape(d.^2,sizes{1})));axis image;title('mean neighbors');
        
%         imagesc(vl_hog('render',reshape(d.^2,sizes{1}),'NumOrientations',18));
        
        subplot(3,3,9);
        
        ddd = sum(reshape(vars.^2,sizes{1}),3);
        ddd = imresize(ddd,[128 128]);
        imagesc(ddd);axis image;title('vars');
    end
    
    %     meanDiff = bsxfun(@minus,f0(:,k) ,f1(:,INN(k,1:knn)));
    meanDiff = f0(:,k)-d;
    z = inf(sizes{1}(1:2));
    for kk = 1:size(meanDiff,2)
        r = reshape(meanDiff(:,kk),[sizes{1}]);
        r = sum(r.^2,3);
        z = min(z,r);
    end
    %     meanDiff = z.^.5;
    %   meanDiff = reshape(meanDiff,[10 10 NaN]);
    
    %     meanDiff = min(bsxfun(@minus,f0(:,k) ,f1(:,INN(k,1:knn))),[],2);
    
    aa = sum(reshape(abs(meanDiff)./vars,sizes{1}),3);
    aa = imresize(aa,[128 128]);
    theSals{k} = aa;
    if (debug_)
        subplot(3,3,3);
        imagesc(HOGpicture(reshape( (meanDiff).^2,sizes{1})));axis image;title('diff');
        
        
        subplot(3,3,8);
        
        imagesc((aa));axis image;title('diff / vars');
        
    end
    %     dd = sum(reshape(meanDiff.^2,sizes{1}),3);
    dd = z;
    dd = imresize(dd,[size(img,1) size(img,2)]);
    if (debug_)
        subplot(3,3,4);
        imagesc(dd);axis image;title('diff magnitude');colorbar;
    end
    
    % second phase - learn the distance again, re-weighted by the
    % 1-probability for match.
    
    for z = 1:1
        %         z
        m = meanDiff;
        m = reshape(m,sizes{1});
        m = repmat(sum(m.^2,3),[1 1 size(m,3)]);
        m = m-min(m(:));
        m = m/max(m(:));
        newDists = sum((bsxfun(@times,1-m(:),bsxfun(@minus,f0(:,k),f1))).^2);
        [r,ir] = sort(newDists,'ascend');
        d = mean(f1(:,ir(1:knn)),2);
        if (debug_)
            subplot(3,3,5);
            imagesc(HOGpicture(reshape(d.^2,sizes{1})));axis image;title('mean neighbors 2');
            
            subplot(3,3,6);
        end
        meanDiff = f0(:,k)-d;
        dd = sum(reshape(meanDiff.^2,sizes{1}),3);
        dd = imresize(dd,[size(img,1) size(img,2)]);
        %         theSals{k} = dd;
        
        if (debug_)
            imagesc(HOGpicture(reshape( (meanDiff).^2,sizes{1})));axis image;title('diff 2');
            
            subplot(3,3,7);
            imagesc(dd);axis image;title('diff magnitude 2');
        end
        %     pause(.1)
    end
    
    
    if (useSal)
        ss  =im2single(sal{k}).*dd;
        %         'dgdfg'
        theSals{k} = ss;
        if (debug_)
            subplot(3,3,8);
            imagesc(sal{k});axis image;title('saliency');
            subplot(3,3,9);
            imagesc(ss);axis image;title('sal x diff');
        end
    end
    %     dd = repmat(dd,[1 1 3]);
    %     dd = dd-min(dd(:));
    %     dd = dd/max(dd(:));
    %     dd = dd.^2;
    %     dd = dd.*im2double(img);
    
    %
    if (debug_)
        
        %         figure(2);
        %         imshow(multiImage(ref_imgs(INN(k,1:5)),false,true));
        pause;
    end
end