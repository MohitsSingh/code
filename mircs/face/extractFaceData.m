function [keep,faceLandmarks,ss] = extractFaceData(conf,faceLandmarks,face_images,tt,D)

keep = false(size(faceLandmarks));
ss = {};
for k = 1:length(faceLandmarks)
    keep(k) = size(faceLandmarks(k).xy,1) == 68;
end

face_images = face_images(keep);
faceLandmarks = faceLandmarks(keep);
for q = 1:length(faceLandmarks)
    faceLandmarks(q).xy = boxCenters(faceLandmarks(q).xy);
    faceLandmarks(q).xy = faceLandmarks(q).xy(:);
end

D = D(:,keep);
D = D(keep,:);
% [DD,INN] = sort(D,2,'ascend');

if (nargin >=4 && ~isempty(tt))
    tt = tt(keep);
end

allBoxes = cat(2,[faceLandmarks.xy]);

% D = l2(allBoxes',allBoxes');
% [D,inn] = sort(D,2,'ascend');
knn = 15;
sig_ =10;

featureType = 'graph';

%DD = exp(-DD(k,:)/sig_);
DD = exp(-D/sig_);
for k = 1:length(face_images)
    %boxEstimates = mean(allBoxes(:,INN(k,1:knn)),2);
    w = DD(k,:)';
    w = w/sum(w);
    %boxEstimates = double(allBoxes(:,INN(k,:))*w);
    boxEstimates = allBoxes*w;
    
    boxEstimates = reshape(boxEstimates,size(boxEstimates,1)/2,2);
    curBox = allBoxes(:,k);
    curBox = reshape(curBox,size(curBox,1)/2,2);
    faceLandmarks(k).lowerRim = boxEstimates([60:-1:52 61:68],:);
    faceLandmarks(k).lowerRim_real = curBox([60:-1:52 61:68],:);
    faceLandmarks(k).boxEstimates = boxEstimates;
    
    
end
debug_ = true;
% extract features....
for  k = 1:length(face_images)
    if (nargin >=4  && ~isempty(tt) && ~tt(k))
        continue;
    end
    
    k
    I = face_images{k};
    currentBoxes =  faceLandmarks(k).lowerRim;
%     currentBoxes =  faceLandmarks(k).lowerRim_real;
    
    
    %     currentBoxes =  [currentBoxes currentBoxes];
    realBoxes = faceLandmarks(k).lowerRim_real;
    
    Z = zeros(size(I,1),size(I,2));
    Z = paintLines(Z,currentBoxes(:,[2 1]))>0;
    sLength = sum(Z(:));
    R = imdilate(Z,ones(8));
    
    %[freq, freq_emph, freq_app] = image_hist_RGB_3d(imname,n,gamma)
   
    
%     imagesc(squeeze(sum(freq3)))
    
    currentBoxes_i = round(inflatebbox([currentBoxes currentBoxes],[20 20],'both','abs'));
    %
    %     [R] = drawBoxes(Z,currentBoxes_i,[],1);
    
    %     p = multiCrop(conf,I,currentBoxes_i,[16 16]);
    currentBoxes_i(:,11) = 1;
%     p = visualizeLocs2_new(conf,{I},currentBoxes_i);
    
    n = size(currentBoxes,1);
    %     frames = [round(currentBoxes)';3*ones(1,n);zeros(1,n)];
    %     [F,D] = vl_sift(im2single(rgb2gray(I)),'Frames',frames);
    
    %     subplot(1,3,1);imshow(I);
    %     hold on;
    %     plot(currentBoxes(:,1),currentBoxes(:,2),'r+');
    %     subplot(1,3,2);imshow(im2double(I).*repmat(R,[1 1 3]));
    
    
    
    switch featureType
        case 'graph'
            
            G = im2double(rgb2gray(I));
            [gx,gy] = gradient(G);
            if (debug_)
                subplot(1,3,1);imshow(I);
                hold on;
                plot(currentBoxes(:,1),currentBoxes(:,2),'r-+');
                plot(realBoxes(:,1),realBoxes(:,2),'g-+');
            end
            g = (gx.^2+gy.^2).^.5;
            
            % make shortest path...
            [ii,jj] = sparse_adj_matrix(size(G),1,1);
            
            f = find(~R);
            
            %[c,ia,ib] = intersect(1:prod(size(G)),ii);
            tf = ismember(jj,f);
            ii(tf) = [];
            jj(tf) = [];
            %
            % % %     [c r]=ndgrid(1:128,1:128);
            % % %      GG = im2double(rgb2gray(I));
            % % %     v = GG(ii)-GG(jj);
            % % %     A = sparse(ii, jj, v, prod(size(G)), prod(size(G)));;
            % % %     imshow(I); hold on;
            % % %     gplot(A, [r(:) c(:)]);
            % % %     g(~R) = .5;
            
            
            EE = 100*double(edge(im2double(rgb2gray(I)),'canny'));
            %     figure,imshow(EE)
            [~,pX,pY,dist] = makeImageGraph(EE,R,currentBoxes(1,:),currentBoxes(end,:));
            
            %      v(v==0) = [];
            %     [dist_, path_, pred_]=graphshortestpath(A,min(ii),'Weights',v);
            
            %subplot(1,3,2);imshow(g.*R,[]);
            if (debug_)
                subplot(1,3,2);imshow(EE.*R+.1*EE,[]);
                hold on;
                plot(pX,pY,'r-');
                title(([num2str(dist) '  :  ' num2str(dist/sLength)]));
                
                %     [dist,path,pred] = graphshortestpath(A,min(ii));
                %     hold on;
                %     plot(currentBoxes(:,1),currentBoxes(:,2),'r+');
                
%                 subplot(1,3,3);imshow(multiImage(p,false));
            end
            % construct a graph between all pixels on this course.
            
            
            ss{k} = dist/sLength;
            
            if (debug_)
                pause;
            end
        case 'color'
            
            
            
%             II = vl_xyz2lab(vl_rgb2xyz(I));
%         II = rgb2lab(I);
            II = I;
        
            I_r = II(:,:,1);
            I_g = II(:,:,2);
            I_b = II(:,:,3);
            
            I_h = cat(3, I_r(R(:)), I_g(R(:)), I_b(R(:)));
            
            [freq, freq_emph, freq_app] = image_hist_RGB_3d((I_h),8);
            ss{k}= freq(:)/sum(freq(:));
    end
    %     D = D(:);
    
    %     figure,imshow(I);
    %     hold on;
    %
    %     vl_plotframe(F);
    %
    %     figure,imshow(I);
    %     hold on;
    %     vl_plotsiftdescriptor(D,F);
    
    %     faceLandmarks(k).F = F;
    %     faceLandmarks(k).D = D(:);
    %     %
    %     clf;
    %     imshow(I);
    %     hold on;
    %     plot(currentBoxes(:,1),currentBoxes(:,2),'r+');
    %     plotBoxes2(currentBoxes_i(:,[2 1 4 3]),'g');
    %
    %     pause;
    
end