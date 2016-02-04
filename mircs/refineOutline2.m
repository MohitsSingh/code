
function [res,T,cost,inflateFactor] = refineOutline2(conf,imageID,bb,outlines,debug_)
res = 0;

if (~iscell(outlines))
    outlines = {outlines};
end

cost = 0;
conf.get_full_image = true; % work in full image coordinates for consistency
[I_orig,xmin,xmax,ymin,ymax] = getImage(conf,imageID);
if (min(dsize(I_orig,1:2)) < 30)
    res = [];
    return;
end
% bb = round(bb);
inflateFactor = 2;
bbs_new = inflatebbox(bb,inflateFactor*[1 1],'both',false);
bbs_new = clip_to_image(bbs_new,I_orig);

% bbs_new = bbs_new + [xmin ymin xmin ymin];
bbs_new = round(bbs_new);
I = cropper(I_orig,bbs_new);
ucmFile = fullfile(conf.gpbDir,strrep(imageID,'.jpg','_ucm.mat'));
load(ucmFile); % ucm
sz = size(ucm);
% work in global image coordinates.
ucm = cropper(ucm,bbs_new);
%U = imfilter(ucm,fspecial('gauss',[5 5],1));
U = imfilter(ucm,fspecial('gauss',[18 18],7));
resizeFactor = 1;

U = imresize(U,resizeFactor,'bilinear');
% U = ucm;
[M,O] = gradientMag(im2single(I));
opts = optimset;
opts.Display = 'off';
opts.MaxFunEvals = 10000;
bestT = [0 0 .3 0];
bestCost = inf;
I = imresize(I,resizeFactor,'bilinear');
for ii = 1:length(outlines)
    ii
    
    outline = bsxfun(@plus,outlines{ii},-bbs_new(1:2));
%     outline = outlines{ii};
%     pp = [resizeFactor*(outline(:,1)-bbs_new(1)),resizeFactor*(outline(:,2)-bbs_new(2))];

    %     pp = poly2mask(resizeFactor*(outline(:,1)-bbs_new(1)),resizeFactor*(outline(:,2)-bbs_new(2)),...
    %         size(I,1),size(I,2));
    yy = outline(:,2);
    xx = outline(:,1);
    %     [yy,xx] = find(bwperim(pp));
    
    if (isempty(xx))
        continue;
    end
    x0 = [0 0 .1 0 zeros(1,length(xx)) zeros(1,length(yy))]; % Tx Ty scale-1 rotation
    myCost = @(x) (ucmCost(I, U, xx,yy, x));
    
    % opts.PlotFcns = @optimplotfval;
    %opts.Display = 'iter';
    tic
    [T,cost] = fminsearch(myCost,x0,opts);
    disp(cost)
    if (cost < bestCost)
        bestCost = cost;
        bestT = T;
        x_ = xx;
        y_ = yy;
        sz = size(ucm);
        [x,y] = deform_fn(sz,x_,y_,T);
        if (debug_)
            
            inImage = inImageBounds(sz,[x y]);
            clf; subplot(1,2,1);
            imagesc(I); axis image; hold on;
                    plot(x_,y_,'m.');
            plot(x,y,'g+');
            subplot(1,2,2);
            imagesc(U); axis image; hold on;
                    plot(x_,y_,'m.');
            plot(x,y,'g-+');
            disp(cost);
            disp(bestT)
            pause(.1);
        end
        T(1) = bestT(1)*sz(2);
        T(2) = bestT(2)*sz(1);
        T(3) = 1+bestT(3);
        H = rotationMatrix(bestT(4));
        A = zeros(3);
        A(1:2,1:2) = H*T(3);
        A(1:2,3) = T([2 1])';
        A(3,3) = 1;
        res = [x y];
        res = bsxfun(@plus,res/resizeFactor,bbs_new(1:2)-[xmin ymin]);
        
        
        
%         %
%                 Z = zeros(size(ucm));
%                 Z(sub2ind2(size(Z),round([yy xx]))) = 1;
%                 RR = imtransform2(Z,A);
%                 imshow(RR); hold on; plot(x,y,'g.');
%                 drawnow
%                 pause(.1)
        
        
        %         imshow(RR)
        
    end
    
    
    
    % transform according to best T
    
    
end
% clf;imagesc(I_orig(ymin:ymax,xmin:xmax,:)); hold on; plot(res(:,1),res(:,2),'g--','LineWidth',2);
disp(bestCost)
% pause