function ds = detectRotated(im,model,thresh,theta)
ds = [];
im_orig = im;
sz = size(im);
sz = sz(1:2);
r = ceil(sum(sz.^2)^.5); % side of new image
padSize = ceil([max(r-size(im,1),0),max(r-size(im,2),0)]/2);
% theta = 20
im = padarray(im_orig,padSize,0,'both');
im = imrotate(im,theta,'bilinear','crop');
[ds, bs] = imgdetect(im, model,thresh);
top = nms(ds, 0.5);
ds = ds(top,:);
if (isempty(ds))
    disp('no candidates found above threshold');
    return;
end
disp(['found ' num2str(size(ds,1)) ' candidates']);


% figure(1); showboxes(im,ds);
% if ~isempty(ds)
%     ds(:,1:4) = ds(:,1:4)/2;
% end
% ds_orig = ds;

% ds = ds_orig;
R = rotationMatrix(theta*pi/180);

imCenter = size(im);
imCenter = fliplr(imCenter(1:2)/2);

bc = (ds(:,1:2)+ds(:,3:4))/2;
dd = bsxfun(@minus,bc,imCenter);
rr = (R*dd')';
rr = bsxfun(@plus,rr,imCenter);
ds(:,1:4) = ds(:,1:4)-[bc bc];
ds(:,1:4) = ds(:,1:4)+[rr rr];
ds(:,1:4) = bsxfun(@minus,ds(:,1:4),padSize([2 1 2 1]));
% figure(2); showboxes(im_orig,ds);
% disp('hit any key to continue')
% pause;