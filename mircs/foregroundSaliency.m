
function res = foregroundSaliency(conf,imageID,T,clip,debug_)
if (nargin < 3)
    T = .7;
end
if (nargin < 4)
    clip  = true;
end
if (nargin < 5)
    debug_  = false;
end
[I,I_rect] = getImage(conf,imageID);
L = load(j2m(conf.saliencyDir,imageID));
% curRect = 
res = L.res(1);
%res = 1-res.sal_bd+res.sal;
% res = normalise(res);
% res(res < T) = 0;
res = res.sal;
res = imResample(single(res),size(I));
% rr = poly2mask2(rect2pts(I_rect),size2(I));
%     res = imResample(single(res),size2(I));
% sz = size2(I);
% % I = imResample(I,size(res));
% rr = imResample(single(rr),size(res));
% if (clip)
%     res = res.*rr;
% end
%
% res = imResample(res,sz,'bilinear');
%
% if (debug_)
%
%     clf;
%     vl_tightsubplot(2,2,1);
%     imagesc2(I);
%     vl_tightsubplot(2,2,2); imagesc2(res);
%     q = blendRegion(I,double(res)>T,1);
%     vl_tightsubplot(2,2,3); imagesc2(q);
%     vl_tightsubplot(2,2,4); displayRegions(I,res);
%     colorspecs = {'g--','r--','b--'};
%     for t = 1:min(3,size(headDets,1))
%         hold on;
%         plotBoxes(headDets(t,:),colorspecs{t},'LineWidth',2);
%     end
%     drawnow;
%     pause;
% end
% end