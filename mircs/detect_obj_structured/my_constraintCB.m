function yhat = my_constraintCB(param, model, x, y)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
[centers,scores,angles] = detect_with_offset(x,model,param);
% centers = detections(:,1:2);
% if (param.use_angle)
%     f1 = [sind(y(3)) cosd(y(3))];
%     f2 = [sind(angles) cosd(angles)];
%     angle_penalty = sum(bsxfun(@times,f1,f2),2);
% else
%     angle_penalty = 0;
% end
% offset_penalty = sum(bsxfun(@minus,centers,y(1:2)).^2,2);


losses = my_lossCB(param,[centers angles],y);

%angle_penalty = sum((bsxfun(@minus,f1,f2)).^2,2);
zz = scores+losses;%offset_penalty+angle_penalty;
[r,ir] = max(zz);

if (param.debugging)    
   if (model.w(1)~=0) 
    z = visualizeTerm(zz,centers,size2(x.img));
    clf; 
    imagesc2(sc(cat(3,z,x.img),'prob'));
    r = 0;
   end
end
yhat = [centers(ir,:) angles(ir)];