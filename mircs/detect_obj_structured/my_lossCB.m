function delta = my_lossCB(param, y, ybar)
%     f2 = 1;
% end
if (param.angle_loss)
    f1 = [sind(ybar(3)) cosd(ybar(3))];
    f2 = [sind(y(:,3)) cosd(y(:,3))];
    angle_penalty = sum(bsxfun(@times,f1,f2),2);
else
    angle_penalty = zeros(size(y,1),1);
end
offset_penalty = sum(bsxfun(@minus,y(:,1:2),ybar(1:2)).^2,2);
% offset_penalty = offset_penalty*param.offset_factor;
%delta = .0001*offset_penalty+angle_penalty;
offset_penalty = min(offset_penalty,100);%*param.offset_factor);
delta = double(offset_penalty+angle_penalty);
% %delta = sum([y(1:2)-ybar(1:2),1-sum(f1.*f2)].^2);
% delta = sum((y(1:2)-ybar(1:2)).^2);
% if (param.use_angle)
%     f1 = [sind(y(3)) cosd(y(3))];
%     f2 = [sind(ybar(3)) cosd(ybar(3))];
%     delta = delta + sum(f1.*f2);
% end