function yhat = constraintCB_minimal(param, model, x, y)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
% disp('constraint');
% given an image representation, find the worst possible configuration for
% the locations y of the landmarks.

w_unary = model.w(1);
w_prior = model.w(2);
w_edges = model.w(3);

% w_unary = w(1);
% w_edges = w(2);

[yhat] = apply_graphical_model(param,x,...
    w_unary,w_prior,w_edges,y);


% show the most violated constraint for this image...
% if (model.w(1)~=0)
%     imgSize = param.imgSize;
%     yhat1 = yhat*imgSize;
%     clf; imagesc2(x); hold on; plotPolygons(y*imgSize,'g+');
%      plotPolygons(yhat1,'md');
%      drawnow
%      zzz = 0;
% end

end