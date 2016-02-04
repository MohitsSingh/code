function pred = apply_kp_regressor(X,model)
pred = model.w(:,1:end-1)*X + repmat(model.w(:,end),1,size(X,2));
end