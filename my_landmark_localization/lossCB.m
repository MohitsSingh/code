function delta = lossCB(param, y, ybar)
% disp('loss');

% loss function is the normalized distance between the estimated
% locations and the real locations
if (param.lossType==2)
    delta =  sum( col(y(:,1:2)-ybar(:,1:2)) .^2);
else
    delta =  sum( col(abs(y(:,1:2)-ybar(:,1:2))));
end

%delta = abs(ybar - y) ;
if param.verbose
%     fprintf('delta = %f\n', delta) ;
end
% disp('end loss');

end