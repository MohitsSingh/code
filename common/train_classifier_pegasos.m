function classifier = train_classifier_pegasos(x,y)
% if (nargin == 2)
%     balanceData = 1;
% end
% if (nargin < 3)
%     normalizeFeats = false;
% end
% if (nargin >=  3)
%     if (balanceData == -1 || balanceData == 1)
%         disp('balancing pos and neg data...');
%         balanceData = floor(size(negFeats,2)/size(posFeats,2));
%     elseif (balanceData == 0)
%         balanceData = 1;
%     end
%     posFeats = repmat(posFeats,[1 balanceData]);
% %     posFeats = repmat(posFeats,[1 1 balanceData]);
% end

% x = [posFeats,negFeats];

% l2 normalize the features?
% if (normalizeFeats)
%     x = normalize_vec(x);
%     x = bsxfun(@rdivide,x,sum(x.^2).^.5);
%     x = rootsift(x);
%     x = x(1:3:end,:);
% end

% if any(isnan(x(:)))
%     error('got nan values!');
% end
% y = zeros(size(x,2),1);
% y(1:size(posFeats,2)) = 1;
% y(size(posFeats,2)+1:end) = -1;
%classifier = Pegasos(x,y,'lambda',.000001);
classifier = Pegasos(x,y);
end
