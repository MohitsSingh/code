function pts_ixs = PointsInBB(pts, bb)

% pts_ixs so that pts(:,i) is inside bounding-box bb
%
% Input:
% - bb = [min_x max_x;
%         min_y max_y]
%

t1 = find(pts(1,:) >= bb(1,1));           % indeces of points passing first test: x coord >= min BB x
t2 = t1(find(pts(2,t1) >= bb(2,1)));      % second test: y coord >= min BB y
t3 = t2(find(pts(1,t2) <= bb(1,2)));      % x coord <= max BB x
pts_ixs = t3(find(pts(2,t3) <= bb(2,2))); % y coord <= max BB y
