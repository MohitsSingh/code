function sticks = flic_joint2stick(joints)
% The canonical joint order:
% 1 Nose
% 2 Left shoulder (from observer's perspective)
% 3 Left elbow
% 4 Left wrist
% 5 Left hip
% 6 Right shoulder
% 7 Right elbow
% 8 Right wrist
% 9 Right hip
%
% The canonical part stick order:
% 1 Torso
% 2 Head
% 3 Left Upper Arm
% 4 Left Lower Arm
% 5 Right Upper Arm
% 6 Right Lower Arm

stick_no = 6;
sticks = zeros(4, stick_no);
sticks(:, 1) = [(joints(2, :) + joints(6, :))/2, (joints(5, :) + joints(9, :))/2];  % Torso
sticks(:, 2) = [joints(1, :), (joints(2, :) + joints(6, :))/2];                     % Head
sticks(:, 3) = [joints(2, :), joints(3, :)];                                        % Left Upper Arm
sticks(:, 4) = [joints(3, :), joints(4, :)];                                        % Left Lower Arm
sticks(:, 5) = [joints(6, :), joints(7, :)];                                        % Right Upper Arm
sticks(:, 6) = [joints(7, :), joints(8, :)];                                        % Right Lower Arm
