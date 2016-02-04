% Evaluate the strict Percentage of Correct Parts (PCP) and Percentage of
% Detected Joints (PDJ) on the Leeds Sports Poses (LSP) dataset using
% Observer-Centric (OC) Annotations.
% The canonical joint order:
% 1 Head top
% 2 Neck
% 3 Left shoulder (from observer's perspective)
% 4 Left elbow
% 5 Left wrist
% 6 Left hip
% 7 Left knee
% 8 Left ankle
% 9 Right shoulder
% 10 Right elbow
% 11 Right wrist
% 12 Right hip
% 13 Right knee
% 14 Right ankle
%
% The canonical part stick order:
% 1 Head
% 2 Torso
% 3 Left Upper Arm
% 4 Left Lower Arm
% 5 Left Upper Leg
% 6 Left Lower Leg
% 7 Right Upper Arm
% 8 Right Lower Arm
% 9 Right Upper Leg
% 10 Right Lower Leg
addpath('./code');
addpath('./lsp');
%% (i) load estimations and ground-truth
% load estimation results (i.e., locations of joints)
ests = load('./estimations/lsp_oc_estimations_chen_nips14.mat', 'ests'); ests = ests.ests;
% load ground truth, downloaded from http://www.comp.leeds.ac.uk/mat4saj/lsp.html
joints = load('./lsp/joints.mat', 'joints'); joints = joints.joints;
% convert the LSP testset from person centric (PC) to observer centric (OC) annotation.
% 1001 ~ 2000 are used as testing data
joints = lsp_pc2oc(joints(1:2,:,1001:2000));
% convert to the canonical joint order
joint_order = [14,13,9,8,7,3,2,1,10,11,12,4,5,6];
gt = struct('joints', cell(size(joints,3),1));
for ii = 1:size(joints,3)
    gt(ii).joints = joints(1:2, joint_order,ii)';
end
assert(numel(ests) == numel(gt));
% generate part stick from joints locations
for ii = 1:numel(ests)
    ests(ii).sticks = lsp_joint2stick(ests(ii).joints);
    gt(ii).sticks = lsp_joint2stick(gt(ii).joints);
end
%% (ii) dataset specific configurations
% --- part sticks ---
% symmetry_part_id(i) = j, if part j is the symmetry part of i (e.g., the left
% upper arm is the symmetry part of the right upper arm).
conf.symmetry_part_id = [1,2,7,8,9,10,3,4,5,6];
% show the average pcp performance of each pair of symmetry parts.
conf.show_part_ids = find(conf.symmetry_part_id >= 1:numel(conf.symmetry_part_id));
conf.part_name = {'Head', 'Torso', 'U.arms', 'L.arms', 'U.legs', 'L.legs'};

% ---- joints ----
% the pair of reference joints is used to defined the scale of each pose.
conf.reference_joints_pair = [6, 9];     % right shoulder and left hip (from observer's perspective)
% symmetry_joint_id(i) = j, if joint j is the symmetry joint of i (e.g., the left
% shoulder is the symmetry joint of the right shoulder).
conf.symmetry_joint_id = [2,1,9,10,11,12,13,14,3,4,5,6,7,8];
conf.show_joint_ids = find(conf.symmetry_joint_id >= 1:numel(conf.symmetry_joint_id)); 
conf.joint_name = {'Head', 'Shou', 'Elbo', 'Wris', 'Hip', 'Knee', 'Ankle'};

%% (iii) show evaluation results
eval_methods = {'strict_pcp', 'pdj'};
show_eval(gt, ests, conf, eval_methods);

