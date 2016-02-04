% Evaluate the strict Percentage of Correct Parts (PCP) and Percentage of
% Detected Joints (PDJ) on the Frames Labeled In Cinema (FLIC) using
% Observer-Centric (OC) Annotations.
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
addpath('./code');
addpath('./flic');
%% (i) load estimations and ground-truth
% load estimation results (i.e., locations of joints)
ests = load('./estimations/flic_oc_estimations_chen_nips14.mat', 'ests'); ests = ests.ests;
% load ground truth, downloaded from http://vision.grasp.upenn.edu/cgi-bin/index.php?n=VideoLearning.FLIC
examples = load('./flic/examples.mat', 'examples'); examples = examples.examples;
% convert the FLIC testset from person centric (PC) to observer centric (OC) annotation.
examples = flip_backwards_facing_groundtruth(examples([examples.istest]));
% convert to the canonical joint order
joint_order = [17,4,5,6,10,1,2,3,7];
gt = struct('joints', cell(numel(examples),1));
for ii = 1:numel(examples)
    gt(ii).joints = examples(ii).coords(1:2, joint_order)';
end
assert(numel(ests) == numel(gt));
% generate part stick from joints locations
for ii = 1:numel(ests)
    ests(ii).sticks = flic_joint2stick(ests(ii).joints);
    gt(ii).sticks = flic_joint2stick(gt(ii).joints);
end
%% (ii) dataset specific configurations
% --- part sticks ---
% symmetry_part_id(i) = j, if part j is the symmetry part of i (e.g., the left
% upper arm is the symmetry part of the right upper arm).
conf.symmetry_part_id = [1,2,5,6,3,4];
% show the average pcp performance of each pair of symmetry parts.
conf.show_part_ids = [3,4];
conf.part_name = {'U.arms', 'L.arms'};

% ---- joints ----
% the pair of reference joints is used to defined the scale of each pose.
conf.reference_joints_pair = [5, 6];     % right shoulder and left hip (from observer's perspective)
% symmetry_joint_id(i) = j, if joint j is the symmetry joint of i (e.g., the left
% shoulder is the symmetry joint of the right shoulder).
conf.symmetry_joint_id = [1,6,7,8,9,2,3,4,5];
conf.show_joint_ids = [3,4]; 
conf.joint_name = {'Elbo', 'Wris'};

%% (iii) show evaluation results
eval_methods = {'strict_pcp', 'pdj'};
show_eval(gt, ests, conf, eval_methods);
