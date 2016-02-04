function [ output_args ] = facial_landmarks_script( nimage )
%FACIAL_LANDMARKS_SCRIPT Summary of this function goes here
%   Detailed explanation goes here
cd('/home/amirro/code/mircs');
initpath;
config;
load m_test;
if (~exist('facial_landmarks_test','dir'))
    mkdir('facial_landmarks_test');
end
landmarks = detect_landmarks(conf,m_test(nimage),2);
save(fullfile('facial_landmarks_test',num2str(nimage)),'landmarks');
end

