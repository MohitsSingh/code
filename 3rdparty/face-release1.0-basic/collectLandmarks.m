function [all_lm,all_inds] = collectLandmarks(inDir)
d = dir(fullfile(inDir,'*.mat'));
% load all landmarks
all_lm = {};
for t = 1:length(d)
    if (mod(t,100)==0)
        t/length(d)
    end
    u = str2num(d(t).name(1:end-4));
    %         u
    load(fullfile(inDir,d(t).name));
    all_lm{t}=  landmarks;
end

%     all_lm_bu = all_lm;
all_lm = cat(2,all_lm_bu{:});

%     valids = [all_lm.isvalid]

all_inds = {};
for t = 1:length(all_lm)
    all_inds{t} = t*ones(3,1);
end
all_inds = cat(1,all_inds{:});

for t = 1:length(all_lm)
    if isempty(all_lm(t).s)
        all_lm(t).s = -inf;
    end
end
ss = [all_lm.s];
goods = ~isinf(ss);
all_lm = all_lm(goods);
all_inds = all_inds(goods);
