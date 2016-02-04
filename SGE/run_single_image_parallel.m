function res = run_single_image_parallel(conf,I,reqInfo,moreParams)
if (nargin == 0)
    cd ~/code/action_recognition/;
    startup;
    res.dpm_conf = conf;
    res.conf = conf;    
    return;
end

conf = reqInfo.conf;
res = run_single_image(conf,I);