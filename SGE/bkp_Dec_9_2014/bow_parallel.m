function bow_parallel(baseDir,d,indRange,outDir,tofix)

cd ~/code/mircs;
initpath;
config;

fprintf('initializing dictionaries...');
featConf = init_features(conf,256);
featConf = [featConf,init_features(conf,1024)];
fprintf('done!\n');
if (nargin < 5)
    tofix = false;
end
% tofix = false; %TODO
for k = 1:length(indRange)
    imagePath = fullfile(baseDir,d(indRange(k)).name);
    [~,filename,~] = fileparts(imagePath);
    for iFeatType = 1:length(featConf)
        suffix = featConf(iFeatType).suffix;
        resFileName = fullfile(outDir,[filename '_' suffix '.mat']);
        fprintf('\nchecking if filename %s exists :.... ',resFileName);
        if (exist(resFileName,'file'))
            if (tofix)
                
                %
                try (load(resFileName))
                    bowImage = uint16(bowImage); % make it compact
                    save(resFileName,'bowImage');
                    fprintf('fixed with image %s!\n',filename);
                catch e
                    fprintf('----------------------->deleting %s !:.... \n',resFileName);
                    delete(resFileName);
                end
                %             end
                %             disp('already exists! skipping ... \n');
                %             continue;
            end
            fprintf('YES:.... \n');
            continue;
            
        end
        
        I = imread(imagePath);
        [F,D] = vl_phow(I,featConf(iFeatType).featArgs{:});
        bins = minDists(single(D),single(featConf(iFeatType).bowmodel.vocab),5000);
        
        bowImage = uint16(makeBowImage(I,F,bins));
        %         F = single(F); %#ok<NASGU>
        save(resFileName,'bowImage');
    end
    fprintf('done with image %s!\n',filename);
end
fprintf('\n\n\finished all images in current batch\n\n\n!\n');
end