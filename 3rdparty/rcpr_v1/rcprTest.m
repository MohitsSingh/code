function [p,pRT] = rcprTest( Is, regModel, varargin )
% Apply robust cascaded shape regressor.
%
% USAGE
%  [p,pRT] = rcprTest( Is, regModel, [varargin] )
%
% INPUTS
%  Is       - cell(N,1) input images
%  regModel - learned multi stage shape regressor (see rcprTrain)
%  varargin - additional params (struct or name/value pairs)
%   .RT1      - [1] number of initial shape restarts
%   .initData - [NxDxRT1] or [Nx2xRT1] initial shape (see shapeGt>initTest)
%   .pInit    - (used only if initData is empty)
%               [Nx4] or [Nx2] initial positions or bounding boxes from
%               which to initialize shape   
%   .regPrm   - [REQ] regression params used during training
%   .prunePrm - parameters for smart restarts 
%      .prune     - [0] whether to use or not smart restarts
%      .maxIter   - [2] number of iterations
%      .th        - [.15] threshold used for pruning 
%      .tIni      - [10] iteration from which to prune
%
% OUTPUTS
%  p        - [NxD] shape returned by multi stage regressor
%  pRT      - [NxDxRT1] shape returned by multi stage regressor
%
% EXAMPLE
%
% See also demoRCPR, FULL_demoRCPR, rcprTrain
% Copyright 2013 X.P. Burgos-Artizzu, P.Perona and Piotr Dollar.  
%  [xpburgos-at-gmail-dot-com]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see bsd.txt]
%
%  Please cite our paper if you use the code:
%  Robust face landmark estimation under occlusion, 
%  X.P. Burgos-Artizzu, P. Perona, P. Dollar (c)
%  ICCV'13, Sydney, Australia

dfs={'pInit',[],'RT1',1,'regPrm','REQ','verbose',1,...
    'initData',[],'prunePrm',struct('prune',0)};



[pIni,RT1,regPrm,verbose,initD,prunePrm] =getPrmDflt(varargin,dfs,1);
if(isempty(initD))
    p=shapeGt('initTest',Is,pIni,regModel,RT1);
else p=initD;clear initD;
end

if( RT1==1 )
    % Run regressor starting from a single shape.
    assert(size(p,3)==1);
    p=rcprTest1(Is,regModel,p,regPrm,pIni,verbose,prunePrm);pRT=p;
else
    % Run regressor starting from mulitple shapes,
    % for each of which RCPR is restarted several times,
    % then find mode of all
    [N,D,rt]=size(p);assert(rt==RT1);
    pRT=rcprTest1(Is,regModel,p,regPrm,pIni,verbose,prunePrm);
    p = median(reshape(pRT,[N,D,RT1]),3);
end
end