function dpc(pauseTime,msg)

if nargin == 0
    pauseTime = 0;
end

drawnow;
if pauseTime==0
    pause
else
    pause(pauseTime);
end
