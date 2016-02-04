% OPENGL_PATH_PFX and OSG_PATH_PFX should be the path to the OpenGL and OpenSceneGraph libraries,
% e.g. '/usr'
% The libraries are expected at OPENGL_PATH_PFX/lib, OSG_PATH_PFX/lib and the header files at OPENGL_PATH_PFX/include and OSG_PATH_PFX/include
% If you've extracted the pre-compiled libraries in /home/user/download and they are at /home/user/download/local/ 
% compile_renderer('/home/user/download/local/Mesa-7.0.3','/home/user/download/local/');
%
% If only OSG is installed system-wide
% compile_renderer('/home/user/download/local/Mesa-7.0.3','/usr');
%
% If OpenGL and OSG are both installed system-wide
% compile_renderer('/usr','/usr');
function compile_renderer(OPENGL_PATH_PFX,OSG_PATH_PFX)
cmd='mex renderer.cpp depth.cpp Engine.cpp EngineOSG.cpp util/util.cpp -lGL -lX11 -losg -losgViewer -losgDB -losgGA -losgUtil -lOpenThreads -lGLU -Iutil/';
libpath=fullfile(OSG_PATH_PFX,'lib');
lib64path=fullfile(OSG_PATH_PFX,'lib64');
incpath=fullfile(OSG_PATH_PFX,'include');
GLincpath=fullfile(OPENGL_PATH_PFX,'include');
GLlibpath=fullfile(OPENGL_PATH_PFX,'lib64');
cmd=[cmd sprintf(' -L%s -L%s -L%s -I%s -I%s',libpath,lib64path,GLlibpath,incpath,GLincpath)];
fprintf('Executing %s\n',cmd);
eval(cmd);
