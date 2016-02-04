
function varargout = AnnotationTool(varargin)
% ANNOTATIONTOOL M-file for AnnotationTool.fig
%
%     This M-file implements the Annotation Tool,
%     an image annotation tool developed at the Department 
%     of Photogrammetry, University of Bonn. Annotation Tool 
%     employs LabelMe MATLAB Toolbox developed at MIT.
% 
%     Related documentation:
%         http://www.ipb.uni-bonn.de/~filip/annotation-tool/
%
%     This program is free software; you can redistribute it and/or 
%     modify it. It is distributed in the hope that it will be useful, 
%     but WITHOUT ANY WARRANTY; without even the implied warranty of 
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%
%     If you use this software for research purposes, please acknowledge
%     its use in resulting publication. We would appreciate if you cite:
%     
%     Filip Korc, David Schneider. Annotation Tool. 
%     Technical report TR-IGG-P-2007-01, University of Bonn, 
%     Department of Photogrammetry, 2007.
%
%     This program was developed by Filip Korc (filip.korc@uni-bonn.de)
%     and David Schneider at the Department of Photogrammetry, 
%     University of Bonn.
%
%
%     ANNOTATIONTOOL, by itself, creates a new ANNOTATIONTOOL or raises 
%     the existing singleton*.
% 
%     H = ANNOTATIONTOOL returns the handle to a new ANNOTATIONTOOL or 
%     the handle to the existing singleton*.
% 
%     ANNOTATIONTOOL('CALLBACK',hObject,eventData,handles,...) calls 
%     the local function named CALLBACK in ANNOTATIONTOOL.M with 
%     the given input arguments.
% 
%     ANNOTATIONTOOL('Property','Value',...) creates a new ANNOTATIONTOOL 
%     or raises the existing singleton*.  Starting from the left, property 
%     value pairs are applied to the GUI before 
%     AnnotationTool_OpeningFunction gets called.  An unrecognized 
%     property name or invalid value makes property application stop.  
%     All inputs are passed to AnnotationTool_OpeningFcn via varargin.
% 
%     *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%     instance to run (singleton)".
%
% Last Modified by GUIDE v2.5 22-Oct-2008 14:10:22

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @AnnotationTool_OpeningFcn, ...
                   'gui_OutputFcn',  @AnnotationTool_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before AnnotationTool is made visible.
function AnnotationTool_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to AnnotationTool (see VARARGIN)
%
%*************************************************************************
%
% Description: This function is used to initialize several objects and will be called at startup1
%
%*************************************************************************


% Choose default command line output for AnnotationTool
handles.output = hObject;

% version number
handles.version = '2.40';

% flag: there are annotations to save
handles.MODIFIED = 0;

% flag: start dialog before saving
handles.SAVEDIALOG = 0;

% flag: in aggregatemode
handles.AGGREGATEMODE = 0;

% flag: in adjustregion
handles.ADJUSTREGIONMODE = 0;
handles.AR_USE_POINT = 0;

% flag: in annotation mode
handles.ANNOTATIONMODE = 0;
handles.AN_POLYGONMODE = 0;
handles.AN_CHECKMOTION = 0;

% flag: in browse classes mode:
handles.BROWSECLASSES = 0;

% per cent of the tolerance for adjustregion
handles.AR_TOLERANCE = 0.1;

% image and mn_annotation data paths
handles.HOMEIMAGES = deblank(readtextfile('home-images.txt'));
handles.HOMEANNOTATIONS = deblank(readtextfile('home-annotations.txt'));

% (lbx_parts) array of the mn_annotation-objects's ids from lbx_objects
handles.LBX_PARTS_NUM = [];

% zoom factor for annotate mode, 0 < x < 1
handles.ZOOM_FACTOR = 0.7;

% zoom mode
handles.ZOOMMODE = 0;

% ------------------------------------
% NEW GUI OBJECTS: add here and in 'enable_objects' and modify 'update_uicontrols'
% GuiIDs:

i = 1;
handles.GuiID.MIN = i;

% Mnu File:
handles.GuiID.mn_file			= i; i = i+1;
handles.GuiID.mn_open			= i; i = i+1;
handles.GuiID.mn_save			= i; i = i+1;
handles.GuiID.mn_exit			= i; i = i+1;

% Mnu Image:
handles.GuiID.mn_rectification	= i; i = i+1;
handles.GuiID.mn_zoom			= i; i = i+1;
handles.GuiID.mn_annotate		= i; i = i+1;
handles.GuiID.mn_setscale		= i; i = i+1;
handles.GuiID.mn_rectify		= i; i = i+1;
%handles.GuiID.mn_load			= i; i = i+1;
handles.GuiID.mn_defAnnoClasses	= i; i = i+1;
handles.GuiID.mn_browseclasses = i; i = i+1;

% Mnu Annotation:
handles.GuiID.mn_annotation		= i; i = i+1;
handles.GuiID.mn_adjustregion	= i; i = i+1;
handles.GuiID.mn_editsource		= i; i = i+1;
handles.GuiID.mn_objectnote		= i; i = i+1;
handles.GuiID.mn_delete			= i; i = i+1;
handles.GuiID.mn_sort1			= i; i = i+1;

% Mnu Aggregation:
handles.GuiID.mn_aggregation		= i; i = i+1;
handles.GuiID.mn_set_aggregate		= i; i = i+1;
handles.GuiID.mn_finish_aggregation	= i; i = i+1;
handles.GuiID.mn_add_part			= i; i = i+1;
handles.GuiID.mn_add_part_auto      = i; i = i+1; % MD
handles.GuiID.mn_delete_part		= i; i = i+1;
handles.GuiID.mn_show_parts			= i; i = i+1;

% Mnu Dataset:
handles.GuiID.mn_dataset				= i; i = i+1;
handles.GuiID.mn_defDsetAnnoClasses		= i; i = i+1;

% UIControls ListBoxes:
handles.GuiID.lbx_filenames			= i; i = i+1;
handles.GuiID.lbx_objects			= i; i = i+1;
handles.GuiID.lbx_parts				= i; i = i+1;

% UIControls ImageData:
handles.GuiID.popup_image_source	= i; i = i+1;
handles.GuiID.popup_view_type		= i; i = i+1;
handles.GuiID.btn_scale				= i; i = i+1;

% UIControls Annotate Btns:
handles.GuiID.btn_annotate			= i; i = i+1;
handles.GuiID.btn_add				= i; i = i+1;

% UIControls Annotate PopUpMn:
handles.GuiID.popup_object_name				= i; i = i+1;
handles.GuiID.popupmenu_occlusion			= i; i = i+1;
handles.GuiID.popupmenu_representativeness	= i; i = i+1;
handles.GuiID.popupmenu_uncertainty			= i; i = i+1;

handles.GuiID.MAX = i -1;

% END: GuiIDs:
% ------------------------------------

% sets the new position and size (space defines the space to the bottom, remind toolbar)
space = 26;
units = get(handles.fig_annotation_tool,'Units');
res=get(0,'ScreenSize');
set(handles.fig_annotation_tool,'units','pixels','outerposition',[0 space res(3) res(4)-space]);
set(handles.fig_annotation_tool,'Units', units);

% define colors
handles.colors = 'rgbcmyw';

% initialize UI
disables = ones(handles.GuiID.MAX, 1);
disables(handles.GuiID.mn_file) = 0;
disables(handles.GuiID.mn_open) = 0;
disables(handles.GuiID.mn_exit) = 0;
enable_objects(disables, handles, 'off');


% define annotation structure
handles.DTDORDER_ANNOTATION = ...
	{'filename'
	'folder'
	'sourceImage'
	'sourceAnnotationXML'
	'rectified'
	'viewType'
	'scale'
	'imageWidth'
	'imageHeight'
	'transformationMatrix'
	'annotatedClasses'
	'object'};

	
% define annotation object structure
handles.DTDORDER_OBJECT = ...
	{'name'
	'objectID'
	'occlusion'
	'representativeness'
	'uncertainty'
	'deleted'
	'verified'
	'date'
	'sourceAnnotation'
	'polygon'
	'objectParts'
	'comment'};

% define the dtd filename
handles.DTDFILENAME = 'annotation-xml.dtd';

% parameter call
if prod(size(varargin)) == 2
	% DirectLoad(handles, imgPath, filename)
	handles = ParameterCall(hObject, eventdata, ...
		handles, varargin{1}, varargin{2});
end;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes AnnotationTool wait for user response (see UIRESUME)
% uiwait(handles.fig_mn_annotation_tool);

% --- Outputs from this function are returned to the command line.
function varargout = AnnotationTool_OutputFcn(hObject, eventdata, handles) 
% varar
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes during object creation, after setting all properties.
function lbx_filenames_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lbx_filenames (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

function lbx_objects_Callback(hObject, eventdata, handles)
% --- Executes on selection change in lbx_objects.
% hObject    handle to lbx_objects (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
% Hints: contents = get(hObject,'String') returns lbx_objects contents as cell array
%        contents{get(hObject,'Value')} returns selected item from lbx_objects
%
%*************************************************************************
%
% Description: This function will be called by selecting an object in the object listbox. It plots the graph and its parts (depending on mn_show_parts).
%
%*************************************************************************


i = get(hObject,'Value');
%contents = get(hObject,'String')
%i = contents{get(hObject,'Value')}
%class{n} = handles.mn_annotation.object(i).name; % get object name

% still in AnnotateMode?
if (strcmp(get(handles.btn_add, 'Enable'), 'on') && ...
	~strcmp(get(handles.btn_annotate, 'String'), 'New Annotation'))

	handles.currObject = '';
	set(handles.btn_add, 'Enable', 'off');
end;

handles = update_panel_annotation(handles, i);

cla;
imshow(handles.Image);
axis('equal'), axis('tight'), axis on, hold on

if ~handles.BROWSECLASSES
	[X,Y] = getLMpolygon(handles.annotation.object(i).polygon);
	plot([X; X(1)],[Y; Y(1)], 'LineWidth', 4, 'color', [0 0 0]);
	plot([X; X(1)],[Y; Y(1)], 'LineWidth', 2, 'color', ...
	          [rand, rand, rand] );
	%handles.colors(mod(sum(double(handles.class{i})),7)+1)
end;

% require for "lbx_parts identification problem" (if not aggregate mode)
if (handles.AGGREGATEMODE == 0)
	handles.LBX_PARTS_NUM = [];
end;


if handles.BROWSECLASSES
	
	class = get(hObject, 'String');
	class = class{i};
	
	
	color = [rand, rand, rand];
	
	for k = 1:length(handles.annotation.object)
		
		if ~strcmp(class, handles.annotation.object(k).name)
			continue;
		end;
		
		axis('equal'), axis('tight'), axis on, hold on
		[X,Y] = getLMpolygon(handles.annotation.object(k).polygon);
		plot([X; X(1)],[Y; Y(1)], 'LineWidth', 4, 'color', [0 0 0]);
		plot([X; X(1)],[Y; Y(1)], 'LineWidth', 2, 'color', ...
			color);
	end;		
	
else
	
	% exists(objectParts) && ~exist(currObject) 
	% OR
	% exists(objectParts) && exist(currObject) && empty(currObject)
	if ( isfield(handles.annotation.object(i), 'objectParts') && ...
	        ~isfield(handles, 'currObject') ) || ...
	        ( isfield(handles.annotation.object(i), 'objectParts') && ...
	        isfield(handles, 'currObject') && ...
	        isempty(handles.currObject) ),
	    % read IDs in a numeric matrix
	    IDs = str2num(str2mat(handles.annotation.object(i).objectParts)); %#ok<ST2NM>
		
	    % check for empty matrix
	    if ~isempty(IDs),
	        contents_parts = '';
			
			checkIDs = [];
			
	        % list objects with IDs in <objectParts> in lbx_parts
	        for idx_id = 1:length(IDs)
	           for idx_obj = 1:length(handles.annotation.object),
	               
	              % stored ID equals current object ID
	              if str2double(handles.annotation.object(idx_obj).objectID) ...
	                      == IDs(idx_id),
	                  contents_parts{end+1} = ...
	                      handles.annotation.object(idx_obj).name;
	                 
	                      % require for "lbx_parts identification problem"
	                      handles.LBX_PARTS_NUM = [handles.LBX_PARTS_NUM, idx_obj];
							
	                    % add object's parts if mn_show_parts is checked
	                      if strcmp(get(handles.mn_show_parts,'Checked'), 'on')
	                          axis('equal'), axis('tight'), axis on, hold on
	                          [X,Y] = getLMpolygon(handles.annotation.object(idx_obj).polygon);
	                          plot([X; X(1)],[Y; Y(1)], 'LineWidth', 4, 'color', [0 0 0]);
	                          plot([X; X(1)],[Y; Y(1)], 'LineWidth', 2, 'color', ...
	                          [rand, rand, rand] );
	                      end
	 
						checkIDs = [checkIDs; IDs(idx_id)];
					end
	              
	           end
	        end 
			
			% check object parts
			if prod(size(IDs)) ~= prod(size(checkIDs))
			
				handles.annotation.object(i).objectParts = mat2str(checkIDs);
				% set flag
				handles.MODIFIED = 1;
			end;
	        
	        % update listbox names
	        set(handles.lbx_parts,'Value',1);
	        set(handles.lbx_parts,'String',contents_parts);
	    else
	        set(handles.lbx_parts,'String','')
	    end
	else
	    % ~exist(currObject) 
	    % OR
	    % exist(currObject) && empty(currObject)
	    if ~isfield(handles, 'currObject') || ...
	            (isfield(handles, 'currObject') && ...
	        isempty(handles.currObject) ),
	    
	        set(handles.lbx_parts,'String','')
	        
	    end
	end

	% update ID text
	if (length(handles.annotation.object) > 0)
		i = get(handles.lbx_objects, 'Value');
		set(handles.txt_objID, 'String', strcat('Current ID:  ', num2str(handles.annotation.object(i).objectID)));
	end;
end; % if handles.BROWSECLASSES


% update UI Controls
update_uicontrols(handles);

% Sichern der handles-struktur
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function lbx_objects_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lbx_objects (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor', ...
        get(0,'defaultUicontrolBackgroundColor'));
end


% --- Executes on button press in btn_add.
function btn_add_Callback(hObject, eventdata, handles)
% hObject    handle to btn_add (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: Depending on the mode this button adds a new object to the object list (annotate mode) or save changes (view attribute mode).
%
%*************************************************************************


cont_annot = get(handles.popup_object_name,'String');
cont_occlusion = get(handles.popupmenu_occlusion,'String');
cont_representativeness = get(handles.popupmenu_representativeness,'String');
cont_uncertainty = get(handles.popupmenu_uncertainty,'String');

insert = 0;

if ( strcmp( cont_annot{ ...
	        get(handles.popup_object_name,'Value') }, '' ) == 0)% && ...
	        % ( strcmp( cont_occlusion{ ...
	        % get(handles.popupmenu_occlusion,'Value')},'')==0) && ...
			% ( strcmp( cont_representativeness{ ...
	        % get(handles.popupmenu_representativeness,'Value')},'')==0) && ...
	        % (strcmp( cont_uncertainty{ ...
	        % get(handles.popupmenu_uncertainty,'Value')},'')==0)

	% TEST: annotate mode OR view attribute mode
	if (strcmp(get(handles.btn_add, 'String'), 'Save Object'))
	% view attribute mode

		i = get(handles.lbx_objects, 'Value');


		% object name
	    handles.annotation.object(i).name = handles.objname;

	    % objectID composed from date+time to within miliseconds precision
	        
	    % degree of occlusion
	    handles.annotation.object(i).occlusion = ...
	        (cont_occlusion{get(handles.popupmenu_occlusion,'Value')});

	    % representativeness of the annotated object with respect to the object
	    % category
	    if ~isempty( cont_representativeness{ ...
	            get(handles.popupmenu_representativeness,'Value')}),
	        handles.annotation.object(i).representativeness = ...
	            (cont_representativeness{ ...
	            get(handles.popupmenu_representativeness,'Value') });
	    else
	        handles.annotation.object(1,i).representativeness = 'n/a';
	    end

	    % uncertainty (accuracy) of the mn_annotation [pixels]
	    if ~isempty( cont_uncertainty{ ...
	            get(handles.popupmenu_uncertainty,'Value')}),
	        % uncertainty (accuracy) of the mn_annotation [pixels]
	        handles.annotation.object(i).uncertainty = ...
	            (cont_uncertainty{ get(handles.popupmenu_uncertainty,'Value')});
	    else
	        handles.annotation.object(1,i).uncertainty = 'n/a';
	    end

	    % aux. tags
	    handles.annotation.object(i).date = date;

		handles.class{i} = handles.annotation.object(i).name;

		set(handles.lbx_objects, 'String', handles.class);
		

		reset_panel_annotation(handles);
		
		% set flag
		handles.MODIFIED = 1;
		insert = 1;

	else
	% annotate mode

		if ~isfield(handles,'login')
			% annotation login
			%prompt={'Enter your name:','Enter the place of annotation:'};
			prompt={'Enter your name:'};
			name='Annotation Login';
			numlines=1;
			defaultanswer={'Zimmermann@uni-bonn'};
			cell_aux = inputdlg(prompt,name,numlines,defaultanswer);

			% check the login
			if ~isempty(cell_aux) && ~strcmp(cell_aux{1},''),
				handles.login = cell_aux{1};
			end
		end
			        
		if isfield(handles,'login')
		
			if isfield(handles.annotation,'object')
				i = length(handles.annotation.object);
			else            
				i=0;
				% handles.annotation.object = [1];
			end;

			% object name
			handles.annotation.object(1,i+1).name = handles.currObject.name;

			% objectID composed from date+time to within miliseconds precision
			handles.annotation.object(1,i+1).objectID = ...
				num2str(round(now*1e8));
			
			% degree of occlusion
			if ~isempty( cont_occlusion{ ...
					get(handles.popupmenu_occlusion,'Value')}),
				
				handles.annotation.object(1,i+1).occlusion = ...
				cont_occlusion{get(handles.popupmenu_occlusion,'Value')};
			else
				handles.annotation.object(1,i+1).occlusion = ...
					'n/a';
			end
			
			% representativeness of the annotated object with respect to the object
			% category
			if ~isempty( cont_representativeness{ ...
					get(handles.popupmenu_representativeness,'Value')}),
				
				handles.annotation.object(1,i+1).representativeness = ...
					cont_representativeness{ ...
					get(handles.popupmenu_representativeness,'Value') };
			else
				handles.annotation.object(1,i+1).representativeness = ...
					'n/a';
			end

			% uncertainty (accuracy) of the annotation [pixels]
			if ~isempty( cont_uncertainty{ ...
					get(handles.popupmenu_uncertainty,'Value')}),
				
				handles.annotation.object(1,i+1).uncertainty = ...
					cont_uncertainty{ ...
					get(handles.popupmenu_uncertainty,'Value')};
			else
				handles.annotation.object(1,i+1).uncertainty = 'n/a';
			end

			% aux. tags
			handles.annotation.object(1,i+1).deleted = 0;
			handles.annotation.object(1,i+1).verified = 0;
			handles.annotation.object(1,i+1).date = date;
			%handles.annotation.object(1,i+1).polygon.username = 'AnnotationTool';
			handles.annotation.object(1,i+1).sourceAnnotation = handles.login;
			
			% polygon coordinates
			for l=1:size(handles.currObject.xy,2),
				handles.annotation.object(1,i+1).polygon.pt(1,l).x = ...
					num2str(handles.currObject.xy(1,l));
				
				handles.annotation.object(1,i+1).polygon.pt(1,l).y = ...
					num2str(handles.currObject.xy(2,l));
			end
			
			
			% initialize object parts tag
			handles.annotation.object(1,i+1).objectParts = 'n/a';
			
			% initialize object parts tag
			%handles.mn_annotation.object(1,i+1).polygonBlob = 'n/a';
		   
			clear handles.currObject

			handles.class{end+1} = handles.annotation.object(1,i+1).name;
			set(handles.lbx_objects, 'Value', i+1);
			set(handles.lbx_objects, 'Enable', 'on');
			set(handles.lbx_objects, 'String', handles.class);

			% clear the popup menus
			set(handles.popup_object_name,'Value',1);
			set(handles.popupmenu_occlusion,'Value',1);
			set(handles.popupmenu_representativeness,'Value',1);
			set(handles.popupmenu_uncertainty,'Value',1);
				
			% FLAGS:
			% annotations to save; enable save btn
			handles.MODIFIED = 1;

			% finished closed polygon
			handles.currObject = '';
				
			% disable objects
			set(handles.popup_object_name,'Enable','off');
			set(handles.popupmenu_occlusion,'Enable','off');
			set(handles.popupmenu_representativeness,'Enable','off');
			set(handles.popupmenu_uncertainty,'Enable','off');
			set(handles.btn_add,'Enable','off');
			
			% plot oll objects and save the view			
			XLim = get(gca,'XLim');
			YLim = get(gca,'YLim');
			
			[h, handles.class] = LMplot(handles.annotation, handles.Image);
			
			set(gca,'XLim', XLim);
			set(gca,'YLim', YLim);
		
		else if ~isempty(cell_aux)
			msgbox('Please fill in the login information.', 'Login')
		end;
		end;
	end;
	
	insert = 1;
else
    msgbox('Please fill in the required information.')
end;


if (insert == 1)
	% check handles.annotation.annotatedClasses for annotated object class 
	
	name = handles.annotation.object(end).name;
    
    % make sure className is a cell array (className is not a cell
    % array if there is a single className in annotatedClasses)
    if ~iscell(handles.annotation.annotatedClasses.className),
        handles.annotation.annotatedClasses.className = ...
            {handles.annotation.annotatedClasses.className};
    end        
	
	newClass = 1;
	for i = 1:length(handles.annotation.annotatedClasses.className)
		if strcmp(name, handles.annotation.annotatedClasses.className{i})
			newClass = 0;
			break;
		end;
	end;
	
	if newClass
		handles.annotation.annotatedClasses.className{end +1} = name;
		handles.annotation.annotatedClasses.className = ...
			sort(handles.annotation.annotatedClasses.className);
		
		set(handles.popup_object_name, 'String', ...
			[{''} handles.annotation.annotatedClasses.className {'other...'}]);
	end;
end;


% update UI Controls
update_uicontrols(handles);

% update handles-structure
guidata(hObject, handles)


function popup_object_name_Callback(hObject, eventdata, handles)
% hObject    handle to popup_object_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of popup_object_name as text
%        str2double(get(hObject,'String')) returns contents of popup_object_name as a double

content = get(handles.popup_object_name,'String');
 if ~strcmp( content{end}, ...
        content{get(handles.popup_object_name,'Value')} );
		
    handles.currObject.name = ...
        content{get(handles.popup_object_name,'Value')};
		
 else
    cell_aux = inputdlg('Object Class: ','New Annotation');
    
    % check the returned string
    if ~isempty(cell_aux) && ~strcmp(cell_aux{1},''),
        handles.currObject.name = cell_aux{1};
    else
        set(handles.popup_object_name,'Value',1);
    end
	
end

if (isfield(handles.currObject, 'name') && isstr(handles.currObject.name))
    handles.objname = handles.currObject.name;
end;

set(handles.btn_add, 'Enable', 'on');

% update handles
guidata(hObject, handles) 

% --- Executes during object creation, after setting all properties.
function popup_object_name_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popup_object_name (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

function popup_image_source_Callback(hObject, eventdata, handles)
% hObject    handle to popup_image_source (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
% Hints: get(hObject,'String') returns contents of popup_image_source as text
%        str2double(get(hObject,'String')) returns contents of popup_image_source as a double
%
%*************************************************************************
%
% Description: This function opens a dialog to receive an image source. The source will be saved in handles.annotation.sourceImage.
%
%*************************************************************************


content = get(handles.popup_image_source,'String');
if ~strcmp( content{end}, ...
        content{get(handles.popup_image_source,'Value')} );

    handles.annotation.sourceImage = ...
        content{get(handles.popup_image_source,'Value')};
    handles.MODIFIED = 1;
    set(handles.mn_save,'Enable','on');

else
    cell_aux = inputdlg('Image source:','Current Image');
    
    % check the returned string
    if ~isempty(cell_aux) && ~strcmp(cell_aux{1},''),
        handles.annotation.sourceImage = cell_aux{1};
        handles.MODIFIED = 1;
        set(handles.mn_save,'Enable','on');
    else
        handles.annotation.sourceImage = '';
        set(handles.popup_image_source,'Value',1);
        if ~isempty(cell_aux)
            handles.MODIFIED = 1;
            set(handles.mn_save,'Enable','on');
        end
    end    
end

if strcmp(handles.annotation.sourceImage, '')
	handles.annotation.sourceImage = 'n/a';
end;

% update handles
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function popup_image_source_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popup_image_source (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% --- Executes on button press in savebutton.
function handles = save_annotation(handles)
% hObject    handle to savebutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: This function saves the data struct (handles.annotation).
%
%*************************************************************************


struct_xml.annotation = handles.annotation;
if ~exist([handles.HOMEANNOTATIONS,'/', handles.directory])

    mkdir( handles.HOMEANNOTATIONS, handles.directory );
end


% check obj (in order to support the dtd, delete empty object field)
% and check object order
if isfield(struct_xml.annotation, 'object') && ...
	length(struct_xml.annotation.object) == 0
	
	struct_xml.annotation = rmfield(struct_xml.annotation, 'object');
else
	
	if isfield(struct_xml.annotation, 'object') && ...
		length(struct_xml.annotation.object) > 0
		
		for i = 1:length(handles.DTDORDER_OBJECT)
			if ~isfield(struct_xml.annotation.object(1), ...
				handles.DTDORDER_OBJECT{i})
				
				struct_xml.annotation.object(1).(handles.DTDORDER_OBJECT{i}) = '';
			end;
		end;
		
		struct_xml.annotation.object = ...
			orderfields(struct_xml.annotation.object, ...
			handles.DTDORDER_OBJECT);
	end;
end;


% check rectify
if ~strcmp(struct_xml.annotation.transformationMatrix, 'n/a')
	
	filename = ...
		handles.filenames{get(handles.lbx_filenames , 'value')};
	
	% get filename
	[pathstr, name ] = fileparts(handles.IMAGEfilename);
	
	% check get xml name
	if length(name) >= length('_rect') && ...
		strcmp(name(end-length('_rect') +1:end), '_rect')
		
		
		xmlfile = [name(1:end-length('_rect')) '.xml'];
	else
		xmlfile = [name '_rect' '.xml'];
	end;
	
	xmlfile = [handles.HOMEANNOTATIONS,'/',...
        handles.directory,'/', xmlfile];
	
	if exist(xmlfile)
		
		if isfield(struct_xml.annotation, 'object')
			
			struct_rect = loadXML(xmlfile);
			
            struct_rect.annotation.transformationMatrix = ...
                struct_xml.annotation.transformationMatrix;
            
			struct_rect.annotation.object = struct_xml.annotation.object;
			if isfield(struct_xml.annotation.annotatedClasses, 'className')
				struct_rect.annotation.annotatedClasses.className = ...
					struct_xml.annotation.annotatedClasses.className;
			end;
			
			struct_rect.annotation = transform_labels(struct_rect);
			
			writeXML(xmlfile, struct_rect);
		end;
		
		
		clear('struct_rect');
	end;
	
end;


% save mn_annotation
writeXML([handles.HOMEANNOTATIONS,'/',...
    handles.directory,'/',...
    handles.XMLfilename],...
    struct_xml);
clear('struct_xml');

% disable btn
set(handles.mn_save,'Enable','off');

% flag: annotations to save
handles.MODIFIED = 0;
        
% --- Executes on selection change in popupmenu_occlusion.
function popupmenu_occlusion_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu_occlusion (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns popupmenu_occlusion contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu_occlusion

set(handles.btn_add, 'Enable', 'on');

% --- Executes during object creation, after setting all properties.
function popupmenu_occlusion_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu_occlusion (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end


% --- Executes on selection change in popupmenu_representativeness.
function popupmenu_representativeness_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu_representativeness (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns popupmenu_representativeness contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu_representativeness

set(handles.btn_add, 'Enable', 'on');

% --- Executes during object creation, after setting all properties.
function popupmenu_representativeness_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu_representativeness (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end


% --- Executes on selection change in popupmenu_uncertainty.
function popupmenu_uncertainty_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu_uncertainty (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
% Hints: contents = get(hObject,'String') returns popupmenu_uncertainty contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu_uncertainty
%
%*************************************************************************
%
% Description: This function draws a circle to visualize the uncertainty (see plot_circle).
%
%*************************************************************************


% uncertainty (accuracy) of the mn_annotation [pixels]
contents = get(handles.popupmenu_uncertainty,'String'); hold on

% TEST: annotate mode OR view attribute mode
if strcmp(get(handles.btn_add, 'String'), 'Save Object'),
% view attribute mode

    obj_index = get(handles.lbx_objects,'Value');
    x = zeros(length(handles.annotation.object(obj_index).polygon.pt),1);
    y = zeros(length(handles.annotation.object(obj_index).polygon.pt),1);
    
    for i=1:length(handles.annotation.object(obj_index).polygon.pt), 
        x(i) = str2double( ...
            handles.annotation.object(obj_index).polygon.pt(i).x);
    end
    
    for i=1:length(handles.annotation.object(obj_index).polygon.pt), 
        y(i) = str2double( ...
            handles.annotation.object(obj_index).polygon.pt(i).y);
    end
    % uncertainty tool offset
    x_offset = mean(x);
    y_offset = mean(y);


else
    % uncertainty tool offset
    x_offset = sum(handles.currObject.xy(1,:))/size(handles.currObject.xy,2);
    y_offset = sum(handles.currObject.xy(2,:))/size(handles.currObject.xy,2);
end

for i = 2:size(contents),
    plot_circle( [x_offset; y_offset], str2double(contents{i}), 'g')
end
plot_circle( [x_offset; y_offset], ...
    str2double(contents{get(handles.popupmenu_uncertainty,'Value')}), ...
    'r')
	
set(handles.btn_add, 'Enable', 'on');

% --- Executes during object creation, after setting all properties.
function popupmenu_uncertainty_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu_uncertainty (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% --- aux. function that plots a circle of a given color
function plot_circle(OFFSET, SIZE, color)
%*************************************************************************
%
% Input: OFFSET Two dimensional offset.
% Input: SIZE Size of the circle.
% Input: color Color of the circle.
%
% Description: This functions draws a circle.
%
%*************************************************************************



SIZE = SIZE/2;
%OFFSET = 51;

t = 0:0.01:2*pi;

x = cos(t); 
y = sin(t);

x = x*SIZE+OFFSET(1); 
y = y*SIZE+OFFSET(2);

plot(x,y, color)


% --- Executes during object creation, after setting all properties.
function fig_annotation_tool_CreateFcn(hObject, eventdata, handles)
% hObject    handle to fig_mn_annotation_tool (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in btn_annotate.
function btn_annotate_Callback(hObject, eventdata, handles)
% hObject    handle to btn_annotate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: This function enables the annotation mode, which allows the user to draw polygons.
%
%*************************************************************************

handles = startAnnotateMode(handles);

% update UI Controls
update_uicontrols(handles);

% Update handles structure
guidata(hObject, handles);

function handles = startAnnotateMode(handles)
%*************************************************************************
%
% Input: handles
% Output: handles
%
% Description: starts the annotate mode
%
%*************************************************************************

reset_panel_annotation(handles);

% quit browse classes mode
if handles.BROWSECLASSES
	handles = EndBrowseClassesMode(handles);
end;

if (handles.ANNOTATIONMODE == 0)
	
	% draw polygons
	cla, imshow(handles.Image); axis('off'), hold on
	
	% init values
	handles.an_poly.XY = [];
	handles.an_poly.n = 0;
	handles.anColor = [rand, rand, rand];
	
	% switch modes
	handles.ANNOTATIONMODE = 1;
	handles.AN_CHECKMOTION = 0;
	handles.ZOOMMODE = 1;
end;



% --- reads a text file
function tab = readtextfile(filename)
%*************************************************************************
%
% Input: filename Name of the file.
% Output: tab Read data.
%
% Description: Read a text file into a matrix with one row per input line...
%			and with a fixed number of columns, set by the longest line.
%			Each string is padded with NUL (ASCII 0) characters
%
%*************************************************************************

ip = fopen(filename,'rt');          % 'rt' means read text
if (ip < 0)
    error('could not open file');   % just abort if error 
end;

% find length of longest line
max=0;                              % record length of longest string
cnt=0;                              % record number of strings
s = fgetl(ip);                      % get a line

while (ischar(s))                   % while not end of file
   cnt = cnt+1;
   if (length(s) > max)           % keep record of longest
       max = length(s);
   end;
   s = fgetl(ip);                  % get next line
end;

% rewind the file to the beginning
frewind(ip);

% create an empty matrix of appropriate size
tab=char(zeros(cnt,max));           % fill with ASCII zeros

% load the strings for real
cnt=0;
s = fgetl(ip);

while (ischar(s))
   cnt = cnt+1;
   tab(cnt,1:length(s)) = s;      % slot into table
    s = fgetl(ip);
end;

% close the file and return
fclose(ip);
return;


% --- 
function [h, class] = my_plot(annotation, img)
%*************************************************************************
%
% Input: annotation The annotation structure with the data to plot.
% Input: img The background image.
%
% Output: h
% Output: class
%
% Description: Visualizes the polygons in an image.
%
%*************************************************************************


colors = 'rgbcmyw';

% Draw image
imshow(img); axis('off'); hold on
% Draw each object (only non deleted ones)
h = []; class = [];
if isfield(annotation, 'object')
    Nobjects = length(annotation.object); n=0;
    for i = 1:Nobjects
        n = n+1;
        class{n} = annotation.object(i).name; % get object name
        [X,Y] = getLMpolygon(annotation.object(i).polygon);

        if isfield(annotation.object(i), 'confidence')
            LineWidth = round(annotation.object(i).confidence)
            LineWidth = min(max([2 LineWidth]), 8);
        else
            LineWidth = 4;
        end
        
        plot([X; X(1)],[Y; Y(1)], 'LineWidth', LineWidth, 'color', [0 0 0]); hold on
        h(n) = plot([X; X(1)],[Y; Y(1)], 'LineWidth', LineWidth/2, 'color', colors(mod(sum(double(class{n})),7)+1));
        hold on
    end

    if nargout == 0
        legend(h, class);
    end
end

axis on


% --- 
function enable_objects( v, handles, opt)
%*************************************************************************
%
% Input: v 0/1 Vector with the objects, which shall be dis-/ enable
% Input: handles The handles structure.
% Input: opt Defines, whether the Objects should be shown or not ('on' or 'off')
%
% Description: This function dis-/enables ui controls specified in the vector v. opt defines the state ('on' or 'off).
%
%*************************************************************************

% NEW GUI OBJECTS: add here and in 'AnnotationTool_OpeningFcn' and modify 'update_uicontrols'

% aux index
i = 1;

% Mnu File:
if v(i), set(handles.mn_file,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_open,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_save,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_exit,'Enable',opt); end; i = i+1;

% Mnu Image:
if v(i), set(handles.mn_rectification,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_zoom,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_annotate,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_setscale,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_rectify,'Enable',opt); end; i = i+1;
%if v(i), set(handles.mn_load,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_defAnnoClasses,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_browseclasses,'Enable',opt); end; i = i+1;

% Mnu Annotation:
if v(i), set(handles.mn_annotation,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_adjustregion,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_editsource,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_objectnote,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_delete,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_sort1,'Enable',opt); end; i = i+1;

% Mnu Aggregation:
if v(i), set(handles.mn_aggregation,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_set_aggregate,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_finish_aggregation,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_add_part,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_add_part_auto,'Enable',opt); end; i = i+1; % MD
if v(i), set(handles.mn_delete_part,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_show_parts,'Enable',opt); end; i = i+1;

% Mnu Dataset:
if v(i), set(handles.mn_dataset,'Enable',opt); end; i = i+1;
if v(i), set(handles.mn_defDsetAnnoClasses,'Enable',opt); end; i = i+1;

% UIControls ListBoxes:
if v(i), set(handles.lbx_filenames,'Enable',opt); end; i = i+1;
if v(i), set(handles.lbx_objects,'Enable',opt); end; i = i+1;
if v(i), set(handles.lbx_parts,'Enable',opt); end; i = i+1;

% UIControls ImageData:
if v(i), set(handles.popup_image_source,'Enable',opt); end; i = i+1;
if v(i), set(handles.popup_view_type,'Enable',opt); end; i = i+1;
if v(i), set(handles.btn_scale,'Enable',opt); end; i = i+1;

% UIControls Annotate Btns:
if v(i), set(handles.btn_annotate,'Enable',opt); end; i = i+1;
if v(i), set(handles.btn_add,'Enable',opt); end; i = i+1;

% UIControls Annotate PopUpMn:
if v(i), set(handles.popup_object_name,'Enable',opt); end; i = i+1;
if v(i), set(handles.popupmenu_occlusion,'Enable',opt); end; i = i+1;
if v(i), set(handles.popupmenu_representativeness,'Enable',opt); end; i = i+1;
if v(i), set(handles.popupmenu_uncertainty,'Enable',opt); end; i = i+1;



% --- Executes on selection change in popup_view_type.
function popup_view_type_Callback(hObject, eventdata, handles)
% hObject    handle to popup_view_type (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns popup_view_type contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popup_view_type

content = get(handles.popup_view_type,'String');
if ~strcmp( content{end}, ...
        content{get(handles.popup_view_type,'Value')} );

    handles.annotation.viewType = ...
        content{get(handles.popup_view_type,'Value')};
    handles.MODIFIED = 1;
    set(handles.mn_save,'Enable','on');

else
    cell_aux = inputdlg('View Type:','Current Image');
    
    % check the returned string
    if ~isempty(cell_aux) && ~strcmp(cell_aux{1},''),
        handles.annotation.viewType = cell_aux{1};
        handles.MODIFIED = 1;
        set(handles.mn_save,'Enable','on');
    else
        handles.annotation.viewType = '';
        set(handles.popup_view_type,'Value',1);
        if ~isempty(cell_aux)
            handles.MODIFIED = 1;
            set(handles.mn_save,'Enable','on');
        end
    end    
end

if strcmp(handles.annotation.viewType, '')
	handles.annotation.viewType = 'n/a';
end;

% update handles
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function popup_view_type_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popup_view_type (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc
    set(hObject,'BackgroundColor','white');
else
    set(hObject,'BackgroundColor',get(0,'defaultUicontrolBackgroundColor'));
end

% update handles
guidata(hObject, handles);


% --- Executes on button press in btn_scale.
function btn_scale_Callback(hObject, eventdata, handles)
% hObject    handle to btn_scale (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: This function enables to set an scale value.
%
%*************************************************************************


handles = setscale(handles);

% Update handles structure
guidata(hObject, handles);


function handles = setscale(handles)
%*************************************************************************
%
% Input: handles
% Output: handles
%
% Description: handles set scale event
%
%*************************************************************************

% object height in [pel]
[X,Y] = ginput(2);

% check x,y values
imgSize = size(handles.Image);
for i = 1:length(X)
	if (X(i) < 1)
		X(i) = 1;
	else if (X(i) > imgSize(2))
		X(i) = imgSize(2);
		end;
	end;
	
	if (Y(i) < 1)
		Y(i) = 1;
	else if (Y(i) > imgSize(1))
		Y(i) = imgSize(1);
		end;
	end;
end;
% end check x,y values	

% draw line
cla, imshow(handles.Image); axis('equal'), axis('tight'), axis on, hold on
hold on;

plot(X,Y, 'LineWidth', 4, 'color', [0 0 0]);
plot(X,Y, 'LineWidth', 2, 'color', [rand rand rand]);
% end line

X = X(1) - X(2);
Y = Y(1) - Y(2);

% object height in [cm]
height = inputdlg('Object height [cm]:','Scale Estimation');

if prod(size(height)),
    if ~strcmp(height{1},'') && ~strcmp(height{1},'0'),
    
        % scale estimate
        handles.annotation.scale = num2str(round(norm([X Y])) / str2double(height));
		
        set(handles.text_scale,'String', ...
            ['Scale [pel/cm]: ', num2str(str2num(handles.annotation.scale), '%.3f') ] );

        handles.MODIFIED = 1;
        set(handles.mn_save,'Enable','on');

    end
end


% --- Reads & displays the image information
function handles = read_disp_img_info(hObject, handles)
% hObject    handle to btn_add (see GCBO)
% handles    structure with handles and user data (see GUIDATA)

% image source
if isfield(handles.annotation, 'sourceImage') && ...
	~strcmp(handles.annotation.sourceImage, '') && ...
	~strcmp(handles.annotation.sourceImage, 'n/a')
    s = get(handles.popup_image_source, 'String');
    r=[];
    for i = 1 : size(s,1), 
        r = [ r; strcmp( handles.annotation.sourceImage, s{i,1} ) ]; 
    end
    if find(r)
        set(handles.popup_image_source,'Value',find(r));
    else
        set(handles.popup_image_source,'Value',size(s,1));
    end
else
    handles.annotation.sourceImage = 'n/a';
	set(handles.popup_image_source,'Value',1);    
end

% view type
if isfield(handles.annotation, 'viewType') && ...
	~strcmp(handles.annotation.viewType, '') && ...
	~strcmp(handles.annotation.viewType, 'n/a')
	
    s = get(handles.popup_view_type, 'String');
    r=[];
    for i = 1 : size(s,1),
        r = [ r; strcmp( handles.annotation.viewType, s{i,1} ) ]; 
    end
    if find(r)
        set(handles.popup_view_type,'Value',find(r));
    else
        set(handles.popup_view_type,'Value',size(s,1));
    end
else
    handles.annotation.viewType = 'n/a';
	set(handles.popup_view_type,'Value',1);    
end

% scale estimate
if ~isfield(handles.annotation, 'scale')
	handles.annotation.scale = 'n/a';
end

set(handles.text_scale,'String', ...
    ['Scale [pel/cm]: ', num2str(str2num(handles.annotation.scale), '%.3f') ] ); 

% update handles
guidata(hObject, handles);



function handle = get_line( handle )
%*************************************************************************
%
% Input:		handle.I: image
% Output:		handle.l: vector with coordinates of the two points...
%			(with the format [x1 y1 x2 y2])
% Description: 	To click a line in the current axes. Returns two points that have been clicked in the current axes as vector.
%
%*************************************************************************



line_aux = []; X=[]; Y=[]; n = 1; btn = 1;

% click and plot two points
while n < 3 && btn~=3,
    
    % plot mark
    if size(X,1)>0,
        hold on, plot(X(:), Y(:), 'r*'); hold off
    end
    
    % select point
    [ x_aux, y_aux, btn ] = ginput(1);
    if btn ~= 3,
        [ X(n), Y(n) ] = button_down( x_aux, y_aux, handle );

        % plot lines
        plot_lines( handle.l )

        % check if a point has been selected
        if X(n) ~= 0,

            % output vector
            line_aux = [line_aux, Y(n), X(n)];
            n = n+1;

        end
    end
    
end

if btn ~= 3,
    handle.l = [handle.l; line_aux];
else
    handle.l = [];
end
handle.btn = btn;

% plot lines
plot_lines( handle.l ), drawnow

function [ x, y ] = button_down( x_aux, y_aux, handle )
%*************************************************************************
%
% Description: Zooms in to place to select a point more precisely.

% delta is 4 percent of image diagonal
delta = 4 * sqrt(size(handle.I,1)^2+size(handle.I,2)^2) / 100;

% make even
if mod(round(delta),2)==0, 
    delta=round(delta); 
else
    delta=round(delta)-1; 
end

% zoom in region of size delta*delta
X = min(max(round(x_aux)-delta/2, 1), size(handle.I,2)-delta);
Y = min(max(round(y_aux)-delta/2, 1), size(handle.I,1)-delta);

hold off
imshow( imresize( ...
    handle.I( Y : Y+delta, X : X+delta, :), ...
    6, 'bicubic' ) );

axis('equal'), axis('tight'), axis off

% select point
[mX, mY, button] = ginput(1);

if button==1
    
    point = [[mX,mY]/6 + [X-1,Y-1]];
    x=point(1);
    y=point(2);

else
    x=0;
    y=0;
end

imshow(handle.I); axis('equal'); axis('tight'), hold on


function plot_lines( l )
%*************************************************************************
%
% Input: l Lines
%
% Description: Plots lines.
%
%*************************************************************************


if size(l,2)>2,
    c = 'c';
    for i = 1 : size(l,1),
        if i>2, c = 'm'; end
        line( [l(i,2)'; l(i,4)'], [l(i,1)';l(i,3)'], 'Color', c );
    end
end

function [ Ir, Tr ] = geom_rectify_facade( I, l1, l2 )
%*************************************************************************
%
% Input:	l1, list of points pointing towards 1. vanishing point
% Input:	l2, dt. 2.
%
% Description: rectifies image, assuming principle point being the image centre
%
%*************************************************************************

rmax = size(I,1);
cmax = size(I,2);

% principle point
xH=rmax/2;
yH=cmax/2;

% vanishing points
p1=geom_vanishing_point_from_point_pairs(l1,rmax,cmax);
p2=geom_vanishing_point_from_point_pairs(l2,rmax,cmax);

% Euclidean Coordinates (assuming they are existing)
% otherwise take far point in direction of first coordinates
% for principle distance determination
if abs(p1.p(3))>eps*10  
    p1e=p1.p/p1.p(3);
else
    p1e=p1.p/eps/10;
end;
if abs(p2.p(3))>eps*10 
    p2e=p2.p/p2.p(3);
else
    p2e=p2.p/eps/10;
end;

% principle distance
c=sqrt(abs(-(p1e(1)-xH)*(p2e(1)-xH) - (p1e(2)-yH)*(p2e(2)-yH)));

% calibration matrix
K=[c,0,xH;0,c,yH;0,0,1];

% spatial directions to vanishing points
v1=K^(-1)*p1.p;
v2=K^(-1)*p2.p;

% columns of rotation matrix. 
%x-y change: peculiarity of matlab rectification program
d1=[v1(2),v1(1),v1(3)]'*sign(v1(3));
d2=[v2(2),v2(1),v2(3)]'*sign(v2(3));

% change sign of coordinate axes ->
% no additional rotation or mirroring
if v1(2)/v1(3)+v1(1)/v1(3) < 0 
    d1=-d1; 
end;
if v2(2)/v2(3)+v2(1)/v2(3) < 0 
    d2=-d2; 
end;

% third vanishing point
d3=cross(d1,d2);

% order of v1 and v2 does not matter: 
% direction of d3 should be fix and point in positive z-directions
if sign(d3(3)) > 0 
    Rg=[d1/norm(d1),d2/norm(d2), d3/norm(d3)];
else
    Rg=[d2/norm(d2),d1/norm(d1),-d3/norm(d3)];
end;

% orthogonalize rotation matrix
[Aa,Bb,Cc] = svd(Rg);
Rg = Aa*Cc';

% transformation (rotation matrix)
Tr=Rg;

% create Matlab-specific transformation structure
P=maketform('projective',Tr);

% define coordinate system of normalized image (bounding box)
udata = [-yH/c (cmax-yH)/c];  vdata = [-xH/c (rmax-xH)/c];

[Ir,xdata,ydata] = imtransform(I,P,'bicubic','udata',udata,...
                                            'vdata',vdata,...
                                            'size',size(I),...
                                            'fill',128);

% compute transformation in image coordinates
% define coordinate system of the original normalized image
udata = [-yH/c -yH/c (cmax-yH)/c (cmax-yH)/c];  
vdata = [-xH/c (rmax-xH)/c (rmax-xH)/c -xH/c];

% transform the maximum coordinate values of the orig. normalized img
[xdata, ydata] = tformfwd(P, udata, vdata);

% homogenous coordinates
xy_data_norm_hom = [ xdata; ydata; ones(1,length(xdata)) ];

% translate origin
xy_data = [ 1, 0, -min(xdata); ...
             0, 1, -min(ydata); ...
             0, 0,  1 ] * xy_data_norm_hom;

% norm. coeff.
c_prime(1,1) = abs(max(xdata)-min(xdata)) / size(I,2);
c_prime(2,1) = abs(max(ydata)-min(ydata)) / size(I,1);
c_prime(3,1) = 1;

% transformed corners of original image in target image coordinates [pel]
xy(1,:) = xy_data(1,:)./c_prime(1);
xy(2,:) = xy_data(2,:)./c_prime(2);
x = xy(1,:)';
y = xy(2,:)';

% coordinates of the corners of the original image
u = [1 1 size(I,2) size(I,2)]';
v = [1 size(I,1) size(I,1) 1 ]';

% transformation from original image to rectified image [pel]
tform = maketform('projective',[u v],[x y]);

%
Tr = tform.tdata.T;

return;

function vp=geom_vanishing_point_from_point_pairs(l,rmax,cmax)
%*************************************************************************
%
% Description: determines vanishing point from point-pair list
%
%*************************************************************************


% number of lines
L=size(l,1);

% conditioning of the points
hx=rmax/2;
hy=cmax/2; 
m=2000;

H=[1,0,-hx;0,1,-hy;0,0,m];

A = zeros(2,3);

% Coefficient-Matrix 
for i=1:L
    r=H*[l(i,1);l(i,2);1];
    s=H*[l(i,3);l(i,4);1];
    li=cross(r,s);
    
    for j=1:3
        A(i,j)=li(j);
    end;
end;    

% SVD
[u,d,v]=svd(A);

% solution
p=v(:,3);

% Algebraischer Fehler
if L>= 3, vp.l=d(3,3); else vp.l=0; end;

% solution
vp.p=H^(-1)*p;

return;

% --- tranforms labels (original->rectified or rectified->original)
function annotation = transform_labels( struct )
%*************************************************************************
%
% Input:	struct, mn_annotation (xml structure)
%
% Description: tranforms labels (original->rectified or rectified->original)
%
%*************************************************************************


% load the transformation matrix (to within 15 digits of precision)
Tr = eval(struct.annotation.transformationMatrix);

% create Matlab-specific transformation structure
P = maketform('projective',Tr);

% transform labels
for i_obj=1:length(struct.annotation.object),
    % transform polygon points
    for i_pt = 1:length(struct.annotation.object(i_obj).polygon.pt),
        
        %disp(['K#',num2str(i_pt),' (',struct.annotation.object(i_obj).polygon.pt(i_pt).x,',',struct.annotation.object(i_obj).polygon.pt(i_pt).y,')']);
    
        % map the point to the target image axis
        if str2double(struct.annotation.rectified),
            [x_prime, y_prime] = tformfwd( P, ...
                str2double( ...
                struct.annotation.object(i_obj).polygon.pt(i_pt).x ), ...
                str2double( ...
                struct.annotation.object(i_obj).polygon.pt(i_pt).y ) );
            %disp(['R#',num2str(i_pt),' (',num2str(x_prime),',',num2str(y_prime),')']);
        else
            [x_prime, y_prime] = tforminv( P, ...
                str2double( ...
                struct.annotation.object(i_obj).polygon.pt(i_pt).x ), ...
                str2double( ...
                struct.annotation.object(i_obj).polygon.pt(i_pt).y ) );
            %disp(['I#',num2str(i_pt),' (',num2str(x_prime),',',num2str(y_prime),')']);
        end
        
        % save transformed point
        struct.annotation.object(i_obj).polygon.pt(i_pt).x = ...
            num2str(x_prime);
        struct.annotation.object(i_obj).polygon.pt(i_pt).y = ...
            num2str(y_prime);
        
    end        
end

% set return variable
annotation = struct.annotation;

% --- Executes on selection change in lbx_parts.
function lbx_parts_Callback(hObject, eventdata, handles)
% hObject    handle to lbx_parts (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: This Function draws the graph of a part.
%
%*************************************************************************


% Hints: contents = get(hObject,'String') returns lbx_parts contents as cell array
%        contents{get(hObject,'Value')} returns selected item from lbx_parts

i = get(hObject,'Value');
%contents = get(hObject,'String')
%i = contents{get(hObject,'Value')}
%class{n} = handles.mn_annotation.object(i).name; % get object name

if i > 0
    cla;
    imshow(handles.Image);  
    axis('equal'), axis('tight'), axis on, hold on
    [X,Y] = getLMpolygon(handles.annotation.object(handles.LBX_PARTS_NUM(i)).polygon);
    plot([X; X(1)],[Y; Y(1)], 'LineWidth', 4, 'color', [0 0 0]);
    plot([X; X(1)],[Y; Y(1)], 'LineWidth', 2, 'color', ...
          [rand, rand, rand] );
    %handles.colors(mod(sum(double(handles.class{i})),7)+1)
end
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function lbx_parts_CreateFcn(hObject, eventdata, handles)
% hObject    handle to lbx_parts (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function mn_open_Callback(hObject, eventdata, handles)
% hObject    handle to mn_open (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: This function opens the images in a folder and updates lbx_filenames.
%
%*************************************************************************

if handles.MODIFIED == 1
	handles = save_annotation(handles);
	handles.MODIFIED = 0;
end;


% get dir
directory = uigetdir(handles.HOMEIMAGES,'Select Image Directory');

if directory,

    % extract the last part of the string
    k=1;
    while true
       [t{k}, directory] = strtok(directory, ['\', '/']);
       if isempty(t{k}),  break;  end
       k=k+1;
    end
    handles.directory = t{k-1};
    
    % get list of filenames
    handles.files = dir([handles.HOMEIMAGES,'/', handles.directory]);
    handles.filenames = {};
    for i=3:length(handles.files),
		if length(handles.files(i).name) > 0 && ...
			~strcmp(handles.files(i).name, '.svn')
			
			handles.filenames{end+1}=handles.files(i).name;
		end;
    end;
    set(handles.lbx_filenames,'String',handles.filenames);
    set(handles.lbx_filenames, 'Value',1);
	
	% update filename text
	set(handles.text10, 'String', 'Filename:');

    % enable filename listbox
    set(handles.lbx_filenames,'Enable','on');
    
    % disenable filename listbox
    set(handles.mn_aggregation,'Enable','off');
    
    % update handles
    guidata(hObject, handles);
        
end

function handles = lbx_filenames_Callback(hObject, eventdata, handles)
% hObject    handle to lbx_filenames (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
% Hints: contents = get(hObject,'String') returns lbx_filenames contents as cell array
%        contents{get(hObject,'Value')} returns selected item from lbx_filenames
%
%*************************************************************************
%
% Description: This function loads the XML data and connect them to the handles.annotation.
%
%*************************************************************************


% select one mn_annotation file from one of the folders:
% filename = fullfile( handles.directory,  handles.filenames{get(handles.lbx_filenames , 'value')});
% % read the image and mn_annotation struct:
% filename
% [mn_annotation, img] = LMread(filename, handles.HOMEIMAGES);
% % plot the annotations
% LMplot(mn_annotation, img)

% still in AnnotateMode?
if (strcmp(get(handles.btn_add, 'Enable'), 'on') && strcmp(get(handles.btn_annotate, 'String'), 'New Annotation'))
	handles.currObject = '';
end;


reset_panel_annotation(handles);

hold off;

% check for unsaved annotations
handles = mn_save_Callback(hObject, eventdata, handles);

% check if saved
if handles.MODIFIED == 0,
	
	
	% get current image filename
	handles.IMAGEfilename = ...
		handles.filenames{get(handles.lbx_filenames , 'value')};
	
	% get filename
	[pathstr, name ] = fileparts(handles.IMAGEfilename);
	handles.XMLfilename = [name,'.xml'];
	
	
	% load xml data
	if exist([handles.HOMEANNOTATIONS,'/',...
		handles.directory,'/', handles.XMLfilename], 'file'),
		
		% load xml
		struct = loadXML([handles.HOMEANNOTATIONS,'/',...
			handles.directory,'/', handles.XMLfilename]);
		
	else
		struct.annotation = [];
		handles.MODIFIED = 1;
	end;
	
	if isfield(struct, 'annotation')
		handles.annotation = struct.annotation;
	else
		handles.annotation = [];
	end;
	
	% check xml
	handles = CheckXml(handles);
	
	
	if isempty(handles.annotation.object)
		% annotations listbox content
		handles.class = {};
		
		% display image
		cla, imshow(handles.Image);	axis('equal'); axis('tight'), axis on
		hold on
		set(handles.lbx_objects, 'String', '');
	else
		[h, handles.class] = LMplot(handles.annotation, handles.Image);
		axis('tight'), axis on
		set(handles.lbx_objects,'Value',1);
		
		
		if ~handles.BROWSECLASSES
			set(handles.lbx_objects, 'String', handles.class);
		else
		
			classes = sort(handles.class');
			n = length(classes) -1;
			i = 1;
			while i <= n
				if strcmp(classes{i}, classes{i+1})
					classes = [classes(1:i) ; classes(i+2:end)];
					n = length(classes) -1;
				else
					i = i+1;
				end;
			end;
			set(handles.lbx_objects, 'String', classes);
		end;		
		
		
		set(handles.lbx_parts, 'String', '');
	end;
	
	
	% clear image info
	set(handles.popup_image_source,'Value',1);
	set(handles.popup_view_type,'Value',1);
	
	set(handles.text_scale,'String', ...
	['Scale [pel/cm]: ', num2str(str2num(handles.annotation.scale), '%.3f') ] );
	
	% clear mn_annotation info
	set(handles.popup_object_name,'Value',1);
	set(handles.popupmenu_occlusion,'Value',1);
	set(handles.popupmenu_representativeness,'Value',1);
	set(handles.popupmenu_uncertainty,'Value',1);
	
	% mn_aggregation dis/enable: check lbx_objects
	set(handles.mn_aggregation, 'Enable', 'off');
	
	
	% read and display image info
    handles = read_disp_img_info(hObject, handles);
	
	
	% set classes
	class = [{''} handles.annotation.annotatedClasses.className {'other...'}];
	set(handles.popup_object_name, 'String', class);
end

% update ID text
set(handles.txt_objID, 'String', 'no selection');

% update filename text
set(handles.text10, 'String', ...
	['Filename: ' handles.filenames{get(handles.lbx_filenames , 'value')}]);

% update UI Controls
update_uicontrols(handles);

% update handles struc
guidata(hObject, handles);



% --------------------------------------------------------------------
function mn_exit_Callback(hObject, eventdata, handles)
% hObject    handle to mn_exit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

fig_annotation_tool_CloseRequestFcn( ...
    handles.fig_annotation_tool, eventdata, handles)



% --------------------------------------------------------------------
function mn_rectify_Callback(hObject, eventdata, handles)
% hObject    handle to mn_rectify (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: This function allows the user to rectify the image by defining the orthogonal sides.
%
%*************************************************************************


% check for unsaved annotations
handles = mn_save_Callback(hObject, eventdata, handles);

% check if saved
if handles.MODIFIED == 0,

    % refresh image
    h.I = handles.Image;
    imshow(h.I); axis('equal'); axis('tight'), hold on

    [pathstr,name,ext] = fileparts( handles.IMAGEfilename );

    h.l = []; 

	% disable UI
	enable_objects(ones(handles.GuiID.MAX, 1), handles, 'off');
	
    % select points and plot lines
    for i=1:4,
        h = get_line( h ); 
        if h.btn == 3, 
            imshow(handles.Image); axis('equal'); axis('tight'), hold on
			% update the ui control
			update_uicontrols(handles);
            % update handles-structure
            guidata(hObject, handles)
            return, 
        end
        if i==2, l1 = h.l; end
        if i==4, l2 = h.l(3:4,:); end
        
    end

    % rectify image
    [ h.I, Tr ] = geom_rectify_facade(handles.Image,l1,l2);
    %image(h.I); axis('equal'); axis('tight'), hold on
    
    % convert matrix A into a string (15 digits of precision)
    handles.annotation.transformationMatrix = mat2str(Tr);
    handles.MODIFIED = 1;

    % save the transformation matrix
    %handles = save_mn_annotation(handles);

    % save rectified image
    fname_rect = strcat(name,'_rect',ext);
    imwrite(h.I,[handles.HOMEIMAGES, '/', handles.directory, '/', fname_rect])
    %fprintf('\n Rectified image written to file: \n %s \n', fname_rect)

    % update filenames lstbx and select the new img as current file
    handles.files = dir([handles.HOMEIMAGES,'/', handles.directory]);
    handles.filenames = {};
    for i=3:length(handles.files),
		if length(handles.files(i).name) > 0 && ...
			~strcmp(handles.files(i).name, '.svn')
			
			handles.filenames{end+1}=handles.files(i).name;
		end;
    end;
    set(handles.lbx_filenames,'String',handles.filenames);
    set(handles.lbx_filenames, 'Value', ...
        get(handles.lbx_filenames, 'Value')+1);
    
    % as if a new file has been selected; will save the transformation
    handles = lbx_filenames_Callback(hObject, eventdata, handles);
    
    % load labels if any
    handles = mn_load_Callback(hObject, eventdata, handles);
    
    % message a/b saving new file
    msgbox( ['Rectified image saved as: ', fname_rect], 'Info')

    % update UI Controls
	update_uicontrols(handles);
	
	% update handles
    guidata(hObject, handles);

end


% --------------------------------------------------------------------
function handles = mn_load_Callback(hObject, eventdata, handles)
% hObject    handle to mn_load (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: load and transform image mn_annotation
%
%*************************************************************************


% load and transform image mn_annotation

% extract name of the file
[pathstr, name ] = fileparts(handles.IMAGEfilename);

% construct input mn_annotation filename
if str2double(handles.annotation.rectified),

    % mn_annotation file of the original image
    if length(name) > length('_rect'),
    
        % construct the original filename by subtracting '_rect'
        XMLfilename_in = [name(1:end-length('_rect')),'.xml'];
        
    else
        
       disp( ['ERROR: Rectified image "myimage.jpg"', ...
           'should be called "myimage_rect.jpg".' ])
    end
    
else
    % mn_annotation file of the rectified image
    XMLfilename_in = [name, '_rect', '.xml'];
end

% load, transform & plot mn_annotation
if exist([handles.HOMEANNOTATIONS,'/',...
        handles.directory,'/', XMLfilename_in],'file'),

    % load mn_annotation
    struct_xml = loadXML([handles.HOMEANNOTATIONS,'/',...
        handles.directory,'/', XMLfilename_in]);

    % keep name, size & rectified flag
    struct_xml.annotation.filename = handles.annotation.filename;
    handles.annotation.imageWidth = num2str(size(handles.Image,2));			
	handles.annotation.imageHeight = num2str(size(handles.Image,1));
    struct_xml.annotation.rectified = handles.annotation.rectified;

    % check for labels
    if isfield(struct_xml.annotation, 'object')
		
        % transform original labels
        handles.annotation = transform_labels( struct_xml );

        % plot labels
        [struct_xml, handles.class] = LMplot( ...
            handles.annotation, handles.Image);
        axis('tight'), axis on
        
        % update mn_annotation lbx
        set(handles.lbx_objects, 'Value', 1);
        set(handles.lbx_objects, 'String', handles.class);
		
    else
        handles.annotation = struct_xml.annotation;
    end

    % read and display image info
    handles = read_disp_img_info(hObject, handles);

    % set flag
    handles.MODIFIED = 1;

else
    msgbox( 'No annotation data to be loaded.', 'Info' )
end

% update UI Controls
update_uicontrols(handles);

% update handles structure
guidata(hObject, handles);


% --------------------------------------------------------------------
function mn_zoom_Callback(hObject, eventdata, handles)
% hObject    handle to mn_zoom (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: This function switches to the zoom mode.
%
%*************************************************************************


if strcmp(get(hObject,'Checked'),'on')
   set(hObject,'Checked','off');
else
    set(hObject,'Checked','on');
end


if strcmp(get(hObject,'Checked'),'on')
	
	handles.ZOOMMODE = 1;
	
else
    handles.ZOOMMODE = 0;    
end;

% update UI Controls
update_uicontrols(handles);

% update handles structure
guidata(hObject, handles);



% --------------------------------------------------------------------
function handles = mn_save_Callback(hObject, eventdata, handles)
% hObject    handle to mn_save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: This function calls save_annotation.
%
%*************************************************************************


if (handles.MODIFIED == 1),


    if (handles.SAVEDIALOG == 0) || ...	
		(strcmp( questdlg( 'Save annotation?', ...
            'Confirm Save','Yes','No','Yes' ), 'Yes' )),
			
			% save mn_annotation
			handles = save_annotation(handles);
            
    else
        % flag: annotations to save
        handles.MODIFIED = 0;
    end
end;

% set SaveDialog to default (show a dialog)
% handles.SAVEDIALOG = 1;

% update handles
guidata(hObject, handles);


% --- Executes when user attempts to close fig_mn_annotation_tool.
function fig_annotation_tool_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to fig_mn_annotation_tool (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: Executes when user attempts to close fig_mn_annotation_tool.
%
%*************************************************************************


mn_save_Callback(hObject, eventdata, handles);

% Hint: delete(hObject) closes the figure
delete(hObject);



% --------------------------------------------------------------------
function mn_sort_Callback(hObject, eventdata, handles)
% hObject    handle to mn_sort (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: This functions sorts the objects in lbx_objects. 
%
%*************************************************************************


% ---- updates uicontrols ------------------------------------------
function update_uicontrols(handles)
% NEW GUI OBJECTS: add in 'AnnotationTool_OpeningFcn' and 'enable_objects' and modify here
%
%*************************************************************************
%
% Description: This functions updates the ui control by using enable_objects.
%
%*************************************************************************


enables = [];
disables = [];

% -----------------------------------------
% ZOOM MODE
if strcmp(get(handles.mn_zoom, 'Checked'), 'on')
	
	disables = ones(handles.GuiID.MAX, 1);
	
	disables(handles.GuiID.mn_file) = 0;	
	disables(handles.GuiID.mn_exit) = 0;
	disables(handles.GuiID.mn_rectification) = 0;
	disables(handles.GuiID.mn_zoom) = 0;

% END ZOOM MODE
% -----------------------------------------
% AGGREGATE MODE
else if (handles.AGGREGATEMODE == 1)
	
	disables = ones(handles.GuiID.MAX, 1);
	
	disables(handles.GuiID.mn_aggregation) = 0;
	disables(handles.GuiID.mn_finish_aggregation) = 0;
	disables(handles.GuiID.mn_add_part) = 0;
    disables(handles.GuiID.mn_add_part_auto) = 0; % MD
	disables(handles.GuiID.lbx_objects) = 0;
	disables(handles.GuiID.lbx_parts) = 0;
	
	% if there are parts -> enable mn_delete_part
	if (length(handles.LBX_PARTS_NUM) > 0)
		disables(handles.GuiID.mn_delete_part) = 0;
	end

% END AGGREGATE MODE
% -----------------------------------------
% ANNOTATE MODE (set polygon)
else if (handles.ANNOTATIONMODE)
	
	disables = ones(handles.GuiID.MAX, 1);
	
% END ANNOTATE MODE(set polygon)
% -----------------------------------------
% ANNOTATE MODE (inactive mode, e.g. set settings)
else if (strcmp(get(handles.btn_add, 'String'), 'Add Object') && ...
		(strcmp(get(handles.btn_add, 'Enable'), 'on'))) && ...
		~handles.BROWSECLASSES
	
	enables = ones(handles.GuiID.MAX, 1);
	enables(handles.GuiID.mn_finish_aggregation) = 0;
	enables(handles.GuiID.mn_add_part) = 0;	
    enables(handles.GuiID.mn_add_part_auto) = 0; % MD	
	
	% if there are no objects
	if (~isfield(handles.annotation,'object') || ...
		length(handles.annotation.object) <= 0)

		enables(handles.GuiID.mn_annotation) = 0;
		enables(handles.GuiID.mn_aggregation) = 0;
	end;
	
	% if there are no parts
	if (length(handles.LBX_PARTS_NUM) <= 0)
		enables(handles.GuiID.mn_delete_part) = 0;
	end
	
	% enable save mn?
	if 	(handles.MODIFIED == 0)
		enables(handles.GuiID.mn_save) = 0;
	end;

% END ANNOTATE MODE(inactive mode, e.g. set settings)
% -----------------------------------------
% ADJUSTREGION MODE
else if (handles.ADJUSTREGIONMODE)
	
	disables = ones(handles.GuiID.MAX, 1);
	
	disables(handles.GuiID.mn_file)				= 0;
	disables(handles.GuiID.mn_exit)				= 0;
	
	disables(handles.GuiID.mn_annotation)		= 0;
	disables(handles.GuiID.mn_adjustregion)		= 0;

% END ADJUSTREGION MODE
% -----------------------------------------
% BROWSE CLASSES MODE
else if (handles.BROWSECLASSES)
	
	enables = ones(handles.GuiID.MAX, 1);
	
	enables(handles.GuiID.mn_aggregation) = 0;
	enables(handles.GuiID.mn_annotation) = 0;
	enables(handles.GuiID.lbx_parts) = 0;
	enables(handles.GuiID.btn_add) = 0;
	enables(handles.GuiID.btn_annotate:handles.GuiID.popupmenu_uncertainty) = 0;
	
	% if there are no objects
	if isfield(handles,'annotation') && ...
		(~isfield(handles.annotation,'object') || ...
		length(handles.annotation.object) <= 0)
		
		enables(handles.GuiID.mn_annotation) = 0;
		enables(handles.GuiID.mn_aggregation) = 0;
		enables(handles.GuiID.lbx_objects) = 0;
		enables(handles.GuiID.popup_object_name:handles.GuiID.popupmenu_uncertainty) = 0;
	end;
	
	
	% enable save mn?
	if 	(handles.MODIFIED == 0)
		enables(handles.GuiID.mn_save) = 0;
	end;
	
	
	
% END BROWSE CLASSES MODE
% -----------------------------------------
% ADD MORE MODI HERE
% -----------------------------------------
% NONE MODE
else
	
	enables = ones(handles.GuiID.MAX, 1);

	enables(handles.GuiID.mn_finish_aggregation) = 0;
	enables(handles.GuiID.mn_add_part) = 0;
    enables(handles.GuiID.mn_add_part_auto) = 0; % MD
    enables(handles.GuiID.btn_add) = 0;

	% if there are no parts
	if (length(handles.LBX_PARTS_NUM) <= 0)
		
		enables(handles.GuiID.mn_delete_part) = 0;
		enables(handles.GuiID.lbx_parts) = 0;
	

		% if there are no objects
		if isfield(handles,'annotation') && ...
			(~isfield(handles.annotation,'object') || ...
			length(handles.annotation.object) <= 0)

			enables(handles.GuiID.mn_annotation) = 0;
			enables(handles.GuiID.mn_aggregation) = 0;
			enables(handles.GuiID.lbx_objects) = 0;
			enables(handles.GuiID.popup_object_name:handles.GuiID.popupmenu_uncertainty) = 0;
			
			% if there are no images
			if (~isfield(handles, 'files') || ...
				length(handles.files) <= 0)
		
				enables(handles.GuiID.mn_rectification) = 0;
				enables(handles.GuiID.lbx_objects) = 0;
				enables(handles.GuiID.lbx_filenames) = 0;
			
			
		
			end; % images
		end; % objects
	end; % parts
	
	
	% enable save mn?
	if 	(handles.MODIFIED == 0)
		enables(handles.GuiID.mn_save) = 0;
	end;

% END NONE MODE
% -----------------------------------------


end		% BROWSE CLASSES MODE
end		% ADJUSTREGION MODE
end		% ANNOTATE MODE(set polygon)
end		% ANNOTATE MODE(inactive mode, e.g. set settings)
end		% END AGGREGATE MODE
end		% END ZOOM MODE

% choose an dis-/enable option
if ~prod(size(disables))
	
	disables = mod(enables +1, 2);
else
	enables = mod(disables +1, 2);
end;


enable_objects(enables, handles, 'on');
enable_objects(disables, handles, 'off');


% --------------------------------------------------------------------
function mn_delete_Callback(hObject, eventdata, handles)
% hObject    handle to mn_delete (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: This function deletes an entry in lbx_objects.
%
%*************************************************************************


if( strcmp( ...
        questdlg( [ 'Do you want to delete "', ...
        handles.annotation.object( ...
        get(handles.lbx_objects,'Value') ).name, '"' ] ), ...
        'Yes' )),

	% search for relationships between other objects
	% scan objects and delete selected parents
	for objIDs = 1:length(handles.annotation.object)
	if (objIDs ~= get(handles.lbx_objects,'Value')) && exist('handles.annotation.object(objIDs).objectParts')
		
			partsIDs = str2num(str2mat(handles.annotation.object(objIDs).objectParts));
				
	        newObjParts = [];
	        % scan parts
			if (length(partsIDs) > 0)
		        for i = 1:length(partsIDs)

					% id in parts?
					if str2num(handles.annotation.object(get(handles.lbx_objects,'Value')).objectID) == partsIDs(i)

						% delete id in parts
			            for j = 1:length(partsIDs)

							if (str2num(handles.annotation.object(get(handles.lbx_objects,'Value')).objectID)) ~= partsIDs(j)
							newObjParts = [newObjParts; partsIDs(j)];
			                        
			                end
						end
							
						handles.annotation.object(objIDs).objectParts = mat2str(newObjParts);

		            end
		        end
		    end
		end
	end
	% END search for relationships between other objects

		
    % delete object
    handles.annotation.object( ...
        get(handles.lbx_objects,'Value') ) = [];
    
    % update the listbox
    handles.class(get(handles.lbx_objects,'Value')) = [];
    set(handles.lbx_objects, 'Value',1);
    set(handles.lbx_objects, 'String', handles.class);
	
	
    % enable save btn
    handles.MODIFIED = 1;
    set(handles.mn_save,'Enable','on');
    
end

% update UI controls
update_uicontrols(handles);

% update handles structure
guidata(hObject, handles);


% --------------------------------------------------------------------
function mn_set_aggregate_Callback(hObject, eventdata, handles)
% hObject    handle to mn_set_aggregate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if ~isfield(handles,'currObject') || ~prod(size(handles.currObject)),
    
    % save the index of the object whose parts are to be defined
    handles.currObject = get(handles.lbx_objects,'Value');
    
    % get current object string array
    current_objects = get(handles.lbx_objects,'String');

    % set txt_object string
    set( handles.txt_parts, 'String', [ '"', ...
        handles.annotation.object( ...
        get(handles.lbx_objects,'Value')).name, '" Parts:']);
    
    % disable uicontrols
    % TODO
    
    % enbl txt_objects, lbx_objects, btn_parts, btn_save_parts, 
    % txt_parts, lbx_parts
    % TODO
    set(handles.txt_parts,'Enable','on')
    set(handles.lbx_parts,'Enable','inactive')
	
end

% set aggregatemode
handles.AGGREGATEMODE = 1;

% update UI Controls
update_uicontrols(handles);

% update handles
guidata(hObject, handles);



% --------------------------------------------------------------------
function mn_delete_part_Callback(hObject, eventdata, handles)
% hObject    handle to mn_delete_part (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: This function deletes apart of an object.
%
%*************************************************************************


% if there are parts
if prod(size(handles.LBX_PARTS_NUM))
	
	% get names of parts
	content_parts = get(handles.lbx_parts,'String');
	
	% get index of the part to be deleted
	idx_part = get(handles.lbx_parts,'Value');
	
	% update NUM
	handles.LBX_PARTS_NUM = [ handles.LBX_PARTS_NUM(1:(idx_part-1)), ...
		handles.LBX_PARTS_NUM((idx_part+1):end)];
	
	% delete the name
	content_parts(idx_part) = '';
	
	% update lbx_parts
	set(handles.lbx_parts,'Value',1);
	set(handles.lbx_parts,'String', content_parts);
	
	% get index of the object whose part is to be deleted
	if handles.AGGREGATEMODE
		idx_obj = handles.currObject;
	else
		idx_obj = get(handles.lbx_objects, 'Value');
	end;
	
	% get its IDs
	ID = str2num(handles.annotation.object(idx_obj).objectParts);
	
	% delete the ID
	ID = [ID(1:idx_part -1); ID(idx_part +1:end)];	
	
	% update the <objectParts>
	if prod(size(ID))
		handles.annotation.object(idx_obj).objectParts = mat2str(ID);
	else
		handles.annotation.object(idx_obj).objectParts = '';
	end;
	
	% set flag
	handles.MODIFIED = 1;

	% update UI controls
	update_uicontrols(handles);

	% update handles
	guidata(hObject, handles);
end;

% --------------------------------------------------------------------
function mn_add_part_Callback(hObject, eventdata, handles)
% hObject    handle to mn_add_part (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: This function adds a part to an object by adding the part's id.
%
%*************************************************************************


if isfield(handles,'currObject') && prod(size(handles.currObject)),
    
    % get the object index (new part)
    idx_part = get(handles.lbx_objects,'Value');
	
    % add objectID of the new part to the <objectParts> set of IDs
    if ~isfield(handles.annotation.object(handles.currObject), ...
            'objectParts'), 
        
        % will save the ID number only (as a string)
        handles.annotation.object(handles.currObject).objectParts = ...
            mat2str(str2num(handles.annotation.object(idx_part).objectID));

        % get current parts
        current_parts = get(handles.lbx_parts,'String');

        % add object name to current parts
        current_parts{end+1} = handles.annotation.object(idx_part).name;
        set(handles.lbx_parts, 'String', current_parts);
		
		% add object ID to LBX_PARTS_NUM
		handles.LBX_PARTS_NUM(end +1) = idx_part;
		
    else
        
        IDs = str2num(str2mat( ...
            handles.annotation.object(handles.currObject).objectParts ));
        
        % check for multiple instances and self occlusion
        if (isempty(IDs) || ( sum( ismember( IDs, str2num( ...
                handles.annotation.object(idx_part).objectID))) == 0 )) && ...
				( str2num( handles.annotation.object(handles.currObject).objectID ) ~= ...
                str2num(handles.annotation.object(idx_part).objectID) )
			
            % add ID
            IDs(end+1,1) = str2num( ...
				handles.annotation.object(idx_part).objectID);
            
            % convert matrix to a string in MATLAB syntax
            handles.annotation.object(handles.currObject).objectParts = ...
                mat2str(IDs);
            
            % get current parts
            current_parts = get(handles.lbx_parts,'String');

            % add object name to current parts
            current_parts{end+1} = handles.annotation.object(idx_part).name;
            set(handles.lbx_parts, 'String', current_parts);
			
			% add object ID to LBX_PARTS_NUM
			handles.LBX_PARTS_NUM(end +1) = idx_part;
        end
    end
end


% update UI controls
update_uicontrols(handles);

% update handles struct
guidata(hObject, handles);


% --------------------------------------------------------------------
function mn_finish_aggregation_Callback(hObject, eventdata, handles)
% hObject    handle to mn_finish_aggregation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% enable uicontrols
% TODO

% set text description to default
set(handles.txt_parts,'String','Current Object Parts: ');

% clear var
handles.currObject = '';

% set flag
handles.MODIFIED = 1;
set(handles.mn_save,'Enable','on');

% update handles
guidata(hObject, handles);

% set aggregatemode
handles.AGGREGATEMODE = 0;

% update UI Controls
update_uicontrols(handles);

guidata(hObject, handles);

% --------------------------------------------------------------------
function mn_show_parts_Callback(hObject, eventdata, handles)
% hObject    handle to mn_show_parts (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% just change check-state
if strcmp(get(hObject,'Checked'),'on')
   set(hObject,'Checked','off');
else
    set(hObject,'Checked','on');
end


% refresh image
i = get(handles.lbx_objects,'Value');

% redraw the polygons if there's a image
if length(handles.annotation.object) > 0
    cla;
	imshow(handles.Image);
 
    axis('equal'), axis('tight'), axis on, hold on
    [X,Y] = getLMpolygon(handles.annotation.object(i).polygon);
    plot([X; X(1)],[Y; Y(1)], 'LineWidth', 4, 'color', [0 0 0]);
    plot([X; X(1)],[Y; Y(1)], 'LineWidth', 2, 'color', ...
              [rand, rand, rand] );

    if strcmp(get(handles.mn_show_parts,'Checked'), 'on') && ( isfield(handles.annotation.object(i), 'objectParts') && ...
            ~isfield(handles, 'currObject') ) || ...
            ( isfield(handles.annotation.object(i), 'objectParts') && ...
            isfield(handles, 'currObject') && ...
            isempty(handles.currObject) ),
        % read IDs in a numeric matrix
        IDs = str2num(str2mat(handles.annotation.object(i).objectParts)); %#ok<ST2NM>

            % list objects with IDs in <objectParts> in lbx_parts
            for idx_id = 1:length(IDs)
               for idx_obj = 1:length(handles.annotation.object),

                  if str2double(handles.annotation.object(idx_obj).objectID) ...
                          == IDs(idx_id),

						if (length(handles.LBX_PARTS_NUM) > 0) && strcmp(get(handles.mn_show_parts,'Checked'), 'on')

                        % add object's parts
                        axis('equal'), axis('tight'), axis on, hold on
                        [X,Y] = getLMpolygon(handles.annotation.object(idx_obj).polygon);
                        plot([X; X(1)],[Y; Y(1)], 'LineWidth', 4, 'color', [0 0 0]);
                        plot([X; X(1)],[Y; Y(1)], 'LineWidth', 2, 'color', ...
                        [rand, rand, rand] );
					end

                  end
               end
            end
    end
end

guidata(hObject, handles);


% ---- update the mn_annotation Panel ------------------------------------------
% ---- i: number of object ------------------------------------------
function handles = update_panel_annotation(handle, i)
%*************************************************************************
%
% Description: This function updates the mn_annotation panel.
%
%*************************************************************************


obj = handle.annotation.object(i);

% enable annotate and add button
set(handle.btn_annotate, 'Enable', 'on');
set(handle.btn_add, 'Enable', 'off');

% enable controls
set(handle.popup_object_name, 'Enable', 'on');
set(handle.popupmenu_occlusion, 'Enable', 'on');
set(handle.popupmenu_representativeness, 'Enable','on');
set(handle.popupmenu_uncertainty, 'Enable', 'on');

% set panel and buttons
set(handle.newannotpanel, 'Title', 'Current Object');
set(handle.btn_annotate, 'String', 'New Annotation');
set(handle.btn_add, 'String', 'Save Object');


% find indices
% ---------------------------------------------------
% ----------------------
% find name
str = get(handle.popup_object_name, 'String');

oname = 1;
j = 2;
while ((oname == 1) && (j <= length(str)))
	if (strcmp(obj.name, str(j)))
		oname = j;
    end
	j = j+1;
end


if (oname == 1)
	handle.objname = handle.annotation.object(i).name;
	oname = length(str);
else
	handle.objname = str{oname};
end;

% ---------------------------------------------------
% find occlusion
str = get(handle.popupmenu_occlusion, 'String');

oocclusion = 1;
j = 2;
while ((oocclusion == 1) && (j <= length(str)))
    if (str2double(obj.occlusion) == str2double(str(j)))
		oocclusion = j;
    end
	j = j+1;	
end

% ---------------------------------------------------
% find representativeness
str = get(handle.popupmenu_representativeness, 'String');

orepresentativeness = 1;
j = 2;
while ((orepresentativeness == 1) && (j <= length(str)))
    if (str2double(obj.representativeness) == str2double(str(j)))
		orepresentativeness = j;
    end
	j = j+1;	
end

% ---------------------------------------------------
% find uncertainty
str = get(handle.popupmenu_uncertainty, 'String');

ouncertainty = 1;
j = 2;
while ((ouncertainty == 1) && (j <= length(str)))
    if (str2double(obj.uncertainty) == str2double(str(j)))
		ouncertainty = j;
    end
	j = j+1;	
end

% ----------------------
% ---------------------------------------------------



% set controls
set(handle.popup_object_name, 'Value', oname);
set(handle.popupmenu_occlusion, 'Value', oocclusion);
set(handle.popupmenu_representativeness, 'Value', orepresentativeness);
set(handle.popupmenu_uncertainty, 'Value',  ouncertainty);


% save obj
handle.annotation.object(i) = obj;

% Update handle structure
handles = handle;



% ---- reset mn_annotation Panel ------------------------------------------
function reset_panel_annotation(handles)
%*************************************************************************
%
% Description: This function resets the mn_annotation panel.
%
%*************************************************************************


if strcmp(get(handles.btn_annotate, 'String'), 'New Annotation')

	% enable annotate btn
	set(handles.btn_annotate, 'Enable', 'on'),

	% disable Add btn
	set(handles.btn_add, 'Enable', 'off'),

	% set panel and buttons
	set(handles.newannotpanel, 'Title', 'New Annotation'),
	set(handles.btn_annotate, 'String', 'Annotate'),
	set(handles.btn_add, 'String', 'Add Object'),

	% reset controls
	set(handles.popup_object_name, 'Value', 1),
	set(handles.popupmenu_occlusion, 'Value', 1),
	set(handles.popupmenu_representativeness, 'Value', 1),
	set(handles.popupmenu_uncertainty, 'Value',  1),
	
	% disable controls
	set(handles.popup_object_name, 'Enable', 'off'),
	set(handles.popupmenu_occlusion, 'Enable', 'off'),
	set(handles.popupmenu_representativeness, 'Enable', 'off'),
	set(handles.popupmenu_uncertainty, 'Enable',  'off'),

end




% --------------------------------------------------------------------
function mn_objectnote_Callback(hObject, eventdata, handles)
% hObject    handle to mn_objectnote (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: This function calls an dialog to get a object's note.
%
%*************************************************************************


i = get(handles.lbx_objects, 'Value');
if ((i > 0) && isfield(handles.annotation, 'object'))
	
	% create dialog
	if (isfield(handles.annotation.object(i), 'name') && isfield(handles.annotation.object(i), 'objectID'))
		
		str = ['Enter a Note for: ', ...
			handles.annotation.object(i).name, ...
			' (ID: ', handles.annotation.object(i).objectID, ...
			')'];
	else
		str = strcat('Enter a Note');
	end;
	

	dlg_msg = {str};
	dlg_title = 'Object Note';
	dlg_num_lines = 1;

	
	if isfield(handles.annotation.object(i), 'comment')
        
        if (~isstr(handles.annotation.object(i).comment))
            handles.annotation.object(i).comment = '';
        end;
        
    		dlg_def = cellstr(handles.annotation.object(i).comment);        
	else
		% there are no comments -> initialise comment
		for (j = 1:length(handles.annotation.object))
			handles.annotation.object(j).comment = '';
		end
		
		% flag: there are annotations to save
		handles.MODIFIED = 1;
		
		dlg_def = {''};
	end;
	
	
	dlg_options.Resize='on';
	dlg_options.WindowStyle='modal';
	dlg_options.Interpreter='none';
	

    ans = inputdlg(dlg_msg, dlg_title, dlg_num_lines, dlg_def, dlg_options);
    
    if (length(ans) > 0)
		handles.annotation.object(i).comment = cell2mat(ans);
		
		% flag: there are annotations to save
		handles.MODIFIED = 1;
    end;

end;


% Update handles structure
guidata(hObject, handles);


% --------------------------------------------------------------------
function mn_editsource_Callback(hObject, eventdata, handles)
% hObject    handle to mn_editsource (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: This function allows the user to edit the author's list and enter new authors. Authors can be deleted by leaving the entry blank.
%
%*************************************************************************


i = get(handles.lbx_objects, 'Value');
if ((i > 0) && isfield(handles.annotation, 'object'))
	
	% create dialog

	
	dlg_title = 'Author List';
	dlg_num_lines = 1;

	
	if isfield(handles.annotation.object(i), 'sourceAnnotation')

		% get length of longest string-part (k)
	
		% build dlg_def
		[str,remain] = strtok(handles.annotation.object(i).sourceAnnotation, ',');
		
		dlg_def = {str};
		
		
		while (length(remain) > 0)
			
			[str,remain] = strtok(remain, ',');
			
			dlg_def(end+1) = {str};

		end;	
		
		dlg_def(end+1) = {''};
		
		
		
		dlg_msg = {'Author:'};
		for j = 2:(length(dlg_def) -1)
			dlg_msg(end+1) = {'Author:'};
		end;
		dlg_msg(end +1) = {'optional additional Author:'};
		
		
		
		
	else
		% there are no sourceAnnotation -> initialise sourceAnnotation
		for (j = 1:length(handles.annotation.object))
			handles.annotation.object(j).sourceAnnotation = '';
		end
		
		% flag: there are annotations to save
		handles.MODIFIED = 1;
		
		dlg_msg = {'Author:'};
		dlg_def = {''};
	end;
	
	
	dlg_options.Resize='on';
	dlg_options.WindowStyle='modal';
	dlg_options.Interpreter='none';
	
	
    ans = inputdlg(dlg_msg, dlg_title, dlg_num_lines, dlg_def, dlg_options);
    
	% save input
    if (length(ans) > 0)

		source = '';
		
		
		for j = 1:length(ans)

			if ~strcmp(ans(j,:), '')
				
				if strcmp(source, '')
					source = ans(j,:);
				else
					source = strcat(source, ',', ans(j, :));
				end;
			end;
		end;
		
		
		if ~strcmp(handles.annotation.object(i).sourceAnnotation, cell2mat(source))
		
			handles.annotation.object(i).sourceAnnotation = cell2mat(source);
		
			% flag: there are annotations to save
			handles.MODIFIED = 1;
		end;
    end;

end;

% update UI Controls
update_uicontrols(handles);

% Update handles structure
guidata(hObject, handles);



% --------------------------------------------------------------------
function mn_about_Callback(hObject, eventdata, handles)
% hObject    handle to mn_about (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

text = sprintf('%s%s\n\n%s\n\n%s\n%s\n\n%s', ...
    'Annotation Tool Version ',handles.version, ...
    'Annotation Tool is an image annotation tool developed at the Department of Photogrammetry, University of Bonn.', ...
    'Related Documentation:', ...
    'www.ipb.uni-bonn.de/~filip/annotation-tool',...
    'Annotation Tool employs LabelMe MATLAB Toolbox developed at MIT.');

msgbox(text,'About Annotation Tool')



% --------------------------------------------------------------------
function mn_adjustregion_Callback(hObject, eventdata, handles)
% hObject    handle to mn_adjustregion (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: This function allows the user to adjust annotated region boundary.
%
%*************************************************************************



if strcmp(get(hObject,'Checked'),'on')

	set(hObject,'Checked','off');
	handles.ADJUSTREGIONMODE = 0;
	handles.ZOOMMODE = 0;
	
	% flag: there are annotations to save
	handles.MODIFIED = 1;
	
else
    set(hObject,'Checked','on');
	
	% draw polygons
	cla, imshow(handles.Image); axis('off'), hold on
	
	% draw polygons
	handles.arColor = [rand, rand, rand];
	[X,Y] = getLMpolygon(handles.annotation.object(get(handles.lbx_objects, 'Value')).polygon);
	plot([X; X(1)],[Y; Y(1)], 'LineWidth', 4, 'color', [0 0 0]);
	plot([X; X(1)],[Y; Y(1)], 'LineWidth', 2, 'color', ...
	handles.arColor);
	
	
	% store polygons
	handles.ar_poly.X = X;
	handles.ar_poly.Y = Y;	
	
	
	% switch mode
	handles.ADJUSTREGIONMODE = 1;
	handles.ZOOMMODE = 1;
end;

% update UI Controls
update_uicontrols(handles);

% Update handles structure
guidata(hObject, handles);


% --- Executes on mouse press over figure background, over a disabled or
% --- inactive control, or over an axes background.
function fig_annotation_tool_WindowButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to fig_annotation_tool (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% if in adjust region mode
if (handles.ADJUSTREGIONMODE)
	
	% if left mouse button pressed
	if strcmp(get(hObject,'SelectionType'), 'normal')
		
		% get selected position
		set(hObject, 'Units', 'pixels');
		output = get(hObject,'CurrentPoint');
		Xc = output(1);
		Yc = output(2);
		set(hObject, 'Units', 'Characters');
		
		
		% transform to image coordinates
		[X,Y] = FigCoord2ImgCoord(handles, Xc, Yc);
		
		
		% get image size
		imgSize = size(handles.Image);
		
		
		% check x,y values
		if (X < 1) || (X > imgSize(2)) || ...
			(Y < 1) || (Y > imgSize(1))
			
			return;
		end;
		% end check x,y values
		
		
		% START: convert polygons from image size to pixel size
		imgSize = size(handles.Image);
		
		
		% get nearest vertex
		j = 1;
		diff = norm([(handles.ar_poly.X(1) - X) ...
						(handles.ar_poly.Y(1) - Y)]);
						
		for i = 2:length(handles.ar_poly.X)
			
			tempDiff = norm([(handles.ar_poly.X(i) - X) ...
						(handles.ar_poly.Y(i) - Y)]);
			
			if (tempDiff < diff)
				diff = tempDiff;
				j = i;
			end;
		end;
		% END: get nearest vertex
		
		
		% check nearest vertex
		if (diff > norm([imgSize(2) imgSize(1)]) * handles.AR_TOLERANCE)
			handles.AR_USE_POINT = 0;
		else
			handles.ar_poly.selection = j;
			handles.AR_USE_POINT = 1;
		end;
		% END: check nearest vertex
		
		
	% if right mouse button pressed
	else if strcmp(get(hObject,'SelectionType'), 'alt')
		
		% quit adjust region mode 
		set(handles.mn_adjustregion,'Checked','off');
		handles.ADJUSTREGIONMODE = 0;
		handles.ZOOMMODE = 0;
	
		% flag: there are annotations to save
		handles.MODIFIED = 1;
	end;
	end;
	
	% update UI Controls
	update_uicontrols(handles);

	% Update handles structure
	guidata(hObject, handles);
end;


% if in annotation mode
if (handles.ANNOTATIONMODE)
	
	% get image size
	imgSize = size(handles.Image);
	
	% test for visability of image
	XLim = get(gca,'XLim');
	YLim = get(gca,'YLim');
	
	if (XLim(1) <= 0) && (XLim(2) <= 0) || ...
		(XLim(1) >= imgSize(2)) && (XLim(2) >= imgSize(2)) || ...
		(YLim(1) <= 0) && (YLim(2) <= 0) || ...
		(YLim(1) >= imgSize(1)) && (YLim(2) >= imgSize(1))
		
		% quit annotation mode
		handles.AN_POLYGONMODE = 0;
		handles.ANNOTATIONMODE = 0;
		handles.ZOOMMODE = 0;
		%handles = zoom_reset(handles);
		
		% update UI Controls
		update_uicontrols(handles);
		
		% Update handles structure
		guidata(hObject, handles);

		return;
	end;
	% END: test for visability of image
	
	
	handles.AN_CHECKMOTION = 1;
	
	% get selected position
	set(hObject, 'Units', 'pixels');
	output = get(hObject,'CurrentPoint');
	Xc = output(1);
	Yc = output(2);
	set(hObject, 'Units', 'Characters');
	
	
	% transform to image coordinates
	[X,Y] = FigCoord2ImgCoord(handles, Xc, Yc);
	
	
	% check x,y values
	if (X < 1) || (X > imgSize(2)) || ...
		(Y < 1) || (Y > imgSize(1))
		
		return;
	end;
	% end check x,y values	
	
	% polymode:
	% if left is pressed first time
	if strcmp(get(hObject,'SelectionType'), 'normal')
		
		handles.AN_POLYGONMODE = 1;
		
		% add points to list
		handles.an_poly.n = handles.an_poly.n +1;
		handles.an_poly.XY(:,handles.an_poly.n) = [X; Y];
		
		% draw points
		cla, imshow(handles.Image); axis('off'), hold on
		
		% if >1 points
		if (handles.an_poly.n > 1)
			
			plot([handles.an_poly.XY(1,:)],[handles.an_poly.XY(2,:)], ...
				'LineWidth', 4, 'color', [0 0 0]);
			
			plot([handles.an_poly.XY(1,:)],[handles.an_poly.XY(2,:)], ...
				'LineWidth', 2, 'color', handles.anColor );
			
		% if ==1 points
		else if (handles.an_poly.n == 1)
			
			plot(X,Y,'ro')
		end;
		end;
	
	% polymode:
	% if middle is pressed -> delete last point
	else if strcmp(get(hObject,'SelectionType'), 'extend')
		
		handles.AN_CHECKMOTION = 0;
		
		if (handles.AN_POLYGONMODE)
			
			if (handles.an_poly.n > 1)
				
				% reset values
				handles.an_poly.n = handles.an_poly.n-1;
				handles.an_poly.XY = handles.an_poly.XY(:,1:handles.an_poly.n);
				
				
				% draw points
				cla, imshow(handles.Image); axis('off'), hold on
				
				if (handles.an_poly.n > 1)
					
					plot([handles.an_poly.XY(1,:)],[handles.an_poly.XY(2,:)], ...
						'LineWidth', 4, 'color', [0 0 0]);
					
					plot([handles.an_poly.XY(1,:)],[handles.an_poly.XY(2,:)], ...
						'LineWidth', 2, 'color', handles.anColor );
				else
					plot(handles.an_poly.XY(1,:),handles.an_poly.XY(2,:),'ro');
				end;				
				
			else
				
				% reset values
				handles.an_poly.XY = [];
				handles.an_poly.n = 0;
				
				% clear screen
	            cla, imshow(handles.Image); axis('off'), hold on;
				
				handles.AN_POLYGONMODE = 0;
			end;
		end;
	
	% polymode:
	% if right is pressed >1 time and polygon mode
	else if strcmp(get(hObject,'SelectionType'), 'alt') && ...
		(handles.AN_POLYGONMODE == 1)
		
		if size(handles.an_poly.XY,2)>0
			
			% draw points
			cla, imshow(handles.Image); axis('off'), hold on
			
			plot([handles.an_poly.XY(1,:), handles.an_poly.XY(1,1)],[handles.an_poly.XY(2,:), handles.an_poly.XY(2,1)], ...
				'LineWidth', 4, 'color', [0 0 0]);
			plot([handles.an_poly.XY(1,:), handles.an_poly.XY(1,1)],[handles.an_poly.XY(2,:), handles.an_poly.XY(2,1)], ...
				'LineWidth', 2, 'color', handles.anColor);
			
			handles.currObject.xy = handles.an_poly.XY;
			
		else
			clear handles.currObject;
		end;
		
		% quit annotation mode
		handles.AN_POLYGONMODE = 0;
		handles.ANNOTATIONMODE = 0;
		handles.ZOOMMODE = 0;
		%handles = zoom_reset(handles);
		set(handles.btn_add, 'Enable', 'on');
	
	
	% boxmode:
	% if right is pressed and no polygon mode
	else if strcmp(get(hObject,'SelectionType'), 'alt') && ...
		~(handles.AN_POLYGONMODE)
		
		% draw points
		%	cla, imshow(handles.Image); axis('off'), hold on
		%	plot(X,Y,'ro');
		
		handles.an_poly.n = 2;
		handles.an_poly.XY(:,1) = [X; Y];
		handles.an_poly.XY(:,2) = [X; Y];
		
	end;	% boxmode:
	end;	% polymode: if right is pressed >1 time and polygon mode
	end;	% polymode: if middle is pressed -> delete last point
	end;	% polymode: if left is pressed first time
	
	
	% update UI Controls
	update_uicontrols(handles);

	% Update handles structure
	guidata(hObject, handles);
end;


% --- Executes on mouse motion over figure - except title and menu.
function fig_annotation_tool_WindowButtonMotionFcn(hObject, eventdata, handles)
% hObject    handle to fig_annotation_tool (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% if in adjust region mode with selected point
if (handles.ADJUSTREGIONMODE) && ...
	(handles.AR_USE_POINT)
	
	
	% get selected position
	set(hObject, 'Units', 'pixels');
	output = get(hObject,'CurrentPoint');
	Xsel = output(1);
	Ysel = output(2);
	set(hObject, 'Units', 'Characters');
	
	
	% transform coord
	[X,Y] = FigCoord2ImgCoord(handles, Xsel, Ysel);
	
	
	% check x,y values
	imgSize = size(handles.Image);
	if (X < 1)
		X = 1;
	else if (X > imgSize(2))
		X = imgSize(2);
		end;
	end;
	
	if (Y < 1)
		Y = 1;
	else if (Y > imgSize(1))
		Y = imgSize(1);
		end;
	end;
	% end check x,y values	
	
	% change polygons
	handles.ar_poly.X(handles.ar_poly.selection) = X;
	handles.ar_poly.Y(handles.ar_poly.selection) = Y;
	
	
	% draw new polygons
	cla, imshow(handles.Image); axis('off'), hold on
	
	plot([handles.ar_poly.X; handles.ar_poly.X(1)],[handles.ar_poly.Y; handles.ar_poly.Y(1)], 'LineWidth', 4, 'color', [0 0 0]);
	plot([handles.ar_poly.X; handles.ar_poly.X(1)],[handles.ar_poly.Y; handles.ar_poly.Y(1)], 'LineWidth', 2, 'color', ...
	handles.arColor);
	
	% Update handles structure
	guidata(hObject, handles);
end;



% if in annotation mode
if (handles.ANNOTATIONMODE) && ...
	(handles.AN_CHECKMOTION)
	
	% get selected position
	set(hObject, 'Units', 'pixels');
	output = get(hObject,'CurrentPoint');
	Xc = output(1);
	Yc = output(2);
	set(hObject, 'Units', 'Characters');
	
	
	% transform to image coordinates
	[X,Y] = FigCoord2ImgCoord(handles, Xc, Yc);
	
	
	% get image size
	imgSize = size(handles.Image);
	
	
	% check x,y values
	if (X < 1) || (X > imgSize(2)) || ...
		(Y < 1) || (Y > imgSize(1))
		
		return;
	end;
	% end check x,y values
	
	% if in polygon mode
	if (handles.AN_POLYGONMODE == 1)
		
		% change point
		handles.an_poly.XY(:,handles.an_poly.n) = [X; Y];
		
		% draw points
		cla, imshow(handles.Image); axis('off'), hold on
		
		% if >1 points
		if (handles.an_poly.n > 1)
			
			plot([handles.an_poly.XY(1,:)],[handles.an_poly.XY(2,:)], ...
				'LineWidth', 4, 'color', [0 0 0]);
			
			plot([handles.an_poly.XY(1,:)],[handles.an_poly.XY(2,:)], ...
				'LineWidth', 2, 'color', handles.anColor );
				
		% if ==1 points
		else if (handles.an_poly.n == 1)
			
			plot(X,Y,'ro');
		end;
		end;
		
		
	else
	% if box mode
		
		% change point
		handles.an_poly.XY(:,2) = [X; Y];
		handles.an_poly.n = 2;
		
		
		% get left and right
		if handles.an_poly.XY(1,1) < handles.an_poly.XY(1,2)
		
			left = handles.an_poly.XY(1,1);
			right = handles.an_poly.XY(1,2);
		else
			left = handles.an_poly.XY(1,2);
			right = handles.an_poly.XY(1,1);
		end;
		
		% get top and bottom
		if handles.an_poly.XY(2,1) < handles.an_poly.XY(2,2)
		
			top = handles.an_poly.XY(2,1);
			bottom = handles.an_poly.XY(2,2);
		else
			top = handles.an_poly.XY(2,2);
			bottom = handles.an_poly.XY(2,1);
		end;
		
		% set values
			% top left
			XY(1,1) = left;
			XY(2,1) = top;
			% top right
			XY(1,2) = right;
			XY(2,2) = top;
			% bottom right
			XY(1,3) = right;
			XY(2,3) = bottom;
			% bottom left
			XY(1,4) = left;
			XY(2,4) = bottom;
		% END: set values
		
		% draw polygons
		cla, imshow(handles.Image); axis('off'), hold on
		plot([XY(1,:), XY(1,1)],[XY(2,:), XY(2,1)], ...
			'LineWidth', 4, 'color', [0 0 0]);
		plot([XY(1,:), XY(1,1)],[XY(2,:), XY(2,1)], ...
			'LineWidth', 2, 'color', handles.anColor);
		
		
	end;
	
	% Update handles structure
	guidata(hObject, handles);
end;



function [X, Y] = FigCoord2ImgCoord(handles, Xc, Yc)
%
%*************************************************************************
%
% Input: handles, the global handle structure.
% Input: Xc, Yc, the coordinates to convert.
% Input: handles, the global handle structure.
% Output: [X,Y], the image coordinates.
%
% Description: This function converts figure coordinates to image coordinates.
%
%*************************************************************************

% get image size
XLim = get(gca,'XLim');
YLim = get(gca,'YLim');
imgSize(2) = floor(abs(XLim(1) - XLim(2)));
imgSize(1) = floor(abs(YLim(1) - YLim(2)));


% get image size and position
set(handles.axes1, 'Units', 'pixels');
output = get(handles.axes1, 'Position');
set(handles.axes1, 'Units', 'normalized');

Ximg = output(1);
Yimg = output(2);
XSizeimg = output(3);
YSizeimg = output(4);

% check values
if imgSize(1) == 0 || ...
	YSizeimg == 0
	
	X = -1;
	Y = -1;
	return;
end;

% calc proportion
pImg = imgSize(2) / imgSize(1);
pAxes = XSizeimg / YSizeimg;

if (pAxes >= pImg)
	
	% calc image X size
	proportion = imgSize(2) / imgSize(1);
	sizeX = proportion * YSizeimg;
	
	% middle
	middle = Ximg + XSizeimg/2;
	
	% x position
	posX = middle - sizeX / 2;
	
	% Y values
	sizeY = YSizeimg;
	posY = Yimg;
	
else
	% X values
	sizeX = XSizeimg;
	posX = Ximg;
	
	% calc image Y size
	proportion = imgSize(1) / imgSize(2);
	sizeY = proportion * XSizeimg;
	
	% middle
	middle = Yimg + YSizeimg/2;
	
	% Y position
	posY = middle - sizeY / 2;
	
end;


X = ((Xc - posX) / sizeX) * imgSize(2);
Y = imgSize(1) - ((Yc - posY) / sizeY)  * imgSize(1);

X = X + XLim(1);
Y = Y + YLim(1);


% check borders
if (X < XLim(1)) || (X < 1)
	if (XLim(1) < 1)
		X = 1;
	else
		X = XLim(1);
	end;
else if (X > XLim(2))
	X = XLim(2);
end;
end;

if (Y < YLim(1)) || (Y < 1)
	if (YLim(1) < 1)
		Y = 1;
	else
		Y = YLim(1);
	end;
else if (Y > YLim(2))
	Y = YLim(2);
end;
end;




% --- Executes on mouse press over figure background, over a disabled or
% --- inactive control, or over an axes background.
function fig_annotation_tool_WindowButtonUpFcn(hObject, eventdata, handles)
% hObject    handle to fig_annotation_tool (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% if adjust region mode
if (handles.ADJUSTREGIONMODE) && ...
	(handles.AR_USE_POINT)
	
	handles.AR_USE_POINT = 0;
	
	if strcmp(get(hObject,'SelectionType'), 'normal')
		
		% save current point
		
		handles.annotation.object(1, get(handles.lbx_objects, ...
			'Value')).polygon.pt(1, handles.ar_poly.selection).x = ...
			num2str((handles.ar_poly.X(handles.ar_poly.selection)));
		
	    handles.annotation.object(1, get(handles.lbx_objects, ...
			'Value')).polygon.pt(1, handles.ar_poly.selection).y = ...
			num2str((handles.ar_poly.Y(handles.ar_poly.selection)));
		
		
	else if strcmp(get(hObject,'SelectionType'), 'alt')
		
		% quit adjust region mode 
		set(handles.mn_adjustregion,'Checked','off');
		handles.ADJUSTREGIONMODE = 0;
		handles.ZOOMMODE = 0;
	
		% flag: there are annotations to save
		handles.MODIFIED = 1;
		
		% draw polygons
		cla, imshow(handles.Image); axis('off'), hold on
		
		[X,Y] = getLMpolygon(handles.annotation.object(get(handles.lbx_objects, 'Value')).polygon);
		plot([X; X(1)],[Y; Y(1)], 'LineWidth', 4, 'color', [0 0 0]);
		plot([X; X(1)],[Y; Y(1)], 'LineWidth', 2, 'color', ...
		handles.arColor);
		
		% update UI Controls
		update_uicontrols(handles);
	end
	end;
	
	% update UI Controls
	update_uicontrols(handles);
	
	% Update handles structure
	guidata(hObject, handles);
end;


% if annotation mode
if (handles.ANNOTATIONMODE) && ...
	(handles.AN_CHECKMOTION)
	
	handles.AN_CHECKMOTION = 0;
	
	
	% if in polygon mode
	if (handles.AN_POLYGONMODE == 1)
	
		% nothing to do
	
	% if in box mode
	else
		if (handles.an_poly.XY(:,1) == handles.an_poly.XY(:,2))
			% quit annotation mode
			handles.ANNOTATIONMODE = 0;
			handles.ZOOMMODE = 0;
			%handles = zoom_reset(handles);
			set(handles.btn_add, 'Enable', 'on');
		end;
		
		
		% get left and right
		if handles.an_poly.XY(1,1) < handles.an_poly.XY(1,2)
		
			left = handles.an_poly.XY(1,1);
			right = handles.an_poly.XY(1,2);
		else
			left = handles.an_poly.XY(1,2);
			right = handles.an_poly.XY(1,1);
		end;
		
		% get top and bottom
		if handles.an_poly.XY(2,1) < handles.an_poly.XY(2,2)
		
			top = handles.an_poly.XY(2,1);
			bottom = handles.an_poly.XY(2,2);
		else
			top = handles.an_poly.XY(2,2);
			bottom = handles.an_poly.XY(2,1);
		end;
		
		% set values
			% top left
			handles.an_poly.XY(1,1) = left;
			handles.an_poly.XY(2,1) = top;
			% top right
			handles.an_poly.XY(1,2) = right;
			handles.an_poly.XY(2,2) = top;
			% bottom right
			handles.an_poly.XY(1,3) = right;
			handles.an_poly.XY(2,3) = bottom;
			% bottom left
			handles.an_poly.XY(1,4) = left;
			handles.an_poly.XY(2,4) = bottom;
		% END: set values
		
		% draw polygons
		cla, imshow(handles.Image); axis('off'), hold on
		plot([handles.an_poly.XY(1,:), handles.an_poly.XY(1,1)],[handles.an_poly.XY(2,:), handles.an_poly.XY(2,1)], ...
			'LineWidth', 4, 'color', [0 0 0]);
		plot([handles.an_poly.XY(1,:), handles.an_poly.XY(1,1)],[handles.an_poly.XY(2,:), handles.an_poly.XY(2,1)], ...
			'LineWidth', 2, 'color', handles.anColor);
		
		% save XY values
		handles.currObject.xy = handles.an_poly.XY;
		
		% quit annotation mode
		handles.ANNOTATIONMODE = 0;
		handles.ZOOMMODE = 0;
		%handles = zoom_reset(handles);
		set(handles.btn_add, 'Enable', 'on');
	end;
	
	% update UI Controls
	update_uicontrols(handles);
	
	% Update handles structure
	guidata(hObject, handles);
end;




function [handles, out] = ManageKeyPress(hObject, handles)
%*************************************************************************
%
% Input: hObject
% Input: handles
%
% Output: handles
% Output: out: 0 -> save data (guidata) and update uicontrol (update_uicontrols)
%
% Description: This functions draws a circle.
%
%*************************************************************************

out = 0;
currKey = get(handles.fig_annotation_tool,'CurrentCharacter');


% switch to annotate mode
if strcmp(currKey, 'a') && ...
	strcmp(get(handles.btn_annotate, 'Enable'), 'on') %&& ...
	%prod(size(findstr(cell2mat(handles.LASTKEYMODI ), 'control')))
	
	handles = startAnnotateMode(handles);
	
	out = 1;	
end;


% (annotate mode) zoom in
if isfield(handles, 'Image') && ...
	strcmp(currKey, 'x') && ...
	handles.ZOOMMODE
	
	% get image size
	imgSize = size(handles.Image);
	
	
	% get display size
	XLim = get(gca,'XLim');
	YLim = get(gca,'YLim');
	
	% check display size
	if abs(XLim(1) - XLim(2)) < 20 || ...
		abs(YLim(1) - YLim(2)) < 20
		
		return;
	end;
	
	% check display size
	if XLim(1) < 1
		XLim(1) = 1;
	else if XLim(2) > imgSize(2)
		XLim(2) = imgSize(2);
	end;
	end;
	
	if YLim(1) < 1
		YLim(1) = 1;
	else if YLim(2) > imgSize(1)
		YLim(2) = imgSize(1);
	end;
	end;
	% END: check display size
	% END: get display size
	
	
	% get selected position
	set(hObject, 'Units', 'pixels');
	output = get(hObject,'CurrentPoint');
	Xc = output(1);
	Yc = output(2);
	set(hObject, 'Units', 'Characters');
	
	
	% get cursor position
	[X, Y] = FigCoord2ImgCoord(handles, Xc, Yc);
	
	
	% check x,y values
	if (X < XLim(1))
		X = XLim(1);
	else if (X > XLim(2))
		X = XLim(2);
	end;
	end;
	
	if (Y < YLim(1))
		Y = YLim(1);
	else if (Y > YLim(2))
		Y = YLim(2);
	end;
	end;
	% end check x,y values
	
	
	% calc new xlim(1) and xlim(2) in x1, x2
	newimgsize1 = (XLim(2) - XLim(1)) * (handles.ZOOM_FACTOR / 2);
	X1 = X - newimgsize1;
	X2 = X + newimgsize1;
	
	if (X1 < XLim(1))
		X1 = XLim(1);
		X2 = XLim(1) + 2* newimgsize1;
	else if (X2 > XLim(2))
		X2 = XLim(2);
		X1 = XLim(2) - 2* newimgsize1;
	end;
	end;
	% END: calc new xlim(1) and xlim(2) in x1, x2
	
	
	% calc new ylim(1) and ylim(2) in y1, y2
	newimgsize2 = (YLim(2) - YLim(1)) * (handles.ZOOM_FACTOR / 2);
	Y1 = Y - newimgsize2;
	Y2 = Y + newimgsize2;
	
	if (Y1 < YLim(1))
		Y1 = YLim(1);
		Y2 = YLim(1) + 2* newimgsize2;
	else if (Y2 > YLim(2))
		Y2 = YLim(2);
		Y1 = YLim(2) - 2* newimgsize2;
	end;
	end;
	% END: calc new ylim(1) and ylim(2) in y1, y2
	
	% set new lim
	set(gca,'XLim', [X1, X2]);
	set(gca,'YLim', [Y1, Y2]);
	
	out = 1;
end;


% (annotate mode) zoom out
if isfield(handles, 'Image') && ...
	strcmp(currKey, 'z') && ...
	handles.ZOOMMODE
	
	% get image size
	imgSize = size(handles.Image);
	
	
	% get display size
	XLim = get(gca,'XLim');
	YLim = get(gca,'YLim');
	
	
	% check display size
	if (XLim(1) == 1) && ...
		(XLim(2) == imgSize(2)) && ...
		(YLim(1) == 1) && ...
		(YLim(2) == imgSize(1))
		
		return;
	end;
	
	
	% check display size
	if XLim(1) < 1
		XLim(1) = 1;
	else if XLim(2) > imgSize(2)
		XLim(2) = imgSize(2);
	end;
	end;
	
	if YLim(1) < 1
		YLim(1) = 1;
	else if YLim(2) > imgSize(1)
		YLim(2) = imgSize(1);
	end;
	end;
	% END: check display size
	% END: get display size
	
	
	% calc new xlim(1) and xlim(2) in x1, x2
	newimgsize1 = ((handles.ZOOM_FACTOR) * (XLim(2) - XLim(1))) /2;
	X1 = XLim(1) - newimgsize1;
	X2 = XLim(2) + newimgsize1;
	
	if (X1 < 1)
		X1 = 1;
		X2 = 1 + XLim(1) + (XLim(2) - XLim(1)) + 2* newimgsize1;
		
	else if (X2 > imgSize(2))
		X2 = XLim(2);
		X1 = imgSize(2) - (XLim(2) - XLim(1)) - 2* newimgsize1;
	end;
	end;
	% END: calc new xlim(1) and xlim(2) in x1, x2
	
	
	% calc new ylim(1) and ylim(2) in y1, y2
	newimgsize2 = ((handles.ZOOM_FACTOR) * (YLim(2) - YLim(1))) /2;
	Y1 = YLim(1) - newimgsize2;
	Y2 = YLim(2) + newimgsize2;
	
	if (Y1 < 1)
		Y1 = 1;
		Y2 = 1 + YLim(1) + (YLim(2) - YLim(1)) + 2* newimgsize2;
	else if (Y2 > imgSize(1))
		Y2 = YLim(2);
		Y1 = imgSize(1) - (YLim(2) - YLim(1)) - 2* newimgsize2;
	end;
	end;
	% END: calc new ylim(1) and ylim(2) in y1, y2
	
	
	% check values
	if X1 < 1
		X1 = 1;
	end;
	if X2 > imgSize(2)
		X2 = imgSize(2);
	end;
	if Y1 < 1
		Y1 = 1;
	end;
	if Y2 > imgSize(1)
		Y2 = imgSize(1);
	end;
	
	
	% set new lim
	set(gca,'XLim', [X1, X2]);
	set(gca,'YLim', [Y1, Y2]);
	
	out = 1;
end;


function handles = zoom_reset(handles)
% resets the image

% get image size
imgSize = size(handles.Image);

set(gca,'XLim', [1, imgSize(2)]);
set(gca,'YLim', [1, imgSize(1)]);


% --- Executes on key press over fig_annotation_tool with no controls selected.
function fig_annotation_tool_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to fig_annotation_tool (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


[handles, out] = ManageKeyPress(hObject, handles);

if out == 1
	
	% update UI Controls
	update_uicontrols(handles);
	
	% Update handles structure
	guidata(hObject, handles);
end;


% --- Executes on key press over lbx_filenames with no controls selected.
function lbx_filenames_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to lbx_filenames (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


[handles, out] = ManageKeyPress(hObject, handles);

if out == 1
	
	% update UI Controls
	update_uicontrols(handles);
	
	% Update handles structure
	guidata(hObject, handles);
end;


% --- Executes on key press over lbx_objects with no controls selected.
function lbx_objects_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to lbx_objects (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


[handles, out] = ManageKeyPress(hObject, handles);

if out == 1
	
	% update UI Controls
	update_uicontrols(handles);
	
	% Update handles structure
	guidata(hObject, handles);
end;

% --- Executes on key press over btn_annotate with no controls selected.
function btn_annotate_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to btn_annotate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


[handles, out] = ManageKeyPress(hObject, handles);

if out == 1
	
	% update UI Controls
	update_uicontrols(handles);
	
	% Update handles structure
	guidata(hObject, handles);
end;

% --- Executes on key press over lbx_parts with no controls selected.
function lbx_parts_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to lbx_parts (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


[handles, out] = ManageKeyPress(hObject, handles);

if out == 1
	
	% update UI Controls
	update_uicontrols(handles);
	
	% Update handles structure
	guidata(hObject, handles);
end;



% --------------------------------------------------------------------
function mn_annotate_Callback(hObject, eventdata, handles)
% hObject    handle to mn_annotate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: This function enables the annotation mode, which allows the user to draw polygons.
%
%*************************************************************************

handles = startAnnotateMode(handles);

% update UI Controls
update_uicontrols(handles);

% Update handles structure
guidata(hObject, handles);


% --------------------------------------------------------------------
function mn_setscale_Callback(hObject, eventdata, handles)
% hObject    handle to mn_setscale (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.MODIFIED = 1;

handles = setscale(handles);

% Update handles structure
guidata(hObject, handles);


function handles = CheckXml(handles)
%
%*************************************************************************
%
% Description: This function checks the xml.
%
%*************************************************************************


% --------------------------------------------------------------------
% get image info, image, xmlfilename

% get current image filename
handles.IMAGEfilename = ...
	handles.filenames{get(handles.lbx_filenames , 'value')};

% read image data
imgInfo = imfinfo([handles.HOMEIMAGES, '/', ...
	handles.directory, '/', handles.IMAGEfilename]);

handles.Image = imread([handles.HOMEIMAGES, '/', ...
	handles.directory, '/', handles.IMAGEfilename]);

% get filename
[pathstr, filename] = fileparts(handles.IMAGEfilename);

% set struct
struct = handles.annotation;



% --------------------------------------------------------------------
% test annotation.filename


if ~isfield(struct, 'filename')
	struct.filename = handles.IMAGEfilename;
	handles.MODIFIED = 1;
end;


% --------------------------------------------------------------------
% test annotation.folder


if ~isfield(struct, 'folder') || ...
	~strcmp(struct.folder, handles.directory)
	
	struct.folder = handles.directory;
	handles.MODIFIED = 1;
end;


% --------------------------------------------------------------------
% test annotation.sourceImage


if ~isfield(struct, 'sourceImage')
	struct.sourceImage = '';
	handles.MODIFIED = 1;
end;


% --------------------------------------------------------------------
% test annotation.sourceAnnotationXML


if ~isfield(struct, 'sourceAnnotationXML') || ...
	~strcmp(struct.sourceAnnotationXML, ...
	['Annotation Tool Version ' handles.version])
	
	handles.MODIFIED = 1;
end;

struct.sourceAnnotationXML = ...
	['Annotation Tool Version ' handles.version];


% --------------------------------------------------------------------
% test annotation.rectified


if ~isfield(struct, 'rectified')
	
	if length(filename) > length('_rect')
		
		struct.rectified = ...
			num2str(strcmp(filename(end-4:end),'_rect'));
	else
		struct.rectified = '0';
	end
	handles.MODIFIED = 1;
end;


% --------------------------------------------------------------------
% test annotation.viewType


if ~isfield(struct, 'viewType')
	struct.viewType = '';
	handles.MODIFIED = 1;
end;


% --------------------------------------------------------------------
% test annotation.scale

if ~isfield(struct, 'scale')
	struct.scale = 'n/a';
	handles.MODIFIED = 1;
end;


% --------------------------------------------------------------------
% test annotation.imageWidth


if ~isfield(struct, 'imageWidth') || ...
	...
	(isstr(struct.imageWidth) && ...
	str2num(struct.imageWidth) ~= imgInfo.Width) || ...
	...
	(isnumeric(struct.imageWidth) && ...
	(struct.imageWidth ~= imgInfo.Width))
	
	struct.imageWidth = num2str(imgInfo.Width);
	handles.MODIFIED = 1;
end;


% --------------------------------------------------------------------
% test annotation.imageHeight


if ~isfield(struct, 'imageHeight') || ...
	...
	(isstr(struct.imageHeight) && ...
	str2num(struct.imageHeight) ~= imgInfo.Height) || ...
	...
	(isnumeric(struct.imageHeight) && ...
	(struct.imageHeight ~= imgInfo.Height))
	
	struct.imageHeight = num2str(imgInfo.Height);
	handles.MODIFIED = 1;
end;


% --------------------------------------------------------------------
% test annotation.transformationMatrix


if ~isfield(struct, 'transformationMatrix') || ...
    strcmp(struct.transformationMatrix, 'n/a')
    
	struct.transformationMatrix = 'n/a';
	handles.MODIFIED = 1;
	
	% check for '_rect'
	% get filename
	[pathstr, name ] = fileparts(handles.IMAGEfilename);

	% check get xml name
	if length(name) >= length('_rect') && ...
		strcmp(name(end-length('_rect') +1:end), '_rect')
		
		xmlfile = [name(1:end-length('_rect')) '.xml'];
	else
		xmlfile = [name '_rect' '.xml'];
	end;
	
    xmlfile = [handles.HOMEANNOTATIONS,'/',...
        handles.directory,'/', xmlfile];
    
	if exist(xmlfile)
		struct_rect = loadXML(xmlfile);
		
		if isfield(struct_rect, 'annotation') && ...
			isfield(struct_rect.annotation, 'transformationMatrix') && ...
			~strcmp(struct_rect.annotation.transformationMatrix, 'n/a')
			
			struct.transformationMatrix = struct_rect.annotation.transformationMatrix;
		end;
	end;
end;


% --------------------------------------------------------------------
% test/set annotation.annotatedClasses


if isfield(struct, 'object') && ...
		length(struct.object) > 0
	
	% get object's classes
	org_class = {};
	for i = 1:length(struct.object)
		org_class{i} = struct.object(i).name;
	end;
	org_class = sort(org_class);
	str = org_class{1};
	class = {str};
	for i = 2:length(org_class)
		if ~strcmp(str, org_class{i})
			class{end +1} = org_class{i};
			str = org_class{i};
		end;
	end;
end;


if ~isfield(struct, 'annotatedClasses')
	
	% get object-classes.txt classes
	tab = readtextfile('object-classes.txt');
	for i=1:size(tab,1),
	    class_txt(i,1) = {deblank( tab(i,:) )};
	end
	class_txt = class_txt';
	
	% check for objects
	if ~isfield(struct, 'object') || ...
		length(struct.object) == 0
		% no objects to add
		struct.annotatedClasses.className = sort(class_txt);
	else
		% add objects to list and delete double
		class = sort([class class_txt]);
		
		str = class{1};
		class_out = {str};
		for i = 2:length(class)
			if ~strcmp(str, class{i})
				class_out{end +1} = class{i};
				str = class{i};
			end;
		end;
		
		struct.annotatedClasses.className = class_out;
	end;
	
	handles.MODIFIED = 1;
	
else
	if ~isfield(struct.annotatedClasses, 'className')
		struct.annotatedClasses.className = {};
	end;
	
	% checkl annotatedClasses
	
	if isfield(struct, 'object') && ...
		length(struct.object) > 0
		
		structClass = struct.annotatedClasses.className;
		% check for cell
		if isstr(structClass)
			% check for empty string
			if length(structClass) == 0
				structClass = {};
			else
				structClass = {structClass};
			end;
		else
			structClass = sort(structClass);
		end;
		
		
		% get class list
		class = sort([structClass class]);
		
		str = class{1};
		class_out = {str};
		for i = 2:length(class)
			if ~strcmp(str, class{i})
				class_out{end +1} = class{i};
				str = class{i};
			else
				handles.MODIFIED = 1;
			end;
		end;
		
		
		struct.annotatedClasses.className = class_out;
	end;
end;


% --------------------------------------------------------------------
% test annotation.object


if ~isfield(struct, 'object')
	
	struct.object = [];
	handles.MODIFIED = 1;
else
	del_list = [];
	
	% check objectID (e.g. convert version 1 data)
	id = round(now*1e8);
	for (i = 1:length(struct.object))
		if (~(isfield(struct.object(i), 'objectID')) || ...
			strcmp(struct.object(i).objectID, '') || ...
			~prod(size((struct.object(i).objectID))))
		
		% save new ids
		handles.MODIFIED = 1;
		
		% objectID composed from date+time to within miliseconds precision
		struct.object(i).objectID = ...
			num2str(id);
		
		id = id+1;
		end;
		
		
		% check polygon for 'pt' part
		if ~isfield(struct.object(i), 'polygon') ...
			|| ~isfield(struct.object(i).polygon, 'pt') ...
			|| ~isfield(struct.object(i).polygon.pt, 'x') ...
			|| ~isfield(struct.object(i).polygon.pt, 'y')
			
			del_list = [del_list i];
        end;
	end;
	
	k = 0;
	for i = 1:length(del_list)
		k = del_list(i);
		struct.object = [struct.object(1:k-1) struct.object(k+1:end)];
	end;
	
end;


% --------------------------------------------------------------------

% sort struct
struct = orderfields(struct, handles.DTDORDER_ANNOTATION);

% set struct
handles.annotation = struct;



% --------------------------------------------------------------------
function [structClass, out] = openClassWindow(handles)
%*************************************************************************
%
% Description: Opens a window to select classes.
%
%*************************************************************************

structClass = {};
out = 1; % 0 -> ok, 1 -> cancel

% get class list 'object-classes.txt'
tab = readtextfile('object-classes.txt');
for i=1:size(tab,1),
    class_txt(i,1) = {deblank( tab(i,:) )};
end


% get object class list
class_obj = {};
if ~length(handles.annotation.object) > 0
else
	for i = 1:length(handles.annotation.object)
		class_obj{end +1} = handles.annotation.object(i).name;
	end;
end;
class_obj = sort(class_obj);


% delete double (for later)
if length(class_obj) > 0
	str = class_obj{1};
	i = 2;
	while i <= length(class_obj) % MD (=)
		
		if strcmp(str, class_obj{i})
			class_obj = [class_obj(1:i-1) class_obj(i+1:end)];
			continue;
		end;
		
		str = class_obj{i};
		i = i+1;
	end;
end;

% make sure className is a cell array (className is not a cell
% array if there is a single className in annotatedClasses)
if ~iscell(handles.annotation.annotatedClasses.className),
    handles.annotation.annotatedClasses.className = ...
        {handles.annotation.annotatedClasses.className};
end        

% get xml class list
class_xml = handles.annotation.annotatedClasses.className;

% compare them
class = [class_txt' class_xml];
class = sort(class);

if length(class) > 0
	str = class{1};
	i = 2;
	while i <= length(class)
		
		if strcmp(str, class{i})
			class = [class(1:i-1) class(i+1:end)];
			continue;
		end;
		
		str = class{i};
		i = i+1;
	end;
end;


% --------------------------------------------------------------------
% get checked and obj values
class_checkedValue = zeros(1, length(class));
class_objValue = zeros(1, length(class));

if length(class) > 0
	m = 1;
	n = 1;
	
	for i = 1:length(class)
		
		if length(class_xml) >= m && ...
			strcmp(class{i}, class_xml{m})
			
			class_checkedValue(i) = 1;
			m = m +1;
		end;
		
		if length(class_obj) >= n  && ...
			strcmp(class{i}, class_obj{n})
			
			class_objValue(i) = 1;
			n = n +1;
		end;
	end;
end;


% --------------------------------------------------------------------


% setup setting (CHANGE JUST THESE ONES)
itemHeight = 20;		% checkbox height
itemSpace = 5;			% space betweeen checkboxes
itemBorder = 20;		% border between items and window border
btnHeight = 20;			% height of the buttons
windowWidth = 250;		% window width
bgColor = [0.8784313725490196 ...
	0.8745098039215686 ...
	0.8901960784313725];	% background color
itemInactiveColor = 0.5;	% color of the inactive items


% window size
windowHeight = (itemSpace + itemHeight) * length(class) + ...
	itemBorder * 2 + btnHeight;


screenSize = get(0,'ScreenSize');

% figure position / size
fig_left = screenSize(1) + screenSize(3) / 2 - windowWidth / 2;
fig_bottom = screenSize(2) + screenSize(4) / 2 - windowHeight / 2;
fig_width = windowWidth;
fig_height = windowHeight;

% checkbox position / size
chb_left = itemBorder;
chb_bottom = windowHeight - itemHeight / 2;
chb_width = windowWidth - itemBorder * 2;
chb_height = itemHeight;


% figure
fig_acDialog = figure('WindowStyle', 'modal', ...
	'Name', 'Set Classes', ...
	'NumberTitle', 'off', ...
	'Position', [fig_left, fig_bottom, fig_width, fig_height], ...
	'Resize', 'off', ...
	'Color', bgColor);

% checkboxes
chbx = [];

for i = 1:length(class)
	
	chbx(i) = uicontrol('Parent', fig_acDialog, ...
		'Style', 'checkbox',...
		'Tag', ['chbx' num2str(i)],...
		'String', class{i},...
		'Position', ...
			[chb_left ...
			chb_bottom - (itemSpace + itemHeight) * i...
			chb_width ...
			chb_height], ...
		'Value', class_checkedValue(i), ...
		'BackgroundColor', bgColor);
	
	if class_objValue(i)
		set(chbx(i), 'ForegroundColor', ...
			[itemInactiveColor ...
			itemInactiveColor ...
			itemInactiveColor]);
		set(chbx(i), 'Enable', 'inactive');
	end;
end;

% btn OK
btn = uicontrol('Parent', fig_acDialog, ...
	'Position', ...
		[itemBorder ...
		itemBorder ...
		windowWidth/2 - 1.5*itemBorder ...
		btnHeight], ...
	'String', 'OK', ...
	'Callback', 'uiresume(gcbf)');

% btn cancel
btn = uicontrol('Parent', fig_acDialog, ...
	'Position', ...
		[windowWidth/2 + 0.5 * itemBorder...
		itemBorder ...
		windowWidth/2 - 1.5*itemBorder ...
		btnHeight], ...
	'String', 'Cancel', ...
	'Callback', 'close(gcbf)');


uiwait(fig_acDialog);

% check values
class_value = [];
try
	for i = 1:length(class)
		class_value(i) = get(chbx(i), 'Value');
	end;
	% set save flag
	handles.MODIFIED = 1;
	
	close(fig_acDialog);
	out = 0; % ok pressed
catch
	out = 1; % cancel pressed
	return;
end;


% get new class struct
structClass = {};
for i = 1:length(class)
	
	if class_value(i)
		structClass{end +1} = class{i};
	end;
end;



% --------------------------------------------------------------------
function mn_defAnnoClasses_Callback(hObject, eventdata, handles)
% hObject    handle to mn_defAnnoClasses (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: This function opens the annotation classes dialog.
%
%*************************************************************************

[structClass, out] = openClassWindow(handles);
if out
	return;
end;


% finally set new values
handles.annotation.annotatedClasses.className = structClass;

set(handles.popup_object_name, 'String', ...
	[{''} structClass {'other...'}]);

% Update handles structure
guidata(hObject, handles);


% --------------------------------------------------------------------
function imagepanel_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to imagepanel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function uipanel5_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to uipanel5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function newannotpanel_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to newannotpanel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



% --------------------------------------------------------------------
function mn_defDsetAnnoClasses_Callback(hObject, eventdata, handles)
% hObject    handle to mn_defDsetAnnoClasses (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%
%*************************************************************************
%
% Description: This function opens the annotation classes dialog (for the whole Dataset).
%
%*************************************************************************


[structClass, out] = openClassWindow(handles);
if out
	return;
end;


% ------------------------------------------------------------------------
% finally set new values for the dataset

for i = 1:length(handles.filenames)
	
	% check for current xml -> continue
	if strcmp(handles.filenames{i}, ...
		handles.annotation.filename)
		continue;
	end;
	
	% get filename -> filename
	[pathstr, name ] = fileparts(handles.filenames{i});
	filename = [handles.HOMEANNOTATIONS, '/', ...
		handles.directory, '/', name,'.xml'];
	
	if ~exist(filename)
		continue;
	end;
	
	struct = loadXML(filename);
	
	% get object classes -> class
	if isfield(struct.annotation, 'object') && ...
			length(struct.annotation.object) > 0
		
		% get object's classes
		org_class = {};
		for i = 1:length(struct.annotation.object)
			org_class{i} = struct.annotation.object(i).name;
		end;
		org_class = sort(org_class);
		str = org_class{1};
		class = {str};
		for i = 2:length(org_class)
			if ~strcmp(str, org_class{i})
				class{end +1} = org_class{i};
				str = org_class{i};
			end;
		end;
	else
		class = {};
	end;
	
	
	% set both classes and delete double
	class = sort([class structClass]);
	
	if length(class) > 0
	
		str = class{1};
		class_out = {str};
		for i = 2:length(class)
			if ~strcmp(str, class{i})
				class_out{end +1} = class{i};
				str = class{i};
			end;
		end;
	end;
	
	
	% set new classes
	struct.annotation.annotatedClasses.className = ...
		class_out;
	
	
	% write xml
	writeXML(filename, struct);
	clear('struct');
end;


% ------------------------------------------------------------------------
% set new values for this xml
handles.annotation.annotatedClasses.className = structClass;
handles.MODIFIED = 1;

set(handles.popup_object_name, 'String', ...
	[{''} structClass {'other...'}]);

% Update handles structure
guidata(hObject, handles);


% --------------------------------------------------------------------
function mn_validateFiles_Callback(hObject, eventdata, handles)
% hObject    handle to mn_validateFiles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



ans = questdlg(['Are you sure, you want to ' ...
	'validate the Dataset?'], ...
	'Validate Annotation Files', 'No');

if ~strcmp(ans, 'Yes')
	return;
end;


% ------------------------------------------------------------------------
% loop files



filenames = {};

for i = 1:length(handles.filenames)
	
	% get filenames
	[pathstr, name ] = fileparts(handles.filenames{i});
	filename = [handles.HOMEANNOTATIONS, '/', ...
		handles.directory, '/', name,'.xml'];
	
	if ~exist(filename)
		continue;
	end;
	
	filenames{end +1} = filename;
end;

changes = parseXML(filenames, handles.DTDFILENAME);


if length(changes) <= 0
	
	text = 'Validation successful!';
else
	text{1} = 'Update and validation successful!';
	text{2} = 'Updating following files:';
	
	for i = 1:length(changes)
		text{3+i} = changes{i};
	end;
end;

msgbox(text, 'Validate Annotation Files', 'help');

% --------------------------------------------------------------------
function mn_objects_Callback(hObject, eventdata, handles)
% hObject    handle to mn_objects (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function mn_sort1_Callback(hObject, eventdata, handles)
% hObject    handle to mn_sort1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% prepare for sorting the names
name = [];
for i=1:length(handles.annotation.object),
    name = char(name,handles.annotation.object(i).name);
end
name = name(2:end,:);
    
% sort the names
[B,index] = sortrows(name);

% sort annotated objects
for i=1:length(handles.annotation.object),
    sorted_obj(i) = handles.annotation.object(index(i));
end
handles.annotation.object = sorted_obj;

% update current annotations list
for i=1:length(handles.annotation.object),
    handles.class{i} = deblank(B(i,:));
end
set(handles.lbx_objects, 'Value', 1);
set(handles.lbx_objects, 'String', handles.class);

% enable
set(handles.mn_save,'Enable','on');
handles.MODIFIED = 1;

% update handles-structure
guidata(hObject, handles)


% --------------------------------------------------------------------
function handles = ParameterCall(hObject, eventdata, ...
	handles, imgPath, filename);
%
%*************************************************************************
%
% Description: This function handles the parameter input. ...
%	For example use: AnnotationTool('eoh-classification', '037_34A.jpg')
%
%*************************************************************************

handles.directory = imgPath;

if ~exist([handles.HOMEIMAGES,'/', handles.directory])
	return;
end;

% get list of filenames
handles.files = dir([handles.HOMEIMAGES,'/', handles.directory]);

handles.filenames = {};
k = 1;
for i=3:length(handles.files),
	if length(handles.files(i).name) > 0 && ...
		~strcmp(handles.files(i).name, '.svn')

		handles.filenames{end+1}=handles.files(i).name;
		if strcmp(handles.filenames{end}, filename)
			k = length(handles.filenames);
		end;
	end;
end;

set(handles.lbx_filenames,'String',handles.filenames);
set(handles.lbx_filenames, 'Value',k);

% enable filename listbox
set(handles.lbx_filenames,'Enable','on');

% disenable filename listbox
set(handles.mn_aggregation,'Enable','off');

handles = lbx_filenames_Callback(hObject, eventdata, handles);

% update handles
guidata(hObject, handles);


% --------------------------------------------------------------------
function mn_browseclasses_Callback(hObject, eventdata, handles)
% hObject    handle to mn_browseclasses (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% still in AnnotateMode?
if (strcmp(get(handles.btn_add, 'Enable'), 'on') && ...
	~strcmp(get(handles.btn_annotate, 'String'), 'New Annotation'))

	handles.currObject = '';
	set(handles.btn_add, 'Enable', 'off');
end;

if length(handles.annotation.object) > 0
	
	i = get(handles.lbx_objects,'Value');
	
	handles = update_panel_annotation(handles, i);
end;


if strcmp(get(hObject,'Checked'),'on')
   set(hObject,'Checked','off');
   handles.BROWSECLASSES = 0;
else
    set(hObject,'Checked','on');
	handles.BROWSECLASSES = 1;
end


if handles.BROWSECLASSES
	
	% start browseclasses mode
	
	set(handles.uipanel5, 'Title', 'Current Classes');
	set(handles.txt_objID, 'String', '');
	set(handles.lbx_parts, 'Enable', 'off');
	set(handles.txt_objects, 'String', 'Class Names:');
	set(handles.lbx_objects, 'Value', 1);
	
	% reset the objects lbx
	classes = sort(get(handles.lbx_objects, 'String'));
	n = length(classes) -1;
	i = 1;
	while i <= n
		if strcmp(classes{i}, classes{i+1})
			classes = [classes(1:i) ; classes(i+2:end)];
			n = length(classes) -1;
		else
			i = i+1;
		end;
	end;
	set(handles.lbx_objects, 'String', classes);
else
    
	% end browseclasses mode
	
	handles = EndBrowseClassesMode(handles);
end;

% update UI Controls
update_uicontrols(handles);


% update handles
guidata(hObject, handles);


% --------------------------------------------------------------------
function handles = EndBrowseClassesMode(handles)

% end browseclasses mode

handles.BROWSECLASSES = 0;

set(handles.mn_browseclasses,'Checked','off');

set(handles.uipanel5, 'Title', 'Current Objects');
set(handles.txt_objID, 'String', 'no selection');
set(handles.lbx_parts, 'Enable', 'off');
set(handles.txt_objects, 'String', 'Object Class Names:');

% reset the objects lbx
set(handles.lbx_objects, 'String', handles.class);


% --------------------------------------------------------------------
function mn_add_part_auto_Callback(hObject, eventdata, handles)
% FUNCTION implemented by Martin Drauschke (MD)
% hObject    handle to mn_add_part_auto (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% dialogue, which classes should be considered
structClass = openPartClassWindow(handles);

curr_id = handles.currObject; % identifier of annotation
nb_objects = length(handles.annotation.object);
%disp(['Chosen Object: ID #',num2str(curr_id),' of ',num2str(nb_objects)]);

% check, if current object has already some parts, then delete them.
% 1st CHECK concerns the data in the GUI (window with parts)
if prod(size(handles.LBX_PARTS_NUM))
    % we have some parts already - remove them
	% update NUM
	handles.LBX_PARTS_NUM = [];
	content_parts = ''; % new list of part names
	% update lbx_parts
	set(handles.lbx_parts,'Value',1);
	set(handles.lbx_parts,'String', content_parts);
end;
% 2nd CHECK concerns the data in the annotation struct which is used for
% writing the xml file
if isfield(handles.annotation.object(curr_id),'objectParts')
    handles.annotation.object(curr_id).objectParts = '';
    % list of objetc parts exists, so reset it as empty.
end;

% Now we are sure, that list of parts is empty.
% Check, which of the other annotations intersect the current one.
[curr.x,curr.y] = getLMpolygon(handles.annotation.object(curr_id).polygon);
% border chain of current object - where we look for its parts
new_IDs = []; % we don't know if any parts are there
current_parts = [];
field_of_indices = [];
for iter_obj = 1:nb_objects,
    if iter_obj ~= curr_id
        % check class membership
        classlabel = handles.annotation.object(iter_obj).name;
        is_in_list = false;
        for i=1:length(structClass),
            if strcmp(classlabel,structClass{i})
                is_in_list = true;
            end;
        end;
        if is_in_list
            [iter.x,iter.y] = getLMpolygon(handles.annotation.object(iter_obj).polygon);
            iter_area = polyarea(iter.x,iter.y);
            intersection = PolygonClip(curr,iter,1);
            nb_sections = size(intersection,2);
            sect_area = 0;
            for iter_sect = 1:nb_sections,
                a = polyarea(intersection(1,iter_sect).x,intersection(1,iter_sect).y);
                sect_area = sect_area + a;
            end;
            ab =  sect_area / iter_area;
            if ab > 0.95 
                % object with id iter_obj is likely included by curr_id, so
                % it's a part of it
                %disp(['Object ',num2str(iter_obj),' is part of object ',num2str(curr_id)]);
                field_of_indices(end+1,1) = iter_obj;
                new_IDs(end+1,1) = str2num(handles.annotation.object(iter_obj).objectID);
                current_parts{end+1} = handles.annotation.object(iter_obj).name;
            end;
        end;
    end;
end;
% Set list to global data
% convert matrix to a string in Matlab syntax
handles.annotation.object(curr_id).objectParts = mat2str(new_IDs);
set(handles.lbx_parts, 'String', current_parts);
for i=1:length(new_IDs),
    handles.LBX_PARTS_NUM(end +1) = field_of_indices(i,1);
    % add object ID to LBX_PARTS_NUM
end;

% update UI controls
update_uicontrols(handles);

% update handles struct
guidata(hObject,handles);

% --------------------------------------------------------------------
function structClass = openPartClassWindow(handles)
% FUNCTION implemented by Martin Drauschke (MD), based on the function
% openClassWindow by Filip Korc and David Schneider

% All Classes are enabled, and not checked, so that the user must choose.
% If dialog is aborted or nothing chosen by user, we take all classes.
structClass = {};
out = 1; % 0 -> ok, 1 -> cancel

% get object class list (what has been annotated so far)
class_obj = {};
if ~length(handles.annotation.object) > 0
else
	for i = 1:length(handles.annotation.object)
		class_obj{end +1} = handles.annotation.object(i).name;
	end;
end;
class_obj = sort(class_obj);

% delete duplicates (copies)
if ~isempty(class_obj)
	str = class_obj{1};
	i = 2;
	while i <= length(class_obj)
		if strcmp(str,class_obj{i})
			class_obj = [class_obj(1:i-1) class_obj(i+1:end)];
			continue;
		end;
		str = class_obj{i};
		i = i+1;
	end;
end;

% all classes are unchecked
class_checkedValue = zeros(1, length(class_obj));
class_objValue = zeros(1, length(class_obj)); % zeros mean unticked

%%%%%%% GUI - graphical implementation of window
% setup setting (CHANGE JUST THESE ONES)
itemHeight = 20;		% checkbox height
itemSpace = 5;			% space betweeen checkboxes
itemBorder = 20;		% border between items and window border
btnHeight = 20;			% height of the buttons
windowWidth = 250;		% window width
bgColor = [0.8784313725490196 ...
	0.8745098039215686 ...
	0.8901960784313725];	% background color
itemInactiveColor = 0.5;	% color of the inactive items

% window size
windowHeight = (itemSpace + itemHeight) * length(class_obj) + ...
	itemBorder * 2 + btnHeight;


screenSize = get(0,'ScreenSize');

% figure position / size
fig_left = screenSize(1) + screenSize(3) / 2 - windowWidth / 2;
fig_bottom = screenSize(2) + screenSize(4) / 2 - windowHeight / 2;
fig_width = windowWidth;
fig_height = windowHeight;

% checkbox position / size
chb_left = itemBorder;
chb_bottom = windowHeight - itemHeight / 2;
chb_width = windowWidth - itemBorder * 2;
chb_height = itemHeight;


% figure
fig_acDialog = figure('WindowStyle', 'modal', ...
	'Name', 'Select Classes of Parts', ...
	'NumberTitle', 'off', ...
	'Position', [fig_left, fig_bottom, fig_width, fig_height], ...
	'Resize', 'off', ...
	'Color', bgColor);

% checkboxes
chbx = [];

for i = 1:length(class_obj)
	chbx(i) = uicontrol('Parent', fig_acDialog, ...
		'Style', 'checkbox',...
		'Tag', ['chbx' num2str(i)],...
		'String', class_obj{i},...
		'Position', ...
			[chb_left ...
			chb_bottom - (itemSpace + itemHeight) * i...
			chb_width ...
			chb_height], ...
		'Value', class_checkedValue(i), ...
		'BackgroundColor', bgColor);
	
	if class_objValue(i)
		set(chbx(i), 'ForegroundColor', ...
			[itemInactiveColor ...
			itemInactiveColor ...
			itemInactiveColor]);
		set(chbx(i), 'Enable', 'inactive');
	end;
end;

% btn OK
btn = uicontrol('Parent', fig_acDialog, ...
	'Position', ...
		[itemBorder ...
		itemBorder ...
		windowWidth/2 - 1.5*itemBorder ...
		btnHeight], ...
	'String', 'OK', ...
	'Callback', 'uiresume(gcbf)');

% btn cancel
btn = uicontrol('Parent', fig_acDialog, ...
	'Position', ...
		[windowWidth/2 + 0.5 * itemBorder...
		itemBorder ...
		windowWidth/2 - 1.5*itemBorder ...
		btnHeight], ...
	'String', 'Cancel', ...
	'Callback', 'close(gcbf)');


uiwait(fig_acDialog);

% check values
class_value = zeros(1, length(class_obj));
try
	for i = 1:length(class_obj)
		class_value(i) = get(chbx(i), 'Value');
	end;
	% set save flag
	handles.MODIFIED = 1;
	
	close(fig_acDialog);
	out = 0; % ok pressed
catch
	out = 1; % cancel pressed
end;

if max(class_value) == 0,
    % all classes stayed unticked, so we take all classes instead
    structClass = class_obj;
else
    % get new class struct
    structClass = {};
    for i = 1:length(class_obj)
        if class_value(i)
            structClass{end +1} = class_obj{i};
        end;
    end;
end;


