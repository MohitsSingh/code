function changes = parseXML(xmlFilenames, dtdFilename)
%
% ParseXML validates the tag structure of a xml file using a Document Type Definition.
%
% This function analyses the dtd file. These informations are used to validate the xml
% structure and in the case of missing or misplaced tags the xml file will be modified 
% and replaced.
%
% xmlFilename:	The filenames of the xml file to analyse (cell or string).
% dtdFilename:	The filename of the dtd file.
% changes:		includes all filenames, that have been changed
%
% Author:	David Schneider (schneide@informatik.uni-bonn.de)
% Date:	August '07
%
% This program is a part of the Annotation Tool developed at the Department 
% of Photogrammetry, University of Bonn (http://www.ipb.uni-bonn.de/~filip/annotation-tool/).
%


% init value
changes = {};

if isstr(xmlFilenames)
	xmlFilenames = {xmlFilenames};
end;

if ~iscell(xmlFilenames)
	return;
end;


% read dtd file
[out, textDTD] = readFile(dtdFilename);
if out ~= 0
	return;
end;

dtdStructName = 'annotation';
[out, dtdStruct, msg] = analyseDTD(textDTD, dtdStructName);
% dtdStruct entry-index = 4

if (out ~= 0)
	return;
end;


for i = 1:length(xmlFilenames)
	
	xmlFilename = xmlFilenames{i};
	
	
	% load xml file
	try
		xmlStruct = loadXML(xmlFilename);
	catch
		continue;
	end
	
	
	% analyse textXML with dtdStruct
	[outputStruct, changed] = analyseXML(dtdStruct, xmlStruct);
	
	if changed
		tempOut = writeFile(xmlFilename, outputStruct);
		
		changes{end +1} = xmlFilename;
	end;
end;


% /////////////////////////////////////////////////////////////////////////////////////
%	reads a text file
function [out, text] = readFile(filename)

out = -1;
text = 'no assignment';

[fid, message] = fopen(filename,'r');
if fid == -1
	text = message;
	out = -1;
	return;
end

text = fread(fid, 'uint8=>char');
fclose(fid);

% Remove 'new line' characters from the chain
text = char(text(:)');
text = strrep(text, char(10), '');
text = strrep(text, char(13), '');

out = 0;


% /////////////////////////////////////////////////////////////////////////////////////
%	writes a text file
function out = writeFile(filename, struct);

writeXML(filename, struct);

out = 0;


% /////////////////////////////////////////////////////////////////////////////////////
% return the next whole word
function [word, rest] = getNextWord(text)

% delete first spaces
while strcmp(text(1), ' ') && (length(t) > 0)
	text = text(2:end);
end;

if length(text) <= 0
	word = '';
	rest = '';
end;

% find spaces
spaces = findstr(' ', text);

if ~prod(size(spaces))
	word = text;
	rest = '';
	return;
end;

% define word and redefine text
word = text(1:spaces(1) -1);
rest = text(spaces(1) +1:end);

% check rest
while strcmp(rest(1), ' ') && (length(rest) > 0)
	rest = rest(2:end);
end;


% /////////////////////////////////////////////////////////////////////////////////////
% analyse a dtd structure
function [out, struct, msg] = analyseDTD(text, start);

% initialisation
out = -1;
msg = 'no assignment';


text = deleteComment(text);
text = clearSpaces(text);



index = 1;
struct(index).name = '(#PCDATA)';
struct(index).down = 0;
struct(index).next = 0;
struct(index).repeat = 0;
struct(index).repeatType = '';
index = index +1;

struct(index).name = 'ANY';
struct(index).down = 0;
struct(index).next = 0;
struct(index).repeat = 0;
struct(index).repeatType = '';
index = index +1;

struct(index).name = 'EMPTY';
struct(index).down = 0;
struct(index).next = 0;
struct(index).repeat = 0;
struct(index).repeatType = '';
index = index +1;


[out, struct, index, msg] = subAnalyseDTD(text, start, struct, index);


% /////////////////////////////////////////////////////////////////////////////////////
% analyse single entry
function [out, struct, index, msg] = subAnalyseDTD(text, element, struct, index);

out = -1;
msg = 'no assignment';


instanceIndex = index;

ELEM = '<!ELEMENT';

elemStart = findstr([ELEM ' ' element ' '], text);


if ~prod(size(elemStart))
	out = -1;
	msg = 'incorrect dtd';
	return;
end;

elemStart = elemStart(1);

elemEnd = findstr(text(elemStart:end), '>');
elemEnd = elemStart - 1 + elemEnd(1);

elemTypeText = text(elemStart + length([ELEM ' ' element]):elemEnd -1);

elemTypeStruct = getElemData(elemTypeText);


if ~prod(size(elemTypeStruct))
	out = -1;
	msg = 'incorrect dtd';
	return;
end;


struct(instanceIndex).name = element;
struct(instanceIndex).down = 0;
struct(instanceIndex).next = 0;
struct(instanceIndex).repeat = 0;
struct(instanceIndex).repeatType = '';


lastIndex = index;


bracket = [];

% check elemTypeStruct
if strcmp(elemTypeStruct{1}, '(#PCDATA)')
	struct(instanceIndex).down = 1;
	
else if strcmp(elemTypeStruct{1}, 'ANY')
	struct(instanceIndex).down = 2;
	
else if strcmp(elemTypeStruct{1}, 'EMPTY')
	struct(instanceIndex).down = 3;
	
else
	for i = 1:prod(size(elemTypeStruct))
		
		if strcmp(elemTypeStruct{i}, '(')
			bracket = [bracket; i];
		
		
		else if strcmp(elemTypeStruct{i}, ')*')
			struct(lastIndex).repeat = bracket(end);
			struct(lastIndex).repeatType = '*';
			bracket = bracket(1:end -1);
		
		else if strcmp(elemTypeStruct{i}, ')+')
			struct(lastIndex).repeat = bracket(end);
			struct(lastIndex).repeatType = '+';
			bracket = bracket(1:end -1);
		
		else if strcmp(elemTypeStruct{i}, ')')
			bracket = bracket(1:end -1);		
		
		else if strcmp(elemTypeStruct{i}, '*')
			struct(lastIndex).repeat = lastIndex;
			struct(lastIndex).repeatType = '*';
		
		else if strcmp(elemTypeStruct{i}, '+')
			struct(lastIndex).repeat = lastIndex;
			struct(lastIndex).repeatType = '+';
		
		
		% real element
		else
			
			index = index +1;
			if (struct(instanceIndex).down == 0)
				struct(instanceIndex).down = index;
				lastIndex = index;
			
			else
				struct(lastIndex).next = index;
				lastIndex = index;
			end;
			
			[out, struct, index, msg] = subAnalyseDTD(text, elemTypeStruct{i}, struct, index);
			
			if (out ~= 0)
				return;
			end;			
		end;
		
		end;
		
		end;
	end;
end;
end;
end;
end;
end;
end;

out = 0;
msg = '';


% /////////////////////////////////////////////////////////////////////////////////////
% analyse a dtd structure
function out = deleteComment(text);

out = '';

% look for comments
commentStart = findstr(text, '<!--');

while prod(size(commentStart))
	
	commentEnd = findstr(text, '-->');
	
	% no correct closed comment
	if ~prod(size(commentEnd))
		out = text(1:commentStart(1) -1);
		return;
	end;
	
	text = strcat(text(1:commentStart(1) -1), text(commentEnd(1) + length('-->'):end));
	
	commentStart = findstr(text, '<!--');
end;


out = text;


% /////////////////////////////////////////////////////////////////////////////////////
% delete double spaces
function out = clearSpaces(text);

spaces = findstr(text, ' ' );

i = 1;
j = 0;
k = 0;

while (i < prod(size(spaces)))
	
	
	% if there are two spaces
	if spaces(i) == spaces(i +1) -1
		
		% check for more in this position
		j = i+1;
		k = 1;
		while (j <= prod(size(spaces))) && ...
			(spaces(i) == spaces(j) -k)
			j = j +1;
			k = k +1;
		end;
		
		% delete them
		text = strcat(text(1:spaces(i)), text(spaces(j -1):end));
		
		spaces = findstr(text, ' ');
		
		% continue
		i = i +1;
	end;
	
	i = i +1;
end;

out = text;


% /////////////////////////////////////////////////////////////////////////////////////
% return a struct of the elem Data
function struct = getElemData(text);

i = 1;

if prod(size(findstr(text, '(#PCDATA)')))
	
	struct{i} = '(#PCDATA)';

else if prod(size(findstr(text, 'ANY')))
	
	struct{i} = 'ANY';
	
else if prod(size(findstr(text, 'EMPTY')))
	
	struct{i} = 'EMPTY';
	
else
	text = strrep(text, ' ', '');
	
	[elem, rest] = getSubElem(text);
	while prod(size(rest))
		struct{i} = elem; i = i +1;
		[elem, rest] = getSubElem(rest);
	end;
	struct{i} = elem; i = i +1;
end;
end;
end;


% /////////////////////////////////////////////////////////////////////////////////////
% return a sub element
function [elem, rest] = getSubElem(text);

elem = '';
rest = '';

if length(text) > 0

	if strcmp(text(1), ',')
		text = text(2:end);
	end;

	if strcmp(text(1), '(')
		elem = '(';
		rest = text(2:end);

	else if length(text) > 2 && ...
		strcmp(text(1:2), ')*')
		
		elem = ')*';
		rest = text(3:end);
		
	else if length(text) > 2 && ...
		 strcmp(text(1:2), ')+')
		 
		elem = ')+';
		rest = text(3:end);
		
	else if strcmp(text(1), ')')
		elem = ')';
		rest = text(2:end);
		
	else if strcmp(text(1), '*')
		elem = '*';
		rest = text(2:end);
		
	else if strcmp(text(1), '+')
		elem = '+';
		rest = text(2:end);

	else
		i = 1;
		
		while ~strcmp(text(i), ',') && ...
			~strcmp(text(i), '(') && ...
			~strcmp(text(i), ')') && ...
			~strcmp(text(i), '+') && ...
			~strcmp(text(i), '*')
			i = i +1;
		end;
		
		elem = text(1:i-1);
		rest = text(i:end);
	end;
	end;
	end;
	end;
	end;
	end;
end;


% /////////////////////////////////////////////////////////////////////////////////////
% analyse a xml text with a dtdStruct
function [outputStruct, changed] = analyseXML(dtdStruct, xmlStruct);

changed = 0;


pointer = 4;
done = 0;
currentLevel = {};


% get current level
subPointer = pointer;
done = 0;
while ~done
	
	% first check repeat type
	if ~strcmp(dtdStruct(subPointer).repeatType, '*') || ...
		isfield(xmlStruct, dtdStruct(subPointer).name)
		
		currentLevel{end +1} = dtdStruct(subPointer).name;
	end;
	
	subPointer = dtdStruct(subPointer).next;
	if ~subPointer
		done = 1;
	end;
end;


% get current fields
if isstruct(xmlStruct)
	currentfields = fieldnames(xmlStruct);
	
	
	if ~strcmp(char(currentfields), char(currentLevel))
		changed = 1;
	end;
end;


% check current level
for i = 1:length(currentLevel)
	
	if ~isfield(xmlStruct, (currentLevel{i}))
		xmlStruct.(currentLevel{i}) = [];
		changed = 1;
	end;
	
	outputStruct.(currentLevel{i}) = [];
end;

% outputStruct now includes the sorted current level and older
outputStruct = orderfields(outputStruct, currentLevel);


% loop sub levels
for i = 1:length(currentLevel)
	
	% if struct
	if (dtdStruct(pointer).down > 3)
		
		% case: empty (*) substruct
		if ~isfield(xmlStruct, (currentLevel{i})) && ...
			strcmp(dtdStruct(dtdStruct(pointer.down)).repeatType, '*')
			
			if ~isstr(xmlStruct.(currentLevel{i})) || ...
				strcmp(xmlStruct.(currentLevel{i}), '')
				changed = 1;
			end;
			
			xmlStruct.(currentLevel{i}) = '';
			pointer = dtdStruct(pointer).next;
			continue;
		end;
		
		k = 1;
		if ((strcmp(dtdStruct(pointer).repeatType, '+') || ...
                strcmp(dtdStruct(pointer).repeatType, '*'))) && ...
                (length(xmlStruct.(currentLevel{i})) > 1)
			
            for j = 1:length(xmlStruct.(currentLevel{i}))

                if j <= 1
                    [outputStruct.(currentLevel{i}), out] = ...
                        subAnalyseXML(dtdStruct, ...
                        xmlStruct.(currentLevel{i})(j), dtdStruct(pointer).down);
                else
                    [outputStruct.(currentLevel{i})(j), out] = ...
                        subAnalyseXML(dtdStruct, ...
                        xmlStruct.(currentLevel{i})(j), dtdStruct(pointer).down);
                end;
				
				if out && ~changed
					changed = 1;
				end;
            end;
        else
			
            [outputStruct.(currentLevel{i}), out] = ...
				subAnalyseXML(dtdStruct, ...
				xmlStruct.(currentLevel{i}), dtdStruct(pointer).down);
    
            if out && ~changed
				changed = 1;
			end;
		end;
    else
        k = 1;
		if (strcmp(dtdStruct(pointer).repeatType, '+') || strcmp(dtdStruct(pointer).repeatType, '*'))
			
			k = length(xmlStruct.(currentLevel{i}));
		end;
        
        if isfield(xmlStruct, currentLevel{i}) && ...
			iscell(xmlStruct.(currentLevel{i}))
			
            for j = 1:k
                [value, out] = ...
                    checkValues(xmlStruct.(currentLevel{i}){j}, dtdStruct(pointer).down);
				
				outputStruct.(currentLevel{i}){j} = value;
				
				if out && ~changed
					changed = 1;
				end;
            end;
        else
            [outputStruct.(currentLevel{i}), out] = ...
				checkValues(xmlStruct.(currentLevel{i}), dtdStruct(pointer).down);
			
			if out && ~changed
					changed = 1;
			end;
        end;
	end;
	
    pointer = dtdStruct(pointer).next;
end;


% /////////////////////////////////////////////////////////////////////////////////////
% subfunction for analyse a xml text with a dtdStruct
function [outputStruct, changed] = subAnalyseXML(dtdStruct, xmlStruct, pointer);

changed = 0;

done = 0;
currentLevel = {};


% get current level
subPointer = pointer;
done = 0;
while ~done
    
	% first check repeat type
	if ~strcmp(dtdStruct(subPointer).repeatType, '*') || ...
		isfield(xmlStruct, dtdStruct(subPointer).name)
		
		currentLevel{end +1} = dtdStruct(subPointer).name;
	end;
	
	subPointer = dtdStruct(subPointer).next;
	if ~subPointer
		done = 1;
	end;
end;


% get current fields
if isstruct(xmlStruct)
	currentfields = fieldnames(xmlStruct);
	
	
	if ~strcmp(char(currentfields), char(currentLevel))
		changed = 1;
	end;
end;


% check current level
for i = 1:length(currentLevel)
	
	if ~isfield(xmlStruct, (currentLevel{i}))
		xmlStruct.(currentLevel{i}) = [];
		changed = 1;
	end;
	
	outputStruct.(currentLevel{i}) = [];
end;


% outputStruct now includes the sorted current level and older
outputStruct = orderfields(outputStruct, currentLevel);


% loop sub levels
for i = 1:length(currentLevel)
	
	% if struct
	if (dtdStruct(pointer).down > 3)
        
		% case: empty (*) substruct
		if ~isstruct(xmlStruct.(currentLevel{i})) || ...
            (~isfield(xmlStruct.(currentLevel{i}), ...
            dtdStruct(dtdStruct(pointer).down).name) && ...
			strcmp(dtdStruct(dtdStruct(pointer).down).repeatType, '*'))
			
			if ~isstr(xmlStruct.(currentLevel{i})) || ...
				~strcmp(xmlStruct.(currentLevel{i}), '')
				changed = 1;
			end;
			
			outputStruct.(currentLevel{i}) = '';
			pointer = dtdStruct(pointer).next;
			continue;
		end;
		
		k = 1;
		if ((strcmp(dtdStruct(pointer).repeatType, '+') || ...
                strcmp(dtdStruct(pointer).repeatType, '*'))) && ...
                (length(xmlStruct.(currentLevel{i})) > 1)
			
            for j = 1:length(xmlStruct.(currentLevel{i}))

                if j <= 1
                    [outputStruct.(currentLevel{i}), out] = ...
                        subAnalyseXML(dtdStruct, ...
                        xmlStruct.(currentLevel{i})(j), dtdStruct(pointer).down);
                else
                    [outputStruct.(currentLevel{i})(j), out] = ...
                        subAnalyseXML(dtdStruct, ...
                        xmlStruct.(currentLevel{i})(j), dtdStruct(pointer).down);
                end;
				
				if out && ~changed
					changed = 1;
				end;
            end;
        else
			
            [outputStruct.(currentLevel{i}), out] = ...
				subAnalyseXML(dtdStruct, ...
				xmlStruct.(currentLevel{i}), dtdStruct(pointer).down);
    
            if out && ~changed
				changed = 1;
			end;
		end;
    else
        k = 1;
		if (strcmp(dtdStruct(pointer).repeatType, '+') || strcmp(dtdStruct(pointer).repeatType, '*'))
			
			k = length(xmlStruct.(currentLevel{i}));
		end;
        
        if isfield(xmlStruct, currentLevel{i}) && ...
			iscell(xmlStruct.(currentLevel{i}))
			
            for j = 1:k
                [value, out] = ...
                    checkValues(xmlStruct.(currentLevel{i}){j}, dtdStruct(pointer).down);
				
				outputStruct.(currentLevel{i}){j} = value;
				
				if out && ~changed
					changed = 1;
				end;
            end;
        else
            [outputStruct.(currentLevel{i}), out] = ...
				checkValues(xmlStruct.(currentLevel{i}), dtdStruct(pointer).down);
			
			if out && ~changed
					changed = 1;
			end;
        end;
	end;
	
    pointer = dtdStruct(pointer).next;
end;


% /////////////////////////////////////////////////////////////////////////////////////
% only checks values
function [outputStruct, changed] = checkValues(xmlStruct, pointer);

outputStruct = [];
changed = 0;

% first check level for endlevel (pointer = 1-3)
if pointer <= 3
	
	if pointer == 1
		% (#PCDATA)
		
        if iscell(xmlStruct)
			xmlStruct = xmlStruct{1};
		end;
        
		if isnumeric(xmlStruct)
			if length(xmlStruct) <= 0
                changed = 1;
            end;
            
            xmlStruct = num2str(xmlStruct);
            
		end;
		
		if ~isstr(xmlStruct)
			outputStruct = '';
			changed = 1;
        else
            outputStruct = xmlStruct;
		end;
		
	else if pointer == 2
		% ANY
		
		outputStruct = xmlStruct;
		
	else if pointer == 3
		% EMPTY
		
		outputStruct = '';
        
        if ~isstr(xmlStruct) || ~strcmp(xmlStruct, '')
            changed = 1;
        end;
		
	end; % 3
	end; % 2
	end; % 1
	
end;
% END: first check level for endlevel (pointer = 1-3)