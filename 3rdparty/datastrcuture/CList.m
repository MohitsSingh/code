classdef CList < handle
% ������һ��������ģ��б�
% list = CList; ����һ���յĶ��ж���
% list = CList(c); ������ж��󣬲���c��ʼ��q����cΪcellʱ��c��Ԫ��Ϊջ����ݣ�
%    ����c����Ϊջ�ĵ�һ�����
%
% ֧�ֲ�����
%     sz = list.size() ���ض�����Ԫ�ظ���Ҳ�������ж϶����Ƿ�ǿա�
%     b = list.empty() ��ն���
%     list.pushtofront(el) ����Ԫ��elѹ���б�ͷ
%     list.pushtorear(el) ����Ԫ��elѹ���б�β��
%     el = list.popfront()  �����б�ͷ��Ԫ�أ��û����Լ�ȷ�����зǿ�
%     el = list.poprear() �����б�β��Ԫ�أ��û����Լ�ȷ���б�ǿ�
%     el = list.front() ���ض���Ԫ�أ��û����Լ�ȷ�����зǿ�
%     el = list.back() ���ض�βԪ�أ��û����Լ�ȷ�����зǿ�
%     list.remove(k) ɾ���k��Ԫ�أ����kΪ���ģ����β����ʼ�� 
%     list.removeall() ɾ���������Ԫ��
%     list.add(el, k) ����Ԫ��el����k��λ�ã����kΪ���ģ���ӽ�β��ʼ��
%     list.contains(el) ���el�Ƿ�������б��У������֣����ص�һ���±�
%     list.get(k) �����б��ƶ�λ�õ�Ԫ�أ����kΪ���ģ����ĩβ��ʼ��
%     list.sublist(from, to) �����б��д�from��to�����ұգ�֮�����ͼ
%     list.content() �����б����ݣ���һάcells�������ʽ���ء�
%     list.toarray() = list.content() content�ı���
%
% See also CStack
%
% copyright: zhangzq@citics.com, 2010.
% url: http://zhiqiang.org/blog/tag/matlab

    properties (Access = private)
        buffer      % һ��cell���飬����ջ�����
        beg         % ������ʼλ��
        len         % ���еĳ���
    end
    
    properties (Access = public)
        capacity    % ջ������������������ʱ����������Ϊ2����
    end
    
    methods
        function obj = CList(c)
            if nargin >= 1 && iscell(c)
                obj.buffer = [c(:); cell(numel(c), 1)];
                obj.beg = 1;
                obj.len = numel(c);
                obj.capacity = 2*numel(c);
            elseif nargin >= 1
                obj.buffer = cell(100, 1);
                obj.buffer{1} = c;
                obj.beg = 1;
                obj.len = 1;
                obj.capacity = 100;                
            else
                obj.buffer = cell(100, 1);
                obj.capacity = 100;
                obj.beg = 1;
                obj.len = 0;
            end
        end
        
        function s = size(obj)
            s = obj.len;
        end
        
        function b = empty(obj)  % �ж��б��Ƿ�Ϊ��
            b = (obj.len == 0);
        end
        
        function pushtorear(obj, el) % ѹ����Ԫ�ص���β
            obj.addcapacity();
            if obj.beg + obj.len  <= obj.capacity
                obj.buffer{obj.beg+obj.len} = el;
            else
                obj.buffer{obj.beg+obj.len-obj.capacity} = el;
            end
            obj.len = obj.len + 1;
        end
        
        function pushtofront(obj, el) % ѹ����Ԫ�ص���β
            obj.addcapacity();
            obj.beg = obj.beg - 1;
            if obj.beg == 0
                obj.beg = obj.capacity; 
            end
            obj.buffer{obj.beg} = el;
            obj.len = obj.len + 1;
        end
        
        function el = popfront(obj) % ��������Ԫ��
            el = obj.buffer(obj.beg);
            obj.beg = obj.beg + 1;
            obj.len = obj.len - 1;
            if obj.beg > obj.capacity
                obj.beg = 1;
            end
        end
        
        function el = poprear(obj) % ������βԪ��
            tmp = obj.beg + obj.len;
            if tmp > obj.capacity
                tmp = tmp - obj.capacity;
            end
            el = obj.buffer(tmp);
            obj.len = obj.len - 1;
        end
        
        function el = front(obj) % ���ض���Ԫ��
            try
                el = obj.buffer{obj.beg};
            catch ME
                throw(ME.messenge);
            end
        end
        
        function el = back(obj) % ���ض�βԪ��
            try
                tmp = obj.beg + obj.len - 1;
                if tmp >= obj.capacity, tmp = tmp - obj.capacity; end;
                el = obj.buffer(tmp);
            catch ME
                throw(ME.messenge);
            end            
        end
        
        function el = top(obj) % ���ض�βԪ��
            try
                tmp = obj.beg + obj.len - 1;
                if tmp >= obj.capacity, tmp = tmp - obj.capacity; end;
                el = obj.buffer(tmp);
            catch ME
                throw(ME.messenge);
            end            
        end
        
        function removeall(obj) % ����б�
            obj.len = 0;
            obj.beg = 1;
        end
        
        % ɾ���k��Ԫ�أ�k����Ϊ���ģ���ʾ��β����ʼ��
        % ���û���趨k����Ϊ����б�����Ԫ��

        function el = getElement(obj, k)
                id = obj.getindex(k);
                el = obj.buffer{id};
        end


        function remove(obj, k)
            if nargin == 1
                obj.len = 0;
                obj.beg = 1;
            else % k ~= 0
                id = obj.getindex(k);

                obj.buffer{id} = [];
                obj.len = obj.len - 1;
                obj.capacity = obj.capacity - 1;

                % ɾ��Ԫ�غ���Ҫ���µ���beg��λ��ֵ
                if id < obj.beg
                    obj.beg = obj.beg - 1;
                end
            end
        end
        
        % ������Ԫ��el����k��Ԫ��֮ǰ�����kΪ��������뵽�����-k��Ԫ��֮��
        function add(obj, el, k)
            obj.addcapacity();
            id = obj.getindex(k);
            
            if k > 0 % �����ڵ�id��Ԫ��֮ǰ
                obj.buffer = [obj.buffer(1:id-1); el; obj.buffer(id:end)];
                if id < obj.beg
                    obj.beg = obj.beg + 1;
                end
            else % k < 0�������ڵ�id��Ԫ��֮��
                obj.buffer = [obj.buffer(1:id); el; obj.buffer(id:end)];
                if id < obj.beg
                    obj.beg = obj.beg + 1;
                end
            end
        end
        
        % ������ʾ����Ԫ��
        function display(obj)
            if obj.size()
                rear = obj.beg + obj.len - 1;
                if rear <= obj.capacity
                    for i = obj.beg : rear
                        disp([num2str(i - obj.beg + 1) '-th element of the stack:']);
                        disp(obj.buffer{i});
                    end
                else
                    for i = obj.beg : obj.capacity
                        disp([num2str(i - obj.beg + 1) '-th element of the stack:']);
                        disp(obj.buffer{i});
                    end     
                    for i = 1 : rear
                        disp([num2str(i + obj.capacity - obj.beg + 1) '-th element of the stack:']);
                        disp(obj.buffer{i});
                    end
                end
            else
                disp('The queue is empty');
            end
        end
        
        
        % ��ȡ�б���������
        function c = content(obj)
            rear = obj.beg + obj.len - 1;
            if rear <= obj.capacity
                c = obj.buffer(obj.beg:rear);                    
            else
                c = obj.buffer([obj.beg:obj.capacity 1:rear]);
            end
        end
        
        % ��ȡ�б��������ݣ���ͬ��obj.content();
        function c = toarray(obj)
            c = obj.content();
        end
    end
    
    
    
    methods (Access = private)
        
        % getindex(k) ���ص�k��Ԫ����buffer���±�λ��
        function id = getindex(obj, k)
            if k > 0
                id = obj.beg + k;
            else
                id = obj.beg + obj.len + k;
            end     
            
            if id > obj.capacity
                id = id - obj.capacity;
            end
        end
        
        % ��buffer��Ԫ�ظ���ӽ���������ʱ��������������һ����
        % ��ʱ��ת�б?ʹ�ô�1��ʼ������б��������������Ͽ�λ��
        function addcapacity(obj)
            if obj.len >= obj.capacity - 1
                sz = obj.len;
                if obj.beg + sz - 1 <= obj.capacity
                    obj.buffer(1:sz) = obj.buffer(obj.beg:obj.beg+sz-1);                    
                else
                    obj.buffer(1:sz) = obj.buffer([obj.beg:obj.capacity, ...
                        1:sz-(obj.capacity-obj.beg+1)]);
                end
                obj.buffer(sz+1:obj.capacity*2) = cell(obj.capacity*2-sz, 1);
                obj.capacity = 2*obj.capacity;
                obj.beg = 1;
            end
        end
    end % private methos
    
    methods (Abstract)
        
    end
end