% OutlookType=struct('Sunny',1,'Rainy',2,'Overcast',3);  
% TemperatureType=struct('hot',1,'warm',2,'cool',3);  
% HumidityType=struct('high',1,'norm',2);  
% WindyType={'True',1,'False',0};  
% PlayGolf={'Yes',1,'No',0};  
% data=struct('Outlook',[],'Temperature',[],'Humidity',[],'Windy',[],'PlayGolf',[]);  
function [decisionTreeModel]=applyDecisionTree()

Outlook=[1,1,3,2,2,2,3,1,1,2,1,3,3,2]';   
Temperature=[1,1,1,2,3,3,3,2,3,3,2,2,1,2]';  
Humidity=[1,1,1,1,2,2,2,1,2,2,2,1,2,1]';  
Windy=[0,1,0,0,0,1,1,0,0,0,1,1,0,1]';  
  
data=[Outlook Temperature Humidity Windy];  
PlayGolf=[0,0,1,1,1,0,1,0,1,1,1,1,1,0]';  
propertyName={'Outlook','Temperature','Humidity','Windy'};  
delta=0.1;  
decisionTreeModel=decisionTree(data,PlayGolf,propertyName,delta);  
end

function decisionTreeModel=decisionTree(data,label,propertyName,delta)    
    global Node;  

    Node=struct('fatherNodeName',[],'EdgeProperty',[],'NodeName',[]);  
    BuildTree('root','Stem',data,label,propertyName,delta);  
    Node(1)=[];  
    model.Node=Node;  
    decisionTreeModel=model;  
end

function BuildTree(fatherNodeName,edge,data,label,propertyName,delta)   
    global Node;  
    sonNode=struct('fatherNodeName',[],'EdgeProperty',[],'NodeName',[]);  
    sonNode.fatherNodeName=fatherNodeName;  
    sonNode.EdgeProperty=edge;  
    if length(unique(label))==1  %当前节点已经纯净
        sonNode.NodeName=label(1);  
        Node=[Node sonNode];  
        return;  
    end  
    if length(propertyName)<1  
        labelSet=unique(label);  
        labelNum=length(labelSet);  
        for i=1:labelNum  
            labelNum=length(find(label==labelSet(i)));  
        end  
        [~,labelIndex]=max(labelNum);  
        sonNode.NodeName=labelSet(labelIndex);  
        Node=[Node sonNode];  
        return;  
    end  

[sonIndex,BuildNode]=CalcuteNode(data,label,delta); 
    if BuildNode  
        dataRowIndex=setdiff(1:length(propertyName),sonIndex);  
        sonNode.NodeName=propertyName(sonIndex);  
        Node=[Node sonNode];  
        propertyName(sonIndex)=[];  
        sonData=data(:,sonIndex);  
        sonEdge=unique(sonData);  

        for i=1:length(sonEdge)  
            edgeDataIndex=find(sonData==sonEdge(i));  
            BuildTree(sonNode.NodeName,sonEdge(i),data(edgeDataIndex,dataRowIndex),label(edgeDataIndex,:),propertyName,delta);  %注意在生成子节点的过程中，子节点的信息熵是和父节点的这条枝干的信息熵做对比。
        end  
    else  
        labelSet=unique(label);  
        labelNum=length(labelSet);  
        for i=1:labelNum  
            labelNum=length(find(label==labelSet(i)));  
        end  
        [~,labelIndex]=max(labelNum);  
        sonNode.NodeName=labelSet(labelIndex);  
        Node=[Node sonNode];  
        return;  
    end  
end

function [NodeIndex,BuildNode]=CalcuteNode(data,label,delta)  
  
LargeEntropy=CEntropy(label);  %当前节点的信息熵
[m,n]=size(data);  
EntropyGain=LargeEntropy*ones(1,n);  %计算不同feature下的信息增益
BuildNode=true;  
for i=1:n  
    pData=data(:,i);  
    itemList=unique(pData);  %同一种feature有itemList个不同取值，计算不同取值的信息熵。因此信息增益倾向于选择feature取值更多地feature作为split feature
    for j=1:length(itemList)  
        itemIndex=find(pData==itemList(j));  
        E1=CEntropy(label(itemIndex));  %若选择feature i作为split feature，则feature i取值为itemList(j)时其子节点label的纯度（熵）
        EntropyGain(i)=EntropyGain(i)-length(itemIndex)/m*E1;%计算第i种feature下进行split时的信息增益  
    end  
    % 此处运行则为增益率，注释掉则为增益  
    % EntropyGain(i)=EntropyGain(i)/CEntropy(pData);   
end  
[maxGainEntropy,NodeIndex]=max(EntropyGain);  
if maxGainEntropy<delta  
    BuildNode=false;  
end 
end

function result=CEntropy(propertyList)  
  
    result=0;  
    totalLength=length(propertyList);  
    itemList=unique(propertyList);  
    pNum=length(itemList);  
    for i=1:pNum  
        itemLength=length(find(propertyList==itemList(i)));  
        pItem=itemLength/totalLength;  
        result=result-pItem*log2(pItem);   %当前节点的熵=-∑ni/N*log(ni/N)，ni表示对于某个特定feature不同取值的个数
    end 
end

%最终结果：
%        -------------root:outlook----------------
%   1:humidity           2. Windy               3.Ture
%1:False  2:Ture      0:Ture   1:False
