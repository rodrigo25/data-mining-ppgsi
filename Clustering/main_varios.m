%load S1.mat %k=15
load S2.mat %k=15
%load S3.mat %k=15
%load S4.mat %k=15
%load D31.mat %k31
%load pathbased.mat
%load t4.8k.mat

%X = X(:,1:2); %para D31 e pathbased

k=15;

[n,m] = size(X);

% Z-SCORE NORALIZATION
X = (X-repmat(mean(X),n,1))./repmat(std(X),n,1);

% K-MEANS CLUSTERING
[U, kpoints] = kmeans(X,k,'e');

% APRESENTAÇÃO DOS RESULTADOS
clusters = cell(k,1);
for i=1:k
  clusters{i} = U(i,:)==1;
end

figure
hold all
markers = {'+','o','*','x','s','d','^','v','>','<','p','h'};
for i=1:k
  scatter(X(clusters{i},1), X(clusters{i},2), markers{mod(i,numel(markers))+1})
end

scatter(kpoints(:,1),kpoints(:,2),40,'k','filled')
