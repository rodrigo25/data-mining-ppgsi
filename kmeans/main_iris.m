load fisheriris.mat
k=3;

[n,m] = size(meas);

% Z-SCORE NORALIZATION
data = (meas-repmat(mean(meas),n,1))./repmat(std(meas),n,1);

% MIN-MAX NORMALIZATION
%min_val = min(meas)-.1;
%max_val = max(meas)+.5;
%data = zeros(n,m);
%for i=1:m
%  data(:,i) = (meas(:,i)-min_val(i))/(max_val(i)-min_val(i));
%end

% K-MEANS CLUSTERING
[U, kpoints] = kmeans(data,k,'e');

%[U, kpoints] = kmeans(data(:,3:4),k); %Teste com menos dimensões

% GRÁFICOS
labels = {'sepal length' 'sepal width' 'petal length' 'petal width'};
plotClusters4D(data,kpoints,U,k,labels);

% APRESENTAÇÃO DOS RESULTADOS
k1 = U(1,:)==1;
k2 = U(2,:)==1;
k3 = U(3,:)==1;

cluster1 = [sum(strcmp(species(k1), 'setosa'))
           sum(strcmp(species(k1), 'versicolor'))
           sum(strcmp(species(k1), 'virginica'))];

cluster2 = [sum(strcmp(species(k2), 'setosa'))
           sum(strcmp(species(k2), 'versicolor'))
           sum(strcmp(species(k2), 'virginica'))];

cluster3 = [sum(strcmp(species(k3), 'setosa'))
           sum(strcmp(species(k3), 'versicolor'))
           sum(strcmp(species(k3), 'virginica'))];

table(cluster1,cluster2,cluster3,'RowNames',{'setosa','versicolor','virginica'})
