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


% GRÁFICO PARA TESTE COM MENOS DIMENSÕES

% plota todos os gráficos em 3D
plotClustersIris3D(data,kpoints,k1,k2,k3);

% plota todos os gráficos em 2D como no wikipedia
plotClustersIris(data,kpoints,k1,k2,k3);




% plota um gráfico em 2D com dimensões selecionadas
%labels = {'sepal length' 'sepal width' 'petal length' 'petal width'};
%dim1 = 1; %dimensões selecionadas
%dim2 = 2; 
%clf
%plot(data(k1,dim1), data(k1,dim2), 'ro')
%hold on
%plot(data(k2,dim1), data(k2,dim2), 'bx')
%plot(data(k3,dim1), data(k3,dim2), 'gd')
%scatter(kpoints(:,dim1),kpoints(:,dim2),40,'k','filled')
%xlabel(labels(dim1));
%ylabel(labels(dim2));
