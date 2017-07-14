%load data_Test1 %k=3
%load data_S1.mat %k=15
%load data_S2.mat %k=15
%load data_S3.mat %k=15
%load data_S4.mat %k=15
%load data_D31.mat %k=31
load data_pathbased.mat %k=3
%load data_t4.8k.mat %k=6

k=6;
[n,m] = size(X);

data = (X-repmat(mean(X),n,1))./repmat(std(X),n,1); % Z-SCORE NORALIZATION

%[U, kpoints] = kmeans(data,k,'e'); % K-MEANS CLUSTERING
%[U, kpoints] = kmeanspp(data,k,'e'); % K-MEANS++ CLUSTERING
%plotClusters2D(data,kpoints,U,k,{'X','Y'});





%%%%
% RODANDO VÁRIAS ITERAÇÕES E ESCOLHENDO A MELHOR
%%%%
itMax = 50; %max iterações
minD = Inf; %função objetivo mínima (menor distancia)
Ustar = zeros(k, n);
KPstar = zeros(k, m);

for it=1:itMax

  [U, kpoints] = kmeans(data,k,'e'); % K-MEANS CLUSTERING
  %[U, kpoints] = kmeanspp(data,k,'e'); % K-MEANS++ CLUSTERING

  %Calcula distancias dos pontos para os centroides
  D = sum(sqrt(sum((data-(U'*kpoints)).^2, 2)));
  %for i=1:n
  %  p = data(i,:);
  %  ind = U(:,i)==1;
  %  D = D + sqrt(sum((p-kpoints(ind,:)).^2, 2));
  %end

  %verifica se foi execução melhor (exclui execuções com clusters sem valores D=NaN)
  if D<minD
    minD = D;
    Ustar = U;
    KPstar = kpoints;
  end

end

fprintf('Distancia = %d\n', minD);
plotClusters2D(data,KPstar,Ustar,k,{'X','Y'});
