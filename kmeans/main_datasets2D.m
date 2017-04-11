%load data_Test1 %k=3
load data_S1.mat %k=15
%load data_S2.mat %k=15
%load data_S3.mat %k=15
%load data_S4.mat %k=15
%load data_D31.mat %k=31
%load data_pathbased.mat %k=3
%load data_t4.8k.mat %k=6

k=15;
[n,m] = size(X);

data = (X-repmat(mean(X),n,1))./repmat(std(X),n,1); % Z-SCORE NORALIZATION

%[U, kpoints] = kmeans(data,k,'e'); % K-MEANS CLUSTERING
[U, kpoints] = kmeanspp(data,k,'e'); % PAM (K-medoids) CLUSTERING

plotClusters2D(data,kpoints,U,k,{'X','Y'});
