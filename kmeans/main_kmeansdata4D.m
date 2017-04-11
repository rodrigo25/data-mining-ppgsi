load kmeansdata
k=4;

[n,m] = size(X);

% X = (X-repmat(mean(X),n,1))./repmat(std(X),n,1); % Z-SCORE NORALIZATION

[U, kpoints] = kmeans(X,k,'e'); % K-means

plotClusters4D(X,kpoints,U,k,{'X','Y','Z','T'});