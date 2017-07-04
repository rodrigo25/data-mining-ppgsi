function [Yh,Yhval] = MLPtreinaOneVsAll(Xtr0, Ytr, Xval0, Yval, Xtest0, ...
    h, mE, alfaType, gradientType, earlyStopping, alfa0, featuresList)
    
    [~,Ytr] = max(Ytr,[],2);
    [~,Yval] = max(Yval,[],2);
    
    classes = unique(Ytr);
    numClasses = length(classes);
    numClassifiers = numClasses - 1;
    
    if nargin < 12
        featuresList = cell(numClassifiers,1);
        for c=1:numClassifiers
            featuresList{c} = 1:size(Xtr0,2);
        end
    end

    
    Xtr = cell(numClassifiers,1);
    Xval = cell(numClassifiers,1);
    Xtest = cell(numClassifiers,1);
    for c=1:numClassifiers
        features = featuresList{c};
        Xtr{c} = Xtr0(:,features);
        Xval{c} = Xval0(:,features);
        Xtest{c} = Xtest0(:,features); 
    end

    N = size(Xtest0,1);
    Nval = size(Xval0,1);
    probabilities = zeros(N, numClassifiers);
    probabilitiesVal = zeros(Nval, numClassifiers);

    parfor (c=1:numClassifiers,numClassifiers)
    %for c=1:numClassifiers
    
        YtrBin = toBinaryClass( Ytr, c );
        YvalBin = toBinaryClass( Yval, c );
        
        [A, B] = MLPtreina(Xtr{c}, YtrBin, Xval{c}, YvalBin, ...
            h, mE, alfaType, gradientType, earlyStopping, alfa0);
        
        Yhval = MLPsaida(Xval{c}, A, B);
        probabilitiesVal( :, c ) = Yhval;
        
        Yh = MLPsaida(Xtest{c}, A, B);
        probabilities( :, c ) = Yh;
    end

    Yh = zeros(N, numClasses);
    Yh(:,1) =  1 - probabilities(:,1);
    for i=2:(numClassifiers)
       Yh(:,i) =  probabilities(:,i-1) - probabilities(:,i);
    end
    Yh(:,numClasses) = probabilities(:,numClassifiers);
    
    [~,Yh] = max(Yh,[], 2);
    
    Yhval = zeros(Nval, numClasses);
    Yhval(:,1) =  1 - probabilitiesVal(:,1);
    for i=2:(numClassifiers)
       Yhval(:,i) =  probabilitiesVal(:,i-1) - probabilitiesVal(:,i);
    end
    Yhval(:,numClasses) = probabilitiesVal(:,numClassifiers);
    
    [~,Yhval] = max(Yhval,[], 2);
end

function Y = toBinaryClass(Y, targetClass)
    Y( Y <= targetClass ) = 0;
    Y( Y > targetClass )  = 1;
end