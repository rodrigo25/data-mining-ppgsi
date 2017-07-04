function Yh = MLPtreinaOneVsAll(X, Yd, Xval, Ydval, Xtest, ...
    h, mE, alfaType, gradientType, earlyStopping, alfa0)
    
    Yh  = oneVsAll( Xtr, Ytr, Xval, Yval, Xtest, h, mE, alfaType, gradientType, earlyStopping, alfa0);
end

function Yh  = oneVsAll( Xtr, Ytr, Xval, Yval, Xtest, h, mE, alfaType, gradientType, earlyStopping, alfa0)

    classes = unique(Ytr);

    probabilities = zeros(size(Xtest,1), length(classes));            

    for c=1:length(classes)

        YtrBin = toBinaryClass( Ytr, c );
        YvalBin = toBinaryClass( Yval, c );

        [A, B] = MLPtreina(Xtr, YtrBin, Xval, YvalBin, h, mE, alfaType, gradientType, earlyStopping, alfa0);

        Yh = MLPsaida(A, B, Xtest);
        probabilities( :, c ) = Yh;
    end

    [~,Yh] = max(probabilities,[], 2);
end

function Y = toBinaryClass(Y, targetClass)
    Y( Y ~= targetClass ) = 0;
    Y( Y == targetClass ) = 1;
end