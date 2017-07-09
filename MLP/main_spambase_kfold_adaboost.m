function [] = main_spambase_kfold_adaboost()

    neurons = [20];
    epochs = [1000 5000];
    T = 35;
    numExecutions = 1;
    foldCount = 5;
    
    fileNames = pre_processing('data_spambase',0, 'kfold', foldCount,'pca');
    baseResultFileDir = 'ensemble-T35-arch1\spambase\';
    for fileIndex=1:numel(fileNames)
        data = load(fileNames{1}); 
        data = load(fileNames{1});
        XtrFolds = data.XtrFolds;
        YtrFolds = data.YtrFolds;
        XvalFolds = data.XvalFolds;
        YvalFolds = data.YvalFolds;
        preProcInfo = data.preProcInfo;
        classes = preProcInfo.classes;

        resultFileDir = [baseResultFileDir num2str(fileIndex) '\'];
        if ~exist(resultFileDir, 'dir')
            mkdir(resultFileDir);
        end
       
        for i=1:length(neurons)
            h = neurons(i);
            for j=1:length(epochs)
                maxEpochs = epochs(j);
                YhEnsemblesPerFold = cell(foldCount,1);
                parfor (k=1:foldCount,4)
                    YhEnsembles = cell(numExecutions,1);
                    [Xtr, Ytr, Xtest, Ytest, Xval, Yval] = createDatasets(XtrFolds, YtrFolds, XvalFolds, YvalFolds, k); 
                    Ytr = multiclassY(Ytr);
                    Ytest = multiclassY(Ytest);
                    Yval = multiclassY(Yval);
                    for t=1:numExecutions
                        resultFileName = [resultFileDir sprintf('maxEpochs%d_h%d_t%d_k%d', maxEpochs, h, t, k)];
                        if exist([resultFileName '.mat'], 'file') == 2
                           continue; 
                        end
                        Yh = adaboostM2(Xtr, Ytr, Xval, Yval, Xtest, Ytest, classes, T, h, maxEpochs, 0);
                        [~,Yh] = max(Yh,[],2);
                        YhEnsembles{t} = Yh;
                        acc = multiclassConfusionMatrix( YtestClasses, Yh, classes );
                        fprintf('h = %d\tfile = %d\tt = %d\tacc = %f\tmaxEpochs = %d\n', h, fileIndex, t, acc, maxEpochs);
                    end
                    YhEnsemblesPerFold{k} = YhEnsembles;
                end
                for k=1:foldCount
                    YhEnsembles = YhEnsemblesPerFold{k}
                    for t=1:numExecutions
                        resultFileName = [resultFileDir sprintf('maxEpochs%d_h%d_t%d_k%d', maxEpochs, h, t)];
                        if exist([resultFileName '.mat'], 'file') == 2
                           continue; 
                        end
                        Yh = YhEnsembles{t};
                        acc = multiclassConfusionMatrix( YtestClasses, Yh, classes, 1, 'Adaboost', [resultFileName '_confusion_matrix'] );
                        save(resultFileName, 'acc', 'Yh');
                    end
                end
            end
        end
    end
end