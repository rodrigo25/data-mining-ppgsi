function [ confMatrix, estatisticas, estatisticas_struct ] = calc_result( Yd, Y )
  
  confMatrix = confusionmat(Yd,Y,'order',[1,0]); % Matriz de confusao

  TP = confMatrix(1,1); % True Positive
  FN = confMatrix(1,2); % False Negative
  FP = confMatrix(2,1); % False Positive
  TN = confMatrix(2,2); % True Negative

  TPR = TP/(TP+FN); % True Positive Rate - Recall
  TNR = TN/(TN+FP); % True Negative Rate - Specificity
  PPV = TP/(TP+FP); % Positive Predictive Value - Precision
  NPV = TN/(TN+FN); % Negative Predictive Value
  FNR = FN/(FN+TP); % False Negative Rate - Miss Rate
  FPR = FP/(FP+TN); % False Positive Rate - Fall-out
  FDR = FP/(FP+TP); % False Discovery Rate
  FOR = FN/(FN+TN); % False Omission Rate
  ACC = (TP+TN)/(TP+TN+FP+FN); % Accuracy
  Fscore = 2*(PPV*TPR)/(PPV+TPR); % F-score
  MCC = (TP*TN - FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)); % Matthews Correlation Coefficient

  
  estatisticas = {'TP', 'True Positive', TP
                  'FN', 'False Negative', FN
                  'FP', 'False Positive', FP
                  'TN', 'True Negative', TN
                  'TPR', 'True Positive Rate (Recall)', TPR
                  'TNR', 'True Negative Rate (Specificity)', TNR
                  'PPV', 'Positive Predictive Value (Precision)', PPV
                  'NPV', 'Negative Predictive Value', NPV
                  'FNR', 'False Negative Rate (Miss Rate)', FNR
                  'FPR', 'False Positive Rate (Fall-out)', FPR
                  'FDR', 'False Discovery Rate', FDR
                  'FOR', 'False Omission Rate', FOR
                  'ACC', 'Accuracy', ACC
                  'Fscore', 'F-score', Fscore
                  'MCC', 'Matthews Correlation Coefficient', MCC};
  
                
  estatisticas_struct = struct('TP', {'True Positive', TP},...
                         'FN', {'False Negative', FN},...
                         'FP', {'False Positive', FP},...
                         'TN', {'True Negative', TN},...
                         'TPR', {'True Positive Rate (Recall)', TPR},...
                         'TNR', {'True Negative Rate (Specificity)', TNR},...
                         'PPV', {'Positive Predictive Value (Precision)', PPV},...
                         'NPV', {'Negative Predictive Value', NPV},...
                         'FNR', {'False Negative Rate (Miss Rate)', FNR},...
                         'FPR', {'False Positive Rate (Fall-out)', FPR},...
                         'FDR', {'False Discovery Rate', FDR},...
                         'FOR', {'False Omission Rate', FOR},...
                         'ACC', {'Accuracy', ACC},...
                         'Fscore', {'F-score', Fscore},...
                         'MCC', {'Matthews Correlation Coefficient', MCC});
  
end

