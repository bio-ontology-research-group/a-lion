grep "ROC" $1  | sed "s/ROC_AUC = //g" | ./calculate_avg.sh
