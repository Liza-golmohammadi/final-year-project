 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 python train_baselines.py


============================================================
Exp 1 MIT-ONLY — HT Variability Baseline
============================================================

Fold 1  |  Threshold: 55.1715  |  Test subjects: 17
Subject-level  Accuracy: 0.5294  F1: 0.6364  AUC: 0.6000

Fold 2  |  Threshold: 54.1460  |  Test subjects: 17
Subject-level  Accuracy: 0.5294  F1: 0.6000  AUC: 0.4444

Fold 3  |  Threshold: 51.4384  |  Test subjects: 17
Subject-level  Accuracy: 0.8824  F1: 0.9231  AUC: 0.9167

Fold 4  |  Threshold: 52.2038  |  Test subjects: 17
Subject-level  Accuracy: 0.6471  F1: 0.6667  AUC: 0.5286

Fold 5  |  Threshold: 56.2850  |  Test subjects: 17
Subject-level  Accuracy: 0.6471  F1: 0.6250  AUC: 0.6500

----- FINAL AVERAGE -----
Mean Accuracy : 0.6471
Mean F1-score : 0.6902
Mean ROC-AUC  : 0.6279

============================================================
Exp 1 MIT-ONLY — Logistic Regression
============================================================

Fold 1  |  Test subjects: 17
Subject-level  Accuracy: 0.4706  F1: 0.4706  AUC: 0.4571

Fold 2  |  Test subjects: 17
Subject-level  Accuracy: 0.7059  F1: 0.6667  AUC: 0.9028

Fold 3  |  Test subjects: 17
Subject-level  Accuracy: 0.6471  F1: 0.7273  AUC: 0.6833

Fold 4  |  Test subjects: 17
Subject-level  Accuracy: 0.8235  F1: 0.8000  AUC: 0.9429

Fold 5  |  Test subjects: 17
Subject-level  Accuracy: 0.7059  F1: 0.6667  AUC: 0.8000

----- FINAL AVERAGE -----
Mean Accuracy : 0.6706
Mean F1-score : 0.6662
Mean ROC-AUC  : 0.7572

============================================================
Exp 2 TAPPY-ONLY — HT Variability Baseline
============================================================

Fold 1  |  Threshold: 34.1290  |  Test subjects: 34
Subject-level  Accuracy: 0.5294  F1: 0.6522  AUC: 0.5926

Fold 2  |  Threshold: 29.6566  |  Test subjects: 37
Subject-level  Accuracy: 0.7027  F1: 0.8000  AUC: 0.5714

Fold 3  |  Threshold: 37.1956  |  Test subjects: 36
Subject-level  Accuracy: 0.5278  F1: 0.6047  AUC: 0.5473

Fold 4  |  Threshold: 29.9935  |  Test subjects: 37
Subject-level  Accuracy: 0.6216  F1: 0.6818  AUC: 0.5697

Fold 5  |  Threshold: 31.7638  |  Test subjects: 37
Subject-level  Accuracy: 0.5946  F1: 0.7368  AUC: 0.5484

----- FINAL AVERAGE -----
Mean Accuracy : 0.5952
Mean F1-score : 0.6951
Mean ROC-AUC  : 0.5659

============================================================
Exp 2 TAPPY-ONLY — Logistic Regression
============================================================

Fold 1  |  Test subjects: 34
Subject-level  Accuracy: 0.5294  F1: 0.6364  AUC: 0.4392

Fold 2  |  Test subjects: 37
Subject-level  Accuracy: 0.6757  F1: 0.7692  AUC: 0.7460

Fold 3  |  Test subjects: 36
Subject-level  Accuracy: 0.5278  F1: 0.6222  AUC: 0.5103

Fold 4  |  Test subjects: 37
Subject-level  Accuracy: 0.5135  F1: 0.6400  AUC: 0.5970

Fold 5  |  Test subjects: 37
Subject-level  Accuracy: 0.4865  F1: 0.6275  AUC: 0.5054

----- FINAL AVERAGE -----
Mean Accuracy : 0.5466
Mean F1-score : 0.6591
Mean ROC-AUC  : 0.5596

============================================================
EXPERIMENT 3: CROSS-DATASET (Train: Tappy → Test: MIT)
============================================================

============================================================
Exp 3 CROSS-DATASET — HT Variability Baseline
============================================================
Threshold (from Tappy train): 31.9934
Subject-level  Accuracy: 0.5647  F1: 0.6942  AUC: 0.6517

Exp 3 CROSS-DATASET — Logistic Regression
Subject-level  Accuracy: 0.5059  F1: 0.0000  AUC: 0.5116

============================================================
Exp 4 BALANCED POOLED — HT Variability Baseline
============================================================

Fold 1  |  Threshold: 43.1523  |  Test subjects: 40
Subject-level  Accuracy: 0.3750  F1: 0.4898  AUC: 0.3929

Fold 2  |  Threshold: 43.3407  |  Test subjects: 41
Subject-level  Accuracy: 0.4878  F1: 0.5116  AUC: 0.5431

Fold 3  |  Threshold: 41.3310  |  Test subjects: 42
Subject-level  Accuracy: 0.6667  F1: 0.7407  AUC: 0.6683

Fold 4  |  Threshold: 41.7375  |  Test subjects: 42
Subject-level  Accuracy: 0.6190  F1: 0.7241  AUC: 0.5543

Fold 5  |  Threshold: 42.0409  |  Test subjects: 42
Subject-level  Accuracy: 0.6429  F1: 0.7368  AUC: 0.4377

----- FINAL AVERAGE -----
Mean Accuracy : 0.5583
Mean F1-score : 0.6406
Mean ROC-AUC  : 0.5192

============================================================
Exp 4 BALANCED POOLED — Logistic Regression
============================================================

Fold 1  |  Test subjects: 40
Subject-level  Accuracy: 0.5500  F1: 0.6250  AUC: 0.6209

Subject-level  Accuracy: 0.6585  F1: 0.6957  AUC: 0.8014

Subject-level  Accuracy: 0.6585  F1: 0.6957  AUC: 0.8014

Subject-level  Accuracy: 0.6585  F1: 0.6957  AUC: 0.8014

Fold 3  |  Test subjects: 42
Subject-level  Accuracy: 0.5952  F1: 0.6383  AUC: 0.7188
Fold 3  |  Test subjects: 42
Subject-level  Accuracy: 0.5952  F1: 0.6383  AUC: 0.7188

Subject-level  Accuracy: 0.5952  F1: 0.6383  AUC: 0.7188


Fold 4  |  Test subjects: 42
Subject-level  Accuracy: 0.6429  F1: 0.7273  AUC: 0.6569
Fold 4  |  Test subjects: 42
Subject-level  Accuracy: 0.6429  F1: 0.7273  AUC: 0.6569

Subject-level  Accuracy: 0.6429  F1: 0.7273  AUC: 0.6569

Fold 5  |  Test subjects: 42

Fold 5  |  Test subjects: 42
Fold 5  |  Test subjects: 42
Subject-level  Accuracy: 0.7619  F1: 0.8148  AUC: 0.7825
Subject-level  Accuracy: 0.7619  F1: 0.8148  AUC: 0.7825

----- FINAL AVERAGE -----
----- FINAL AVERAGE -----
Mean Accuracy : 0.6417
Mean F1-score : 0.7002
Mean ROC-AUC  : 0.7161
PS C:\Users\maede\Downloads\OneDrive_2026-03-22\Solution-4- With Pdrs>


-------------------------------------------------
     python train_baselines_window.py
>> C:\Users\maede\Downloads\OneDrive_2026-03-22\Solution-4- With Pdrs>

============================================================
Exp 1 MIT-ONLY — HT Baseline (Window-level)
============================================================

Fold 1  |  Threshold: 55.1715  |  Test windows: 623  |  Test subjects: 17
Window-level  Accuracy: 0.5072  Balanced Acc: 0.5104  F1: 0.4875  AUC: 0.5727

Fold 2  |  Threshold: 54.1460  |  Test windows: 627  |  Test subjects: 17
Window-level  Accuracy: 0.6523  Balanced Acc: 0.6574  F1: 0.5742  AUC: 0.6798

Fold 3  |  Threshold: 51.4384  |  Test windows: 621  |  Test subjects: 17
Window-level  Accuracy: 0.6892  Balanced Acc: 0.6425  F1: 0.7716  AUC: 0.7298

Fold 4  |  Threshold: 52.2038  |  Test windows: 624  |  Test subjects: 17
Window-level  Accuracy: 0.6474  Balanced Acc: 0.6515  F1: 0.6774  AUC: 0.5851

Fold 5  |  Threshold: 56.2850  |  Test windows: 622  |  Test subjects: 17
Window-level  Accuracy: 0.6013  Balanced Acc: 0.5629  F1: 0.4338  AUC: 0.6571

----- FINAL AVERAGE -----
Mean Accuracy         : 0.6195
Mean Balanced Accuracy: 0.6050
Mean F1-score         : 0.5889
Mean ROC-AUC          : 0.6449

============================================================
Exp 1 MIT-ONLY — LR (Window-level)
============================================================

Fold 1  |  Test windows: 623  |  Test subjects: 17
Window-level  Accuracy: 0.3852  Balanced Acc: 0.3844  F1: 0.4062  AUC: 0.3663

Fold 2  |  Test windows: 627  |  Test subjects: 17
Window-level  Accuracy: 0.7895  Balanced Acc: 0.7808  F1: 0.7130  AUC: 0.8652

Fold 3  |  Test windows: 621  |  Test subjects: 17
Window-level  Accuracy: 0.6892  Balanced Acc: 0.7103  F1: 0.7402  AUC: 0.8018

Fold 4  |  Test windows: 624  |  Test subjects: 17
Window-level  Accuracy: 0.8013  Balanced Acc: 0.7991  F1: 0.7817  AUC: 0.9007

Fold 5  |  Test windows: 622  |  Test subjects: 17
Window-level  Accuracy: 0.7283  Balanced Acc: 0.7764  F1: 0.7101  AUC: 0.8077

----- FINAL AVERAGE -----
Mean Accuracy         : 0.6787
Mean Balanced Accuracy: 0.6902
Mean F1-score         : 0.6703
Mean ROC-AUC          : 0.7484

============================================================
Exp 2 TAPPY-ONLY — HT Baseline (Window-level)
============================================================

Fold 1  |  Threshold: 34.1290  |  Test windows: 35466  |  Test subjects: 34
Window-level  Accuracy: 0.3087  Balanced Acc: 0.4908  F1: 0.3988  AUC: 0.6985

Fold 2  |  Threshold: 29.6566  |  Test windows: 35466  |  Test subjects: 37
Window-level  Accuracy: 0.7914  Balanced Acc: 0.6229  F1: 0.8736  AUC: 0.5638

Fold 3  |  Threshold: 37.1956  |  Test windows: 35466  |  Test subjects: 36
Window-level  Accuracy: 0.5382  Balanced Acc: 0.5474  F1: 0.4041  AUC: 0.6344

Fold 4  |  Threshold: 29.9935  |  Test windows: 35466  |  Test subjects: 37
Window-level  Accuracy: 0.7790  Balanced Acc: 0.6011  F1: 0.8669  AUC: 0.5295

Fold 5  |  Threshold: 31.7638  |  Test windows: 35465  |  Test subjects: 37
Window-level  Accuracy: 0.6167  Balanced Acc: 0.6055  F1: 0.6713  AUC: 0.7885

----- FINAL AVERAGE -----
Mean Accuracy         : 0.6068
Mean Balanced Accuracy: 0.5735
Mean F1-score         : 0.6429
Mean ROC-AUC          : 0.6429

============================================================
Exp 2 TAPPY-ONLY — LR (Window-level)
============================================================

Fold 1  |  Test windows: 35466  |  Test subjects: 34
Window-level  Accuracy: 0.7158  Balanced Acc: 0.6161  F1: 0.8234  AUC: 0.5729

Fold 2  |  Test windows: 35466  |  Test subjects: 37
Window-level  Accuracy: 0.7058  Balanced Acc: 0.6585  F1: 0.7980  AUC: 0.7577

Fold 3  |  Test windows: 35466  |  Test subjects: 36
Window-level  Accuracy: 0.7828  Balanced Acc: 0.7870  F1: 0.7631  AUC: 0.9061

Fold 4  |  Test windows: 35466  |  Test subjects: 37
Window-level  Accuracy: 0.7893  Balanced Acc: 0.6183  F1: 0.8731  AUC: 0.7787

Fold 5  |  Test windows: 35465  |  Test subjects: 37
Window-level  Accuracy: 0.6758  Balanced Acc: 0.6619  F1: 0.7281  AUC: 0.7374

----- FINAL AVERAGE -----
Mean Accuracy         : 0.7339
Mean Balanced Accuracy: 0.6684
Mean F1-score         : 0.7971
Mean ROC-AUC          : 0.7506

============================================================
EXPERIMENT 3: CROSS-DATASET (Train: Tappy → Test: MIT)
============================================================

============================================================
Exp 3 CROSS-DATASET — HT Baseline (Window-level)
============================================================
Threshold (from Tappy train): 31.9934
Test windows: 3117  |  Test subjects: 85
Window-level  Accuracy: 0.5868  Balanced Acc: 0.6025  F1: 0.6869  AUC: 0.6571

Exp 3 CROSS-DATASET — LR (Window-level)
Test windows: 3117  |  Test subjects: 85
Window-level  Accuracy: 0.5338  Balanced Acc: 0.5120  F1: 0.0547  AUC: 0.4772

============================================================
EXPERIMENT 4: BALANCED POOLED MODEL
============================================================

============================================================
Exp 4 BALANCED POOLED — HT Baseline (Window-level)
============================================================

Fold 1  |  Threshold: 43.1523  |  Test windows: 1247  |  Test subjects: 40
Window-level  Accuracy: 0.3184  Balanced Acc: 0.3306  F1: 0.2928  AUC: 0.3394

Fold 2  |  Threshold: 43.3407  |  Test windows: 1247  |  Test subjects: 41
Window-level  Accuracy: 0.6415  Balanced Acc: 0.6350  F1: 0.5714  AUC: 0.6789

Fold 3  |  Threshold: 41.3310  |  Test windows: 1247  |  Test subjects: 42
Window-level  Accuracy: 0.7057  Balanced Acc: 0.6849  F1: 0.7690  AUC: 0.6515

Fold 4  |  Threshold: 41.7375  |  Test windows: 1247  |  Test subjects: 42
Window-level  Accuracy: 0.5662  Balanced Acc: 0.4886  F1: 0.7016  AUC: 0.5039

Fold 5  |  Threshold: 42.0409  |  Test windows: 1246  |  Test subjects: 42
Window-level  Accuracy: 0.6814  Balanced Acc: 0.6786  F1: 0.7074  AUC: 0.6705

----- FINAL AVERAGE -----
Mean Accuracy         : 0.5826
Mean Balanced Accuracy: 0.5635
Mean F1-score         : 0.6085
Mean ROC-AUC          : 0.5688

============================================================
Exp 4 BALANCED POOLED — LR (Window-level)
============================================================

Fold 1  |  Test windows: 1247  |  Test subjects: 40
Window-level  Accuracy: 0.5092  Balanced Acc: 0.5158  F1: 0.5234  AUC: 0.5646

Fold 2  |  Test windows: 1247  |  Test subjects: 41
Window-level  Accuracy: 0.5245  Balanced Acc: 0.5314  F1: 0.4848  AUC: 0.6708
 AUC: 0.8178
 AUC: 0.8178

 AUC: 0.8178
 AUC: 0.8178

 AUC: 0.8178

Fold 4  |  Test windows: 1247  |  Test subjects: 42

Fold 4  |  Test windows: 1247  |  Test subjects: 42
Fold 4  |  Test windows: 1247  |  Test subjects: 42
Window-level  Accuracy: 0.6087  Balanced Acc: 0.6095  F1: 0.7258  AUC: 0.6932

Fold 5  |  Test windows: 1246  |  Test subjects: 42

Fold 5  |  Test windows: 1246  |  Test subjects: 42
Window-level  Accuracy: 0.7657  Balanced Acc: 0.7684  F1: 0.7630  AUC: 0.8281
Fold 5  |  Test windows: 1246  |  Test subjects: 42
Window-level  Accuracy: 0.7657  Balanced Acc: 0.7684  F1: 0.7630  AUC: 0.8281
Window-level  Accuracy: 0.7657  Balanced Acc: 0.7684  F1: 0.7630  AUC: 0.8281


----- FINAL AVERAGE -----
Mean Accuracy         : 0.6266
Mean Accuracy         : 0.6266
Mean Balanced Accuracy: 0.6360
Mean F1-score         : 0.6505
Mean ROC-AUC          : 0.7149
PS C:\Users\maede\Downloads\OneDrive_2026-03-22\Solution-4- With Pdrs>


-------------------------------------------------
python train_experiments.py
>> C:\Users\maede\Downloads\OneDrive_2026-03-22\Solution-4- With Pdrs>

===== DATA SUMMARY =====
Total samples: 180446

Dataset distribution:
dataset
tappy      177329
mit_cs1      1626
mit_cs2      1491
Name: count, dtype: int64

Label distribution:
label_pd
1    128535
0     51911
Name: count, dtype: int64

Number of features used: 26
Feature names: ['dd_cv', 'dd_iqr', 'dd_mean', 'dd_median', 'dd_p95', 'dd_std', 'dig_KK_rate', 'dig_KS_rate', 'dig_SK_rate', 'dig_SS_rate', 'hold_cv', 'hold_iqr', 'hold_mean', 'hold_median', 'hold_p95', 'hold_std', 'n', 'pause_rate_dd_gt_1000', 'pause_rate_dd_gt_500', 'pause_rate_dd_gt_750', 'ud_cv', 'ud_iqr', 'ud_mean', 'ud_median', 'ud_p95', 'ud_std']

============================================================
Exp 1 MIT-ONLY (SVM)
============================================================

Fold 1  |  Test subjects: 17
Subject-level  Accuracy: 0.5294  Balanced Acc: 0.5571  F1: 0.5000  AUC: 0.5857

Fold 2  |  Test subjects: 17
Subject-level  Accuracy: 0.8235  Balanced Acc: 0.8194  F1: 0.8000  AUC: 0.9306

Fold 3  |  Test subjects: 17
Subject-level  Accuracy: 0.8235  Balanced Acc: 0.8167  F1: 0.8696  AUC: 0.7833

Fold 4  |  Test subjects: 17
Subject-level  Accuracy: 0.8235  Balanced Acc: 0.8286  F1: 0.8000  AUC: 0.9429

Fold 5  |  Test subjects: 17
Subject-level  Accuracy: 0.5882  Balanced Acc: 0.6500  F1: 0.5333  AUC: 0.7167

----- FINAL AVERAGE -----
Mean Accuracy         : 0.7176
Mean Balanced Accuracy: 0.7344
Mean F1-score         : 0.7006
Mean ROC-AUC          : 0.7918

============================================================
Exp 1 MIT-ONLY (Random Forest)
============================================================

Fold 1  |  Test subjects: 17
Subject-level  Accuracy: 0.4706  Balanced Acc: 0.4857  F1: 0.4706  AUC: 0.5429

Fold 2  |  Test subjects: 17
Subject-level  Accuracy: 0.8824  Balanced Acc: 0.8750  F1: 0.8571  AUC: 0.9028

Fold 3  |  Test subjects: 17
Subject-level  Accuracy: 0.8235  Balanced Acc: 0.8167  F1: 0.8696  AUC: 0.8667

Fold 4  |  Test subjects: 17
Subject-level  Accuracy: 0.8235  Balanced Acc: 0.8071  F1: 0.7692  AUC: 0.9143

Fold 5  |  Test subjects: 17
Subject-level  Accuracy: 0.6471  Balanced Acc: 0.6917  F1: 0.5714  AUC: 0.7833

----- FINAL AVERAGE -----
Mean Accuracy         : 0.7294
Mean Balanced Accuracy: 0.7352
Mean F1-score         : 0.7076
Mean ROC-AUC          : 0.8020

============================================================
Exp 2 TAPPY-ONLY (SVM)
============================================================

Fold 1  |  Test subjects: 34
Subject-level  Accuracy: 0.5588  Balanced Acc: 0.4048  F1: 0.7059  AUC: 0.4603

Fold 2  |  Test subjects: 37
Subject-level  Accuracy: 0.6216  Balanced Acc: 0.4861  F1: 0.7500  AUC: 0.4841

Fold 3  |  Test subjects: 36
Subject-level  Accuracy: 0.5278  Balanced Acc: 0.3889  F1: 0.6792  AUC: 0.4321

Fold 4  |  Test subjects: 37
Subject-level  Accuracy: 0.5405  Balanced Acc: 0.4652  F1: 0.6909  AUC: 0.5970

Fold 5  |  Test subjects: 37
Subject-level  Accuracy: 0.6216  Balanced Acc: 0.4382  F1: 0.7586  AUC: 0.4624

----- FINAL AVERAGE -----
Mean Accuracy         : 0.5741
Mean Balanced Accuracy: 0.4366
Mean F1-score         : 0.7169
Mean ROC-AUC          : 0.4872

============================================================
Exp 2 TAPPY-ONLY (Random Forest)
============================================================

Fold 1  |  Test subjects: 34
Subject-level  Accuracy: 0.5882  Balanced Acc: 0.4233  F1: 0.7308  AUC: 0.4709

Fold 2  |  Test subjects: 37
Subject-level  Accuracy: 0.6757  Balanced Acc: 0.5218  F1: 0.7931  AUC: 0.4921

Fold 3  |  Test subjects: 36
Subject-level  Accuracy: 0.6111  Balanced Acc: 0.4444  F1: 0.7500  AUC: 0.3210

Fold 4  |  Test subjects: 37
Subject-level  Accuracy: 0.5676  Balanced Acc: 0.4879  F1: 0.7143  AUC: 0.4727

Fold 5  |  Test subjects: 37
Subject-level  Accuracy: 0.7027  Balanced Acc: 0.4194  F1: 0.8254  AUC: 0.3548

----- FINAL AVERAGE -----
Mean Accuracy         : 0.6291
Mean Balanced Accuracy: 0.4594
Mean F1-score         : 0.7627
Mean ROC-AUC          : 0.4223

============================================================
EXPERIMENT 3: CROSS-DATASET (Train: Tappy → Test: MIT)
============================================================

Exp 3 CROSS-DATASET — SVM
Subject-level  Accuracy: 0.4941  Balanced Acc: 0.5000  F1: 0.6614  AUC: 0.6035

Exp 3 CROSS-DATASET — Random Forest
Subject-level  Accuracy: 0.4941  Balanced Acc: 0.5000  F1: 0.6614  AUC: 0.6257

============================================================
EXPERIMENT 4: BALANCED POOLED MODEL
============================================================

============================================================
Exp 4 BALANCED POOLED (SVM)
============================================================

Fold 1  |  Test subjects: 40
Subject-level  Accuracy: 0.6250  Balanced Acc: 0.5962  F1: 0.7059  AUC: 0.6841

Fold 2  |  Test subjects: 41
Subject-level  Accuracy: 0.6829  Balanced Acc: 0.6722  F1: 0.7347  AUC: 0.7560

Fold 3  |  Test subjects: 42
Subject-level  Accuracy: 0.7381  Balanced Acc: 0.7043  F1: 0.8000  AUC: 0.7572

Fold 4  |  Test subjects: 42
Subject-level  Accuracy: 0.6429  Balanced Acc: 0.6408  F1: 0.7273  AUC: 0.6452

Fold 5  |  Test subjects: 42
Subject-level  Accuracy: 0.6667  Balanced Acc: 0.6313  F1: 0.7500  AUC: 0.7029

----- FINAL AVERAGE -----
Mean Accuracy         : 0.6711
Mean Balanced Accuracy: 0.6490
Mean F1-score         : 0.7436
Mean ROC-AUC          : 0.7091

============================================================
Exp 4 BALANCED POOLED (Random Forest)
============================================================

Fold 1  |  Test subjects: 40
Subject-level  Accuracy: 0.6750  Balanced Acc: 0.6676  F1: 0.7347  AUC: 0.7527

Fold 2  |  Test subjects: 41
Subject-level  Accuracy: 0.6585  Balanced Acc: 0.6459  F1: 0.7200  AUC: 0.7225
  AUC: 0.7933

Fold 4  |  Test subjects: 42
Subject-level  Accuracy: 0.6667  Balanced Acc: 0.6276  F1: 0.7586  AUC: 0.6305

Fold 5  |  Test subjects: 42
Subject-level  Accuracy: 0.6905  Balanced Acc: 0.6910  F1: 0.7547  AUC: 0.7560

----- FINAL AVERAGE -----
Mean Accuracy         : 0.6810
Mean Accuracy         : 0.6810
Mean Balanced Accuracy: 0.6586
Mean F1-score         : 0.7522
Mean ROC-AUC          : 0.7310
PS C:\Users\maede\Downloads\OneDrive_2026-03-22\Solution-4- With Pdrs>

--------------------------------------------------------------------------------------------------

>>                                                                     python train_experiments_window.py
>> C:\Users\maede\Downloads\OneDrive_2026-03-22\Solution-4- With Pdrs>

===== DATA SUMMARY =====
Total samples (windows): 180446

Dataset distribution:
dataset
tappy      177329
mit_cs1      1626
mit_cs2      1491
Name: count, dtype: int64

Label distribution (windows):
label_pd
1    128535
0     51911
Name: count, dtype: int64

Number of features used: 26
Feature names: ['dd_cv', 'dd_iqr', 'dd_mean', 'dd_median', 'dd_p95', 'dd_std', 'dig_KK_rate', 'dig_KS_rate', 'dig_SK_rate', 'dig_SS_rate', 'hold_cv', 'hold_iqr', 'hold_mean', 'hold_median', 'hold_p95', 'hold_std', 'n', 'pause_rate_dd_gt_1000', 'pause_rate_dd_gt_500', 'pause_rate_dd_gt_750', 'ud_cv', 'ud_iqr', 'ud_mean', 'ud_median', 'ud_p95', 'ud_std']

============================================================
Exp 1 MIT-ONLY (SVM) — Window-level
============================================================

Fold 1  |  Test windows: 623  |  Test subjects: 17
Window-level  Accuracy: 0.5618  Balanced Acc: 0.5686  F1: 0.5081  AUC: 0.5470

Fold 2  |  Test windows: 627  |  Test subjects: 17
Window-level  Accuracy: 0.8118  Balanced Acc: 0.7936  F1: 0.7306  AUC: 0.8777

Fold 3  |  Test windows: 621  |  Test subjects: 17
Window-level  Accuracy: 0.7440  Balanced Acc: 0.7173  F1: 0.8077  AUC: 0.7458

Fold 4  |  Test windows: 624  |  Test subjects: 17
Window-level  Accuracy: 0.7949  Balanced Acc: 0.7929  F1: 0.7762  AUC: 0.8870

Fold 5  |  Test windows: 622  |  Test subjects: 17
Window-level  Accuracy: 0.5723  Balanced Acc: 0.5910  F1: 0.5199  AUC: 0.6914

----- FINAL AVERAGE -----
Mean Accuracy         : 0.6970
Mean Balanced Accuracy: 0.6927
Mean F1-score         : 0.6685
Mean ROC-AUC          : 0.7498

============================================================
Exp 1 MIT-ONLY (Random Forest) — Window-level
============================================================

Fold 1  |  Test windows: 623  |  Test subjects: 17
Window-level  Accuracy: 0.5746  Balanced Acc: 0.5797  F1: 0.5407  AUC: 0.5202

Fold 2  |  Test windows: 627  |  Test subjects: 17
Window-level  Accuracy: 0.8389  Balanced Acc: 0.8144  F1: 0.7601  AUC: 0.9202

Fold 3  |  Test windows: 621  |  Test subjects: 17
Window-level  Accuracy: 0.7778  Balanced Acc: 0.7422  F1: 0.8373  AUC: 0.8110

Fold 4  |  Test windows: 624  |  Test subjects: 17
Window-level  Accuracy: 0.7436  Balanced Acc: 0.7381  F1: 0.6863  AUC: 0.8361

Fold 5  |  Test windows: 622  |  Test subjects: 17
Window-level  Accuracy: 0.5932  Balanced Acc: 0.6009  F1: 0.5217  AUC: 0.6414

----- FINAL AVERAGE -----
Mean Accuracy         : 0.7056
Mean Balanced Accuracy: 0.6951
Mean F1-score         : 0.6692
Mean ROC-AUC          : 0.7458

============================================================
Exp 2 TAPPY-ONLY (SVM) — Window-level
============================================================

Fold 1  |  Test windows: 35466  |  Test subjects: 34
Window-level  Accuracy: 0.8012  Balanced Acc: 0.5134  F1: 0.8877  AUC: 0.5482

Fold 2  |  Test windows: 35466  |  Test subjects: 37
Window-level  Accuracy: 0.7601  Balanced Acc: 0.6354  F1: 0.8482  AUC: 0.7368

Fold 3  |  Test windows: 35466  |  Test subjects: 36
Window-level  Accuracy: 0.6161  Balanced Acc: 0.6155  F1: 0.6308  AUC: 0.6566

Fold 4  |  Test windows: 35466  |  Test subjects: 37
Window-level  Accuracy: 0.7433  Balanced Acc: 0.5435  F1: 0.8453  AUC: 0.7416

Fold 5  |  Test windows: 35465  |  Test subjects: 37
Window-level  Accuracy: 0.5934  Balanced Acc: 0.5759  F1: 0.6618  AUC: 0.6481

----- FINAL AVERAGE -----
Mean Accuracy         : 0.7028
Mean Balanced Accuracy: 0.5767
Mean F1-score         : 0.7747
Mean ROC-AUC          : 0.6663

============================================================
Exp 2 TAPPY-ONLY (Random Forest) — Window-level
============================================================

Fold 1  |  Test windows: 35466  |  Test subjects: 34
Window-level  Accuracy: 0.8371  Balanced Acc: 0.5372  F1: 0.9097  AUC: 0.5180

Fold 2  |  Test windows: 35466  |  Test subjects: 37
Window-level  Accuracy: 0.7923  Balanced Acc: 0.6306  F1: 0.8736  AUC: 0.7739

Fold 3  |  Test windows: 35466  |  Test subjects: 36
Window-level  Accuracy: 0.4843  Balanced Acc: 0.4714  F1: 0.6213  AUC: 0.5328

Fold 4  |  Test windows: 35466  |  Test subjects: 37
Window-level  Accuracy: 0.7552  Balanced Acc: 0.5054  F1: 0.8573  AUC: 0.7031

Fold 5  |  Test windows: 35465  |  Test subjects: 37
Window-level  Accuracy: 0.5414  Balanced Acc: 0.4759  F1: 0.6924  AUC: 0.5347

----- FINAL AVERAGE -----
Mean Accuracy         : 0.6821
Mean Balanced Accuracy: 0.5241
Mean F1-score         : 0.7909
Mean ROC-AUC          : 0.6125

============================================================
EXPERIMENT 3: CROSS-DATASET (Train: Tappy → Test: MIT)
============================================================
Train windows: 177329  |  Test windows: 3117
Test subjects: 85

Exp 3 CROSS-DATASET — SVM
Window-level  Accuracy: 0.4774  Balanced Acc: 0.5000  F1: 0.6463  AUC: 0.5301

Exp 3 CROSS-DATASET — Random Forest
Window-level  Accuracy: 0.4790  Balanced Acc: 0.5013  F1: 0.6456  AUC: 0.6357

============================================================
EXPERIMENT 4: BALANCED POOLED MODEL
============================================================

============================================================
Exp 4 BALANCED POOLED (SVM) — Window-level
============================================================

Fold 1  |  Test windows: 1247  |  Test subjects: 40
Window-level  Accuracy: 0.6159  Balanced Acc: 0.6115  F1: 0.6566  AUC: 0.6277

Fold 2  |  Test windows: 1247  |  Test subjects: 41
Window-level  Accuracy: 0.5678  Balanced Acc: 0.5902  F1: 0.5614  AUC: 0.6674

Fold 3  |  Test windows: 1247  |  Test subjects: 42
Window-level  Accuracy: 0.8083  Balanced Acc: 0.8156  F1: 0.8429  AUC: 0.8860

Fold 4  |  Test windows: 1247  |  Test subjects: 42
Window-level  Accuracy: 0.7241  Balanced Acc: 0.6483  F1: 0.8236  AUC: 0.7366

Fold 5  |  Test windows: 1246  |  Test subjects: 42
Window-level  Accuracy: 0.6517  Balanced Acc: 0.6514  F1: 0.6651  AUC: 0.7519

----- FINAL AVERAGE -----
Mean Accuracy         : 0.6736
Mean Balanced Accuracy: 0.6634
Mean F1-score         : 0.7099
Mean ROC-AUC          : 0.7339

============================================================
Exp 4 BALANCED POOLED (Random Forest) — Window-level
============================================================

Fold 1  |  Test windows: 1247  |  Test subjects: 40
Window-level  Accuracy: 0.6792  Balanced Acc: 0.6673  F1: 0.7279  AUC: 0.7080

Fold 2  |  Test windows: 1247  |  Test subjects: 41
Window-level  Accuracy: 0.4836  Balanced Acc: 0.5163  F1: 0.5084  AUC: 0.6395
 AUC: 0.9155

Fold 4  |  Test windows: 1247  |  Test subjects: 42
Window-level  Accuracy: 0.7514  Balanced Acc: 0.6822  F1: 0.8425  AUC: 0.7767

Fold 5  |  Test windows: 1246  |  Test subjects: 42
Window-level  Accuracy: 0.6188  Balanced Acc: 0.6189  F1: 0.6304  AUC: 0.6996

----- FINAL AVERAGE -----
Mean Accuracy         : 0.6772
Mean Accuracy         : 0.6772
Mean Balanced Accuracy: 0.6635
Mean F1-score         : 0.7195
Mean ROC-AUC          : 0.7479
PS C:\Users\maede\Downloads\OneDrive_2026-03-22\Solution-4- With Pdrs>
--------------------------------------------------------------------------------------------------

>>                                                                     python train_cnn.py
>> C:\Users\maede\Downloads\OneDrive_2026-03-22\Solution-4- With Pdrs>
2026-03-29 05:24:52.528046: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-03-29 05:24:57.547847: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

===== DATA SUMMARY =====
X shape: (183248, 200, 3)
y distribution: {np.int64(0): np.int64(51889), np.int64(1): np.int64(131359)}
Dataset distribution: {'mit_cs1': np.int64(1604), 'mit_cs2': np.int64(1465), 'tappy': np.int64(180179)}

============================================================
Exp 1 MIT-ONLY (CNN)
============================================================
2026-03-29 05:25:14.685680: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

Fold 1  |  Test subjects: 16
Subject-level  Accuracy: 0.5625  F1: 0.5333  AUC: 0.7143
  [eval] Dropping 2 subject(s) with <3 windows

Fold 2  |  Test subjects: 18
Subject-level  Accuracy: 0.6875  F1: 0.6667  AUC: 0.8000

Fold 3  |  Test subjects: 17
Subject-level  Accuracy: 0.6471  F1: 0.7273  AUC: 0.7333

Fold 4  |  Test subjects: 17
Subject-level  Accuracy: 0.5882  F1: 0.5882  AUC: 0.6528

Fold 5  |  Test subjects: 17
Subject-level  Accuracy: 0.6471  F1: 0.5000  AUC: 0.7083

----- FINAL AVERAGE -----
Mean Accuracy : 0.6265
Mean F1-score : 0.6031
Mean ROC-AUC  : 0.7217

============================================================
Exp 2 TAPPY-ONLY (CNN)
============================================================
  [eval] Dropping 1 subject(s) with <3 windows

Fold 1  |  Test subjects: 34
Subject-level  Accuracy: 0.6364  F1: 0.7692  AUC: 0.5100
  [eval] Dropping 2 subject(s) with <3 windows

Fold 2  |  Test subjects: 37
Subject-level  Accuracy: 0.6571  F1: 0.7857  AUC: 0.5408
  [eval] Dropping 1 subject(s) with <3 windows

Fold 3  |  Test subjects: 36
Subject-level  Accuracy: 0.6000  F1: 0.7083  AUC: 0.5417
  [eval] Dropping 2 subject(s) with <3 windows

Fold 4  |  Test subjects: 37
Subject-level  Accuracy: 0.7143  F1: 0.8276  AUC: 0.5867
  [eval] Dropping 1 subject(s) with <3 windows

Fold 5  |  Test subjects: 37
Subject-level  Accuracy: 0.6944  F1: 0.8000  AUC: 0.5385

----- FINAL AVERAGE -----
Mean Accuracy : 0.6604
Mean F1-score : 0.7782
Mean ROC-AUC  : 0.5435

============================================================
EXPERIMENT 3: CROSS-DATASET (Train: Tappy → Test: MIT)
============================================================
  [eval] Dropping 2 subject(s) with <3 windows

Exp 3 CROSS-DATASET — CNN
Subject-level  Accuracy: 0.4337  F1: 0.5524  AUC: 0.4390

============================================================
Exp 4 BALANCED POOLED (CNN)
============================================================
  [eval] Dropping 7 subject(s) with <3 windows

Fold 1  |  Test subjects: 41
Subject-level  Accuracy: 0.6471  F1: 0.7692  AUC: 0.6542
  [eval] Dropping 7 subject(s) with <3 windows

Fold 2  |  Test subjects: 42
Subject-level  Accuracy: 0.6571  F1: 0.7273  AUC: 0.7067
  [eval] Dropping 8 subject(s) with <3 windows
Fold 3  |  Test subjects: 42
Subject-level  Accuracy: 0.6176  F1: 0.6829  AUC: 0.6929
Fold 3  |  Test subjects: 42
Subject-level  Accuracy: 0.6176  F1: 0.6829  AUC: 0.6929
  [eval] Dropping 7 subject(s) with <3 windows
Fold 3  |  Test subjects: 42
Subject-level  Accuracy: 0.6176  F1: 0.6829  AUC: 0.6929
  [eval] Dropping 7 subject(s) with <3 windows
Fold 3  |  Test subjects: 42
Subject-level  Accuracy: 0.6176  F1: 0.6829  AUC: 0.6929
Fold 3  |  Test subjects: 42
Subject-level  Accuracy: 0.6176  F1: 0.6829  AUC: 0.6929
  [eval] Dropping 7 subject(s) with <3 windows
Fold 3  |  Test subjects: 42
Fold 3  |  Test subjects: 42
Subject-level  Accuracy: 0.6176  F1: 0.6829  AUC: 0.6929
Fold 3  |  Test subjects: 42
Fold 3  |  Test subjects: 42
Fold 3  |  Test subjects: 42
Fold 3  |  Test subjects: 42
Fold 3  |  Test subjects: 42
Fold 3  |  Test subjects: 42
Subject-level  Accuracy: 0.6176  F1: 0.6829  AUC: 0.6929
Subject-level  Accuracy: 0.6176  F1: 0.6829  AUC: 0.6929
  [eval] Dropping 7 subject(s) with <3 windows
  [eval] Dropping 7 subject(s) with <3 windows

Fold 4  |  Test subjects: 41

Fold 4  |  Test subjects: 41
Fold 4  |  Test subjects: 41
Subject-level  Accuracy: 0.5000  F1: 0.5641  AUC: 0.6458
  [eval] Dropping 7 subject(s) with <3 windows

  [eval] Dropping 7 subject(s) with <3 windows

Fold 5  |  Test subjects: 43

Fold 5  |  Test subjects: 43
Fold 5  |  Test subjects: 43
Subject-level  Accuracy: 0.6389  F1: 0.6829  AUC: 0.6698
Subject-level  Accuracy: 0.6389  F1: 0.6829  AUC: 0.6698

----- FINAL AVERAGE -----
----- FINAL AVERAGE -----
----- FINAL AVERAGE -----
----- FINAL AVERAGE -----
----- FINAL AVERAGE -----
Mean Accuracy : 0.6121
Mean F1-score : 0.6853
Mean ROC-AUC  : 0.6739
PS C:\Users\maede\Downloads\OneDrive_2026-03-22\Solution-4- With Pdrs>
--------------------------------------------------------------------------------------------------

>>                                                                     python train_cnn_window.py
>> C:\Users\maede\Downloads\OneDrive_2026-03-22\Solution-4- With Pdrs>
2026-03-29 05:25:02.695396: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-03-29 05:25:04.216655: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

===== DATA SUMMARY =====
X shape: (183248, 200, 3)
y distribution: {np.int64(0): np.int64(51889), np.int64(1): np.int64(131359)}
Dataset distribution: {'mit_cs1': np.int64(1604), 'mit_cs2': np.int64(1465), 'tappy': np.int64(180179)}

============================================================
Exp 1 MIT-ONLY (CNN) — Window-level
============================================================
2026-03-29 05:25:14.593853: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

Fold 1  |  Test windows: 613  |  Test subjects: 16
Window-level  Accuracy: 0.5318  Balanced Acc: 0.5273  F1: 0.4405  AUC: 0.5691

Fold 2  |  Test windows: 612  |  Test subjects: 18
Window-level  Accuracy: 0.4592  Balanced Acc: 0.5813  F1: 0.5196  AUC: 0.7334

Fold 3  |  Test windows: 618  |  Test subjects: 17
Window-level  Accuracy: 0.6343  Balanced Acc: 0.5980  F1: 0.7264  AUC: 0.6477

Fold 4  |  Test windows: 614  |  Test subjects: 17
Window-level  Accuracy: 0.5977  Balanced Acc: 0.5977  F1: 0.5944  AUC: 0.6211

Fold 5  |  Test windows: 612  |  Test subjects: 17
Window-level  Accuracy: 0.5735  Balanced Acc: 0.5719  F1: 0.4314  AUC: 0.5947

----- FINAL AVERAGE -----
Mean Accuracy         : 0.5593
Mean Balanced Accuracy: 0.5753
Mean F1-score         : 0.5425
Mean ROC-AUC          : 0.6332

============================================================
Exp 2 TAPPY-ONLY (CNN) — Window-level
============================================================

Fold 1  |  Test windows: 36036  |  Test subjects: 34
Window-level  Accuracy: 0.8747  Balanced Acc: 0.6276  F1: 0.9316  AUC: 0.7314

Fold 2  |  Test windows: 36036  |  Test subjects: 37
Window-level  Accuracy: 0.7588  Balanced Acc: 0.6287  F1: 0.8549  AUC: 0.7190

Fold 3  |  Test windows: 36036  |  Test subjects: 36
Window-level  Accuracy: 0.6262  Balanced Acc: 0.6231  F1: 0.6750  AUC: 0.7420

Fold 4  |  Test windows: 36036  |  Test subjects: 37
Window-level  Accuracy: 0.5521  Balanced Acc: 0.4399  F1: 0.7003  AUC: 0.4967

Fold 5  |  Test windows: 36035  |  Test subjects: 37
Window-level  Accuracy: 0.5101  Balanced Acc: 0.5652  F1: 0.5228  AUC: 0.5632

----- FINAL AVERAGE -----
Mean Accuracy         : 0.6644
Mean Balanced Accuracy: 0.5769
Mean F1-score         : 0.7369
Mean ROC-AUC          : 0.6504

============================================================
EXPERIMENT 3: CROSS-DATASET (Train: Tappy → Test: MIT)
============================================================

Exp 3 CROSS-DATASET — CNN (Window-level)
Test windows: 3069
Window-level  Accuracy: 0.4435  Balanced Acc: 0.4547  F1: 0.5426  AUC: 0.4435

============================================================
Exp 4 BALANCED POOLED (CNN) — Window-level
============================================================

Fold 1  |  Test windows: 1228  |  Test subjects: 41
Window-level  Accuracy: 0.7231  Balanced Acc: 0.6868  F1: 0.8046  AUC: 0.7656

Fold 2  |  Test windows: 1228  |  Test subjects: 42
Window-level  Accuracy: 0.6710  Balanced Acc: 0.6645  F1: 0.7094  AUC: 0.7426
 AUC: 0.6497

 AUC: 0.6497

Fold 4  |  Test windows: 1227  |  Test subjects: 41
 AUC: 0.6497

 AUC: 0.6497

Fold 4  |  Test windows: 1227  |  Test subjects: 41
 AUC: 0.6497
 AUC: 0.6497

 AUC: 0.6497
 AUC: 0.6497
 AUC: 0.6497
 AUC: 0.6497
 AUC: 0.6497
 AUC: 0.6497

 AUC: 0.6497

Fold 4  |  Test windows: 1227  |  Test subjects: 41

Fold 4  |  Test windows: 1227  |  Test subjects: 41
Window-level  Accuracy: 0.7156  Balanced Acc: 0.6710  F1: 0.8100  AUC: 0.7639
Fold 4  |  Test windows: 1227  |  Test subjects: 41
Window-level  Accuracy: 0.7156  Balanced Acc: 0.6710  F1: 0.8100  AUC: 0.7639

Fold 5  |  Test windows: 1227  |  Test subjects: 43

Fold 5  |  Test windows: 1227  |  Test subjects: 43
Window-level  Accuracy: 0.5990  Balanced Acc: 0.6061  F1: 0.5535  AUC: 0.6480
Fold 5  |  Test windows: 1227  |  Test subjects: 43
Window-level  Accuracy: 0.5990  Balanced Acc: 0.6061  F1: 0.5535  AUC: 0.6480
Window-level  Accuracy: 0.5990  Balanced Acc: 0.6061  F1: 0.5535  AUC: 0.6480


----- FINAL AVERAGE -----
Mean Accuracy         : 0.6657
Mean Accuracy         : 0.6657
Mean Accuracy         : 0.6657
Mean Accuracy         : 0.6657
Mean Accuracy         : 0.6657
Mean Balanced Accuracy: 0.6494
Mean F1-score         : 0.6932
Mean ROC-AUC          : 0.7140
PS C:\Users\maede\Downloads\OneDrive_2026-03-22\Solution-4- With Pdrs>