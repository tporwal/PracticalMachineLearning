---
title: "Practical Machine Learning Coursera Project"
author: "Tanishq Porwal"
date: "11/08/2020"
output: 
  html_document:
    keep_md : true
       

---

# Project Information:

### Background: 
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

### Data
The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

### Goal (What you should submit)
The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.


```r
#Loading the necessary libraries:
library(caret)
library(rpart)
library(ggplot2)
library(e1071)
library(rattle)
```

## Loading Data:

```r
#Loading the necessary libraries:
train <- read.csv("E:\\RStudio\\projects\\PracticalMachineLearning\\pml-training.csv",na.strings=c("NA","#DIV/0!",""))
test <- read.csv("E:\\RStudio\\projects\\PracticalMachineLearning\\pml-testing.csv", na.strings=c("NA","#DIV/0!",""))
```

## Partitioning the training dataset into two parts:

```r
#Loading the necessary libraries:
set.seed(12345)
t <- createDataPartition(train$classe, p=0.6, list = FALSE)
train1 <- train[t, ]
valid <- train[-t, ]
dim(train1); dim(valid)
```

```
## [1] 11776   160
```

```
## [1] 7846  160
```

```r
head(train1)
```

```
##     X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 1   1  carlitos           1323084231               788290 05/12/2011 11:23
## 2   2  carlitos           1323084231               808298 05/12/2011 11:23
## 5   5  carlitos           1323084232               196328 05/12/2011 11:23
## 9   9  carlitos           1323084232               484323 05/12/2011 11:23
## 10 10  carlitos           1323084232               484434 05/12/2011 11:23
## 14 14  carlitos           1323084232               576390 05/12/2011 11:23
##    new_window num_window roll_belt pitch_belt yaw_belt total_accel_belt
## 1          no         11      1.41       8.07    -94.4                3
## 2          no         11      1.41       8.07    -94.4                3
## 5          no         12      1.48       8.07    -94.4                3
## 9          no         12      1.43       8.16    -94.4                3
## 10         no         12      1.45       8.17    -94.4                3
## 14         no         12      1.42       8.21    -94.4                3
##    kurtosis_roll_belt kurtosis_picth_belt kurtosis_yaw_belt skewness_roll_belt
## 1                  NA                  NA                NA                 NA
## 2                  NA                  NA                NA                 NA
## 5                  NA                  NA                NA                 NA
## 9                  NA                  NA                NA                 NA
## 10                 NA                  NA                NA                 NA
## 14                 NA                  NA                NA                 NA
##    skewness_roll_belt.1 skewness_yaw_belt max_roll_belt max_picth_belt
## 1                    NA                NA            NA             NA
## 2                    NA                NA            NA             NA
## 5                    NA                NA            NA             NA
## 9                    NA                NA            NA             NA
## 10                   NA                NA            NA             NA
## 14                   NA                NA            NA             NA
##    max_yaw_belt min_roll_belt min_pitch_belt min_yaw_belt amplitude_roll_belt
## 1            NA            NA             NA           NA                  NA
## 2            NA            NA             NA           NA                  NA
## 5            NA            NA             NA           NA                  NA
## 9            NA            NA             NA           NA                  NA
## 10           NA            NA             NA           NA                  NA
## 14           NA            NA             NA           NA                  NA
##    amplitude_pitch_belt amplitude_yaw_belt var_total_accel_belt avg_roll_belt
## 1                    NA                 NA                   NA            NA
## 2                    NA                 NA                   NA            NA
## 5                    NA                 NA                   NA            NA
## 9                    NA                 NA                   NA            NA
## 10                   NA                 NA                   NA            NA
## 14                   NA                 NA                   NA            NA
##    stddev_roll_belt var_roll_belt avg_pitch_belt stddev_pitch_belt
## 1                NA            NA             NA                NA
## 2                NA            NA             NA                NA
## 5                NA            NA             NA                NA
## 9                NA            NA             NA                NA
## 10               NA            NA             NA                NA
## 14               NA            NA             NA                NA
##    var_pitch_belt avg_yaw_belt stddev_yaw_belt var_yaw_belt gyros_belt_x
## 1              NA           NA              NA           NA         0.00
## 2              NA           NA              NA           NA         0.02
## 5              NA           NA              NA           NA         0.02
## 9              NA           NA              NA           NA         0.02
## 10             NA           NA              NA           NA         0.03
## 14             NA           NA              NA           NA         0.02
##    gyros_belt_y gyros_belt_z accel_belt_x accel_belt_y accel_belt_z
## 1          0.00        -0.02          -21            4           22
## 2          0.00        -0.02          -22            4           22
## 5          0.02        -0.02          -21            2           24
## 9          0.00        -0.02          -20            2           24
## 10         0.00         0.00          -21            4           22
## 14         0.00        -0.02          -22            4           21
##    magnet_belt_x magnet_belt_y magnet_belt_z roll_arm pitch_arm yaw_arm
## 1             -3           599          -313     -128      22.5    -161
## 2             -7           608          -311     -128      22.5    -161
## 5             -6           600          -302     -128      22.1    -161
## 9              1           602          -312     -128      21.7    -161
## 10            -3           609          -308     -128      21.6    -161
## 14            -8           598          -310     -128      21.4    -161
##    total_accel_arm var_accel_arm avg_roll_arm stddev_roll_arm var_roll_arm
## 1               34            NA           NA              NA           NA
## 2               34            NA           NA              NA           NA
## 5               34            NA           NA              NA           NA
## 9               34            NA           NA              NA           NA
## 10              34            NA           NA              NA           NA
## 14              34            NA           NA              NA           NA
##    avg_pitch_arm stddev_pitch_arm var_pitch_arm avg_yaw_arm stddev_yaw_arm
## 1             NA               NA            NA          NA             NA
## 2             NA               NA            NA          NA             NA
## 5             NA               NA            NA          NA             NA
## 9             NA               NA            NA          NA             NA
## 10            NA               NA            NA          NA             NA
## 14            NA               NA            NA          NA             NA
##    var_yaw_arm gyros_arm_x gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y
## 1           NA        0.00        0.00       -0.02        -288         109
## 2           NA        0.02       -0.02       -0.02        -290         110
## 5           NA        0.00       -0.03        0.00        -289         111
## 9           NA        0.02       -0.03       -0.02        -288         109
## 10          NA        0.02       -0.03       -0.02        -288         110
## 14          NA        0.02        0.00       -0.03        -288         111
##    accel_arm_z magnet_arm_x magnet_arm_y magnet_arm_z kurtosis_roll_arm
## 1         -123         -368          337          516                NA
## 2         -125         -369          337          513                NA
## 5         -123         -374          337          506                NA
## 9         -122         -369          341          518                NA
## 10        -124         -376          334          516                NA
## 14        -124         -371          331          523                NA
##    kurtosis_picth_arm kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm
## 1                  NA               NA                NA                 NA
## 2                  NA               NA                NA                 NA
## 5                  NA               NA                NA                 NA
## 9                  NA               NA                NA                 NA
## 10                 NA               NA                NA                 NA
## 14                 NA               NA                NA                 NA
##    skewness_yaw_arm max_roll_arm max_picth_arm max_yaw_arm min_roll_arm
## 1                NA           NA            NA          NA           NA
## 2                NA           NA            NA          NA           NA
## 5                NA           NA            NA          NA           NA
## 9                NA           NA            NA          NA           NA
## 10               NA           NA            NA          NA           NA
## 14               NA           NA            NA          NA           NA
##    min_pitch_arm min_yaw_arm amplitude_roll_arm amplitude_pitch_arm
## 1             NA          NA                 NA                  NA
## 2             NA          NA                 NA                  NA
## 5             NA          NA                 NA                  NA
## 9             NA          NA                 NA                  NA
## 10            NA          NA                 NA                  NA
## 14            NA          NA                 NA                  NA
##    amplitude_yaw_arm roll_dumbbell pitch_dumbbell yaw_dumbbell
## 1                 NA      13.05217      -70.49400    -84.87394
## 2                 NA      13.13074      -70.63751    -84.71065
## 5                 NA      13.37872      -70.42856    -84.85306
## 9                 NA      13.15463      -70.42520    -84.91563
## 10                NA      13.33034      -70.85059    -84.44602
## 14                NA      13.41048      -70.99594    -84.28005
##    kurtosis_roll_dumbbell kurtosis_picth_dumbbell kurtosis_yaw_dumbbell
## 1                      NA                      NA                    NA
## 2                      NA                      NA                    NA
## 5                      NA                      NA                    NA
## 9                      NA                      NA                    NA
## 10                     NA                      NA                    NA
## 14                     NA                      NA                    NA
##    skewness_roll_dumbbell skewness_pitch_dumbbell skewness_yaw_dumbbell
## 1                      NA                      NA                    NA
## 2                      NA                      NA                    NA
## 5                      NA                      NA                    NA
## 9                      NA                      NA                    NA
## 10                     NA                      NA                    NA
## 14                     NA                      NA                    NA
##    max_roll_dumbbell max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell
## 1                 NA                 NA               NA                NA
## 2                 NA                 NA               NA                NA
## 5                 NA                 NA               NA                NA
## 9                 NA                 NA               NA                NA
## 10                NA                 NA               NA                NA
## 14                NA                 NA               NA                NA
##    min_pitch_dumbbell min_yaw_dumbbell amplitude_roll_dumbbell
## 1                  NA               NA                      NA
## 2                  NA               NA                      NA
## 5                  NA               NA                      NA
## 9                  NA               NA                      NA
## 10                 NA               NA                      NA
## 14                 NA               NA                      NA
##    amplitude_pitch_dumbbell amplitude_yaw_dumbbell total_accel_dumbbell
## 1                        NA                     NA                   37
## 2                        NA                     NA                   37
## 5                        NA                     NA                   37
## 9                        NA                     NA                   37
## 10                       NA                     NA                   37
## 14                       NA                     NA                   37
##    var_accel_dumbbell avg_roll_dumbbell stddev_roll_dumbbell var_roll_dumbbell
## 1                  NA                NA                   NA                NA
## 2                  NA                NA                   NA                NA
## 5                  NA                NA                   NA                NA
## 9                  NA                NA                   NA                NA
## 10                 NA                NA                   NA                NA
## 14                 NA                NA                   NA                NA
##    avg_pitch_dumbbell stddev_pitch_dumbbell var_pitch_dumbbell avg_yaw_dumbbell
## 1                  NA                    NA                 NA               NA
## 2                  NA                    NA                 NA               NA
## 5                  NA                    NA                 NA               NA
## 9                  NA                    NA                 NA               NA
## 10                 NA                    NA                 NA               NA
## 14                 NA                    NA                 NA               NA
##    stddev_yaw_dumbbell var_yaw_dumbbell gyros_dumbbell_x gyros_dumbbell_y
## 1                   NA               NA             0.00            -0.02
## 2                   NA               NA             0.00            -0.02
## 5                   NA               NA             0.00            -0.02
## 9                   NA               NA             0.00            -0.02
## 10                  NA               NA             0.00            -0.02
## 14                  NA               NA             0.02            -0.02
##    gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z
## 1              0.00             -234               47             -271
## 2              0.00             -233               47             -269
## 5              0.00             -233               48             -270
## 9              0.00             -232               47             -269
## 10             0.00             -235               48             -270
## 14            -0.02             -234               48             -268
##    magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z roll_forearm
## 1               -559               293               -65         28.4
## 2               -555               296               -64         28.3
## 5               -554               292               -68         28.0
## 9               -549               292               -65         27.7
## 10              -558               291               -69         27.7
## 14              -554               295               -68         27.2
##    pitch_forearm yaw_forearm kurtosis_roll_forearm kurtosis_picth_forearm
## 1          -63.9        -153                    NA                     NA
## 2          -63.9        -153                    NA                     NA
## 5          -63.9        -152                    NA                     NA
## 9          -63.8        -152                    NA                     NA
## 10         -63.8        -152                    NA                     NA
## 14         -63.9        -151                    NA                     NA
##    kurtosis_yaw_forearm skewness_roll_forearm skewness_pitch_forearm
## 1                    NA                    NA                     NA
## 2                    NA                    NA                     NA
## 5                    NA                    NA                     NA
## 9                    NA                    NA                     NA
## 10                   NA                    NA                     NA
## 14                   NA                    NA                     NA
##    skewness_yaw_forearm max_roll_forearm max_picth_forearm max_yaw_forearm
## 1                    NA               NA                NA              NA
## 2                    NA               NA                NA              NA
## 5                    NA               NA                NA              NA
## 9                    NA               NA                NA              NA
## 10                   NA               NA                NA              NA
## 14                   NA               NA                NA              NA
##    min_roll_forearm min_pitch_forearm min_yaw_forearm amplitude_roll_forearm
## 1                NA                NA              NA                     NA
## 2                NA                NA              NA                     NA
## 5                NA                NA              NA                     NA
## 9                NA                NA              NA                     NA
## 10               NA                NA              NA                     NA
## 14               NA                NA              NA                     NA
##    amplitude_pitch_forearm amplitude_yaw_forearm total_accel_forearm
## 1                       NA                    NA                  36
## 2                       NA                    NA                  36
## 5                       NA                    NA                  36
## 9                       NA                    NA                  36
## 10                      NA                    NA                  36
## 14                      NA                    NA                  36
##    var_accel_forearm avg_roll_forearm stddev_roll_forearm var_roll_forearm
## 1                 NA               NA                  NA               NA
## 2                 NA               NA                  NA               NA
## 5                 NA               NA                  NA               NA
## 9                 NA               NA                  NA               NA
## 10                NA               NA                  NA               NA
## 14                NA               NA                  NA               NA
##    avg_pitch_forearm stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
## 1                 NA                   NA                NA              NA
## 2                 NA                   NA                NA              NA
## 5                 NA                   NA                NA              NA
## 9                 NA                   NA                NA              NA
## 10                NA                   NA                NA              NA
## 14                NA                   NA                NA              NA
##    stddev_yaw_forearm var_yaw_forearm gyros_forearm_x gyros_forearm_y
## 1                  NA              NA            0.03            0.00
## 2                  NA              NA            0.02            0.00
## 5                  NA              NA            0.02            0.00
## 9                  NA              NA            0.03            0.00
## 10                 NA              NA            0.02            0.00
## 14                 NA              NA            0.00           -0.02
##    gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
## 1            -0.02             192             203            -215
## 2            -0.02             192             203            -216
## 5            -0.02             189             206            -214
## 9            -0.02             193             204            -214
## 10           -0.02             190             205            -215
## 14           -0.03             193             202            -214
##    magnet_forearm_x magnet_forearm_y magnet_forearm_z classe
## 1               -17              654              476      A
## 2               -18              661              473      A
## 5               -17              655              473      A
## 9               -16              653              476      A
## 10              -22              656              473      A
## 14              -14              659              478      A
```

## Cleaning Data:
Removing zero covariates, from this we will get to know about those variable who has very little variability and not likely good predictors.

```r
nzv <- nearZeroVar(train1, saveMetrics = TRUE)
nzv
```

```
##                          freqRatio percentUnique zeroVar   nzv
## X                         1.000000  1.000000e+02   FALSE FALSE
## user_name                 1.102186  5.095109e-02   FALSE FALSE
## raw_timestamp_part_1      1.038462  7.107677e+00   FALSE FALSE
## raw_timestamp_part_2      1.250000  9.084579e+01   FALSE FALSE
## cvtd_timestamp            1.011111  1.698370e-01   FALSE FALSE
## new_window               48.898305  1.698370e-02   FALSE  TRUE
## num_window                1.040000  7.269022e+00   FALSE FALSE
## roll_belt                 1.097378  8.721128e+00   FALSE FALSE
## pitch_belt                1.061404  1.373981e+01   FALSE FALSE
## yaw_belt                  1.101639  1.437670e+01   FALSE FALSE
## total_accel_belt          1.089256  2.462636e-01   FALSE FALSE
## kurtosis_roll_belt        2.000000  1.944633e+00   FALSE FALSE
## kurtosis_picth_belt       1.000000  1.672894e+00   FALSE FALSE
## kurtosis_yaw_belt         0.000000  0.000000e+00    TRUE  TRUE
## skewness_roll_belt        1.000000  1.936141e+00   FALSE FALSE
## skewness_roll_belt.1      1.500000  1.757812e+00   FALSE FALSE
## skewness_yaw_belt         0.000000  0.000000e+00    TRUE  TRUE
## max_roll_belt             1.666667  1.180367e+00   FALSE FALSE
## max_picth_belt            1.657895  1.698370e-01   FALSE FALSE
## max_yaw_belt              1.055556  4.840353e-01   FALSE FALSE
## min_roll_belt             1.333333  1.078465e+00   FALSE FALSE
## min_pitch_belt            2.218750  1.103940e-01   FALSE FALSE
## min_yaw_belt              1.055556  4.840353e-01   FALSE FALSE
## amplitude_roll_belt       1.100000  7.982337e-01   FALSE FALSE
## amplitude_pitch_belt      3.567568  9.341033e-02   FALSE FALSE
## amplitude_yaw_belt        0.000000  8.491848e-03    TRUE  TRUE
## var_total_accel_belt      1.294118  3.991168e-01   FALSE FALSE
## avg_roll_belt             1.000000  1.129416e+00   FALSE FALSE
## stddev_roll_belt          1.103448  4.330842e-01   FALSE FALSE
## var_roll_belt             1.460000  5.264946e-01   FALSE FALSE
## avg_pitch_belt            1.200000  1.307745e+00   FALSE FALSE
## stddev_pitch_belt         1.026316  2.972147e-01   FALSE FALSE
## var_pitch_belt            1.363636  3.821332e-01   FALSE FALSE
## avg_yaw_belt              1.000000  1.384171e+00   FALSE FALSE
## stddev_yaw_belt           1.551724  3.651495e-01   FALSE FALSE
## var_yaw_belt              2.000000  8.237092e-01   FALSE FALSE
## gyros_belt_x              1.059627  1.095448e+00   FALSE FALSE
## gyros_belt_y              1.180442  5.604620e-01   FALSE FALSE
## gyros_belt_z              1.102589  1.341712e+00   FALSE FALSE
## accel_belt_x              1.035011  1.324728e+00   FALSE FALSE
## accel_belt_y              1.073876  1.163383e+00   FALSE FALSE
## accel_belt_z              1.085437  2.411685e+00   FALSE FALSE
## magnet_belt_x             1.085586  2.522079e+00   FALSE FALSE
## magnet_belt_y             1.103723  2.411685e+00   FALSE FALSE
## magnet_belt_z             1.000000  3.507133e+00   FALSE FALSE
## roll_arm                 50.950000  1.944633e+01   FALSE FALSE
## pitch_arm                92.681818  2.253736e+01   FALSE FALSE
## yaw_arm                  31.353846  2.145890e+01   FALSE FALSE
## total_accel_arm           1.005484  5.519701e-01   FALSE FALSE
## var_accel_arm             2.500000  1.961617e+00   FALSE FALSE
## avg_roll_arm             44.000000  1.638927e+00   FALSE  TRUE
## stddev_roll_arm          44.000000  1.638927e+00   FALSE  TRUE
## var_roll_arm             44.000000  1.638927e+00   FALSE  TRUE
## avg_pitch_arm            44.000000  1.638927e+00   FALSE  TRUE
## stddev_pitch_arm         44.000000  1.638927e+00   FALSE  TRUE
## var_pitch_arm            44.000000  1.638927e+00   FALSE  TRUE
## avg_yaw_arm              44.000000  1.638927e+00   FALSE  TRUE
## stddev_yaw_arm           45.000000  1.630435e+00   FALSE  TRUE
## var_yaw_arm              45.000000  1.630435e+00   FALSE  TRUE
## gyros_arm_x               1.003106  5.239470e+00   FALSE FALSE
## gyros_arm_y               1.396226  3.048573e+00   FALSE FALSE
## gyros_arm_z               1.123810  1.944633e+00   FALSE FALSE
## accel_arm_x               1.099010  6.462296e+00   FALSE FALSE
## accel_arm_y               1.107692  4.449728e+00   FALSE FALSE
## accel_arm_z               1.157895  6.428329e+00   FALSE FALSE
## magnet_arm_x              1.000000  1.108186e+01   FALSE FALSE
## magnet_arm_y              1.076923  7.192595e+00   FALSE FALSE
## magnet_arm_z              1.061538  1.052989e+01   FALSE FALSE
## kurtosis_roll_arm         1.000000  1.630435e+00   FALSE FALSE
## kurtosis_picth_arm        1.000000  1.621943e+00   FALSE FALSE
## kurtosis_yaw_arm          2.000000  1.953125e+00   FALSE FALSE
## skewness_roll_arm         1.000000  1.630435e+00   FALSE FALSE
## skewness_pitch_arm        1.000000  1.621943e+00   FALSE FALSE
## skewness_yaw_arm          1.000000  1.944633e+00   FALSE FALSE
## max_roll_arm             14.666667  1.537024e+00   FALSE FALSE
## max_picth_arm             7.333333  1.409647e+00   FALSE FALSE
## max_yaw_arm               1.153846  4.161005e-01   FALSE FALSE
## min_roll_arm             14.666667  1.452106e+00   FALSE FALSE
## min_pitch_arm            14.666667  1.486073e+00   FALSE FALSE
## min_yaw_arm               1.200000  3.141984e-01   FALSE FALSE
## amplitude_roll_arm       22.000000  1.554008e+00   FALSE  TRUE
## amplitude_pitch_arm      15.000000  1.520041e+00   FALSE FALSE
## amplitude_yaw_arm         1.400000  4.076087e-01   FALSE FALSE
## roll_dumbbell             1.064935  8.738961e+01   FALSE FALSE
## pitch_dumbbell            2.231707  8.519022e+01   FALSE FALSE
## yaw_dumbbell              1.093333  8.673573e+01   FALSE FALSE
## kurtosis_roll_dumbbell    2.000000  1.978601e+00   FALSE FALSE
## kurtosis_picth_dumbbell   2.000000  1.987092e+00   FALSE FALSE
## kurtosis_yaw_dumbbell     0.000000  0.000000e+00    TRUE  TRUE
## skewness_roll_dumbbell    2.000000  1.978601e+00   FALSE FALSE
## skewness_pitch_dumbbell   1.000000  1.978601e+00   FALSE FALSE
## skewness_yaw_dumbbell     0.000000  0.000000e+00    TRUE  TRUE
## max_roll_dumbbell         1.333333  1.825747e+00   FALSE FALSE
## max_picth_dumbbell        1.000000  1.817255e+00   FALSE FALSE
## max_yaw_dumbbell          1.083333  4.840353e-01   FALSE FALSE
## min_roll_dumbbell         1.333333  1.757812e+00   FALSE FALSE
## min_pitch_dumbbell        1.000000  1.851223e+00   FALSE FALSE
## min_yaw_dumbbell          1.083333  4.840353e-01   FALSE FALSE
## amplitude_roll_dumbbell   4.500000  1.919158e+00   FALSE FALSE
## amplitude_pitch_dumbbell  4.500000  1.910666e+00   FALSE FALSE
## amplitude_yaw_dumbbell    0.000000  8.491848e-03    TRUE  TRUE
## total_accel_dumbbell      1.107807  3.566576e-01   FALSE FALSE
## var_accel_dumbbell        3.333333  1.910666e+00   FALSE FALSE
## avg_roll_dumbbell         1.500000  1.978601e+00   FALSE FALSE
## stddev_roll_dumbbell      9.000000  1.936141e+00   FALSE FALSE
## var_roll_dumbbell         9.000000  1.936141e+00   FALSE FALSE
## avg_pitch_dumbbell        1.500000  1.978601e+00   FALSE FALSE
## stddev_pitch_dumbbell     9.000000  1.936141e+00   FALSE FALSE
## var_pitch_dumbbell        9.000000  1.936141e+00   FALSE FALSE
## avg_yaw_dumbbell          1.500000  1.978601e+00   FALSE FALSE
## stddev_yaw_dumbbell       9.000000  1.936141e+00   FALSE FALSE
## var_yaw_dumbbell          9.000000  1.936141e+00   FALSE FALSE
## gyros_dumbbell_x          1.084034  1.919158e+00   FALSE FALSE
## gyros_dumbbell_y          1.314706  2.241848e+00   FALSE FALSE
## gyros_dumbbell_z          1.146974  1.630435e+00   FALSE FALSE
## accel_dumbbell_x          1.045685  3.422215e+00   FALSE FALSE
## accel_dumbbell_y          1.066225  3.744905e+00   FALSE FALSE
## accel_dumbbell_z          1.155405  3.345788e+00   FALSE FALSE
## magnet_dumbbell_x         1.160000  8.797554e+00   FALSE FALSE
## magnet_dumbbell_y         1.196429  6.929348e+00   FALSE FALSE
## magnet_dumbbell_z         1.060870  5.528193e+00   FALSE FALSE
## roll_forearm             11.658291  1.475883e+01   FALSE FALSE
## pitch_forearm            59.487179  2.107677e+01   FALSE FALSE
## yaw_forearm              16.330986  1.422385e+01   FALSE FALSE
## kurtosis_roll_forearm     2.000000  1.528533e+00   FALSE FALSE
## kurtosis_picth_forearm    1.000000  1.528533e+00   FALSE FALSE
## kurtosis_yaw_forearm      0.000000  0.000000e+00    TRUE  TRUE
## skewness_roll_forearm     1.000000  1.537024e+00   FALSE FALSE
## skewness_pitch_forearm    4.000000  1.503057e+00   FALSE FALSE
## skewness_yaw_forearm      0.000000  0.000000e+00    TRUE  TRUE
## max_roll_forearm         18.333333  1.426630e+00   FALSE FALSE
## max_picth_forearm         4.230769  8.661685e-01   FALSE FALSE
## max_yaw_forearm           1.100000  2.632473e-01   FALSE FALSE
## min_roll_forearm         27.500000  1.435122e+00   FALSE  TRUE
## min_pitch_forearm         3.666667  8.916440e-01   FALSE FALSE
## min_yaw_forearm           1.100000  2.632473e-01   FALSE FALSE
## amplitude_roll_forearm   18.333333  1.426630e+00   FALSE FALSE
## amplitude_pitch_forearm   3.294118  9.341033e-01   FALSE FALSE
## amplitude_yaw_forearm     0.000000  8.491848e-03    TRUE  TRUE
## total_accel_forearm       1.117105  5.774457e-01   FALSE FALSE
## var_accel_forearm         4.000000  1.978601e+00   FALSE FALSE
## avg_roll_forearm         27.500000  1.537024e+00   FALSE  TRUE
## stddev_roll_forearm      57.000000  1.528533e+00   FALSE  TRUE
## var_roll_forearm         57.000000  1.528533e+00   FALSE  TRUE
## avg_pitch_forearm        55.000000  1.545516e+00   FALSE  TRUE
## stddev_pitch_forearm     27.500000  1.537024e+00   FALSE  TRUE
## var_pitch_forearm        55.000000  1.545516e+00   FALSE  TRUE
## avg_yaw_forearm          55.000000  1.545516e+00   FALSE  TRUE
## stddev_yaw_forearm       56.000000  1.537024e+00   FALSE  TRUE
## var_yaw_forearm          56.000000  1.537024e+00   FALSE  TRUE
## gyros_forearm_x           1.024169  2.377717e+00   FALSE FALSE
## gyros_forearm_y           1.107143  6.003736e+00   FALSE FALSE
## gyros_forearm_z           1.128814  2.386209e+00   FALSE FALSE
## accel_forearm_x           1.076923  6.547215e+00   FALSE FALSE
## accel_forearm_y           1.178571  8.160666e+00   FALSE FALSE
## accel_forearm_z           1.021277  4.662024e+00   FALSE FALSE
## magnet_forearm_x          1.040816  1.206692e+01   FALSE FALSE
## magnet_forearm_y          1.102041  1.531080e+01   FALSE FALSE
## magnet_forearm_z          1.025641  1.334918e+01   FALSE FALSE
## classe                    1.469065  4.245924e-02   FALSE FALSE
```

Removing the predictors which are not good:

```r
train1<- train1[,!nzv$nzv]; valid<- valid[,!nzv$nzv]; test <- test[,!nzv$nzv];
dim(train1); dim(valid)
```

```
## [1] 11776   130
```

```
## [1] 7846  130
```

```r
head(train1)
```

```
##     X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 1   1  carlitos           1323084231               788290 05/12/2011 11:23
## 2   2  carlitos           1323084231               808298 05/12/2011 11:23
## 5   5  carlitos           1323084232               196328 05/12/2011 11:23
## 9   9  carlitos           1323084232               484323 05/12/2011 11:23
## 10 10  carlitos           1323084232               484434 05/12/2011 11:23
## 14 14  carlitos           1323084232               576390 05/12/2011 11:23
##    num_window roll_belt pitch_belt yaw_belt total_accel_belt kurtosis_roll_belt
## 1          11      1.41       8.07    -94.4                3                 NA
## 2          11      1.41       8.07    -94.4                3                 NA
## 5          12      1.48       8.07    -94.4                3                 NA
## 9          12      1.43       8.16    -94.4                3                 NA
## 10         12      1.45       8.17    -94.4                3                 NA
## 14         12      1.42       8.21    -94.4                3                 NA
##    kurtosis_picth_belt skewness_roll_belt skewness_roll_belt.1 max_roll_belt
## 1                   NA                 NA                   NA            NA
## 2                   NA                 NA                   NA            NA
## 5                   NA                 NA                   NA            NA
## 9                   NA                 NA                   NA            NA
## 10                  NA                 NA                   NA            NA
## 14                  NA                 NA                   NA            NA
##    max_picth_belt max_yaw_belt min_roll_belt min_pitch_belt min_yaw_belt
## 1              NA           NA            NA             NA           NA
## 2              NA           NA            NA             NA           NA
## 5              NA           NA            NA             NA           NA
## 9              NA           NA            NA             NA           NA
## 10             NA           NA            NA             NA           NA
## 14             NA           NA            NA             NA           NA
##    amplitude_roll_belt amplitude_pitch_belt var_total_accel_belt avg_roll_belt
## 1                   NA                   NA                   NA            NA
## 2                   NA                   NA                   NA            NA
## 5                   NA                   NA                   NA            NA
## 9                   NA                   NA                   NA            NA
## 10                  NA                   NA                   NA            NA
## 14                  NA                   NA                   NA            NA
##    stddev_roll_belt var_roll_belt avg_pitch_belt stddev_pitch_belt
## 1                NA            NA             NA                NA
## 2                NA            NA             NA                NA
## 5                NA            NA             NA                NA
## 9                NA            NA             NA                NA
## 10               NA            NA             NA                NA
## 14               NA            NA             NA                NA
##    var_pitch_belt avg_yaw_belt stddev_yaw_belt var_yaw_belt gyros_belt_x
## 1              NA           NA              NA           NA         0.00
## 2              NA           NA              NA           NA         0.02
## 5              NA           NA              NA           NA         0.02
## 9              NA           NA              NA           NA         0.02
## 10             NA           NA              NA           NA         0.03
## 14             NA           NA              NA           NA         0.02
##    gyros_belt_y gyros_belt_z accel_belt_x accel_belt_y accel_belt_z
## 1          0.00        -0.02          -21            4           22
## 2          0.00        -0.02          -22            4           22
## 5          0.02        -0.02          -21            2           24
## 9          0.00        -0.02          -20            2           24
## 10         0.00         0.00          -21            4           22
## 14         0.00        -0.02          -22            4           21
##    magnet_belt_x magnet_belt_y magnet_belt_z roll_arm pitch_arm yaw_arm
## 1             -3           599          -313     -128      22.5    -161
## 2             -7           608          -311     -128      22.5    -161
## 5             -6           600          -302     -128      22.1    -161
## 9              1           602          -312     -128      21.7    -161
## 10            -3           609          -308     -128      21.6    -161
## 14            -8           598          -310     -128      21.4    -161
##    total_accel_arm var_accel_arm gyros_arm_x gyros_arm_y gyros_arm_z
## 1               34            NA        0.00        0.00       -0.02
## 2               34            NA        0.02       -0.02       -0.02
## 5               34            NA        0.00       -0.03        0.00
## 9               34            NA        0.02       -0.03       -0.02
## 10              34            NA        0.02       -0.03       -0.02
## 14              34            NA        0.02        0.00       -0.03
##    accel_arm_x accel_arm_y accel_arm_z magnet_arm_x magnet_arm_y magnet_arm_z
## 1         -288         109        -123         -368          337          516
## 2         -290         110        -125         -369          337          513
## 5         -289         111        -123         -374          337          506
## 9         -288         109        -122         -369          341          518
## 10        -288         110        -124         -376          334          516
## 14        -288         111        -124         -371          331          523
##    kurtosis_roll_arm kurtosis_picth_arm kurtosis_yaw_arm skewness_roll_arm
## 1                 NA                 NA               NA                NA
## 2                 NA                 NA               NA                NA
## 5                 NA                 NA               NA                NA
## 9                 NA                 NA               NA                NA
## 10                NA                 NA               NA                NA
## 14                NA                 NA               NA                NA
##    skewness_pitch_arm skewness_yaw_arm max_roll_arm max_picth_arm max_yaw_arm
## 1                  NA               NA           NA            NA          NA
## 2                  NA               NA           NA            NA          NA
## 5                  NA               NA           NA            NA          NA
## 9                  NA               NA           NA            NA          NA
## 10                 NA               NA           NA            NA          NA
## 14                 NA               NA           NA            NA          NA
##    min_roll_arm min_pitch_arm min_yaw_arm amplitude_pitch_arm amplitude_yaw_arm
## 1            NA            NA          NA                  NA                NA
## 2            NA            NA          NA                  NA                NA
## 5            NA            NA          NA                  NA                NA
## 9            NA            NA          NA                  NA                NA
## 10           NA            NA          NA                  NA                NA
## 14           NA            NA          NA                  NA                NA
##    roll_dumbbell pitch_dumbbell yaw_dumbbell kurtosis_roll_dumbbell
## 1       13.05217      -70.49400    -84.87394                     NA
## 2       13.13074      -70.63751    -84.71065                     NA
## 5       13.37872      -70.42856    -84.85306                     NA
## 9       13.15463      -70.42520    -84.91563                     NA
## 10      13.33034      -70.85059    -84.44602                     NA
## 14      13.41048      -70.99594    -84.28005                     NA
##    kurtosis_picth_dumbbell skewness_roll_dumbbell skewness_pitch_dumbbell
## 1                       NA                     NA                      NA
## 2                       NA                     NA                      NA
## 5                       NA                     NA                      NA
## 9                       NA                     NA                      NA
## 10                      NA                     NA                      NA
## 14                      NA                     NA                      NA
##    max_roll_dumbbell max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell
## 1                 NA                 NA               NA                NA
## 2                 NA                 NA               NA                NA
## 5                 NA                 NA               NA                NA
## 9                 NA                 NA               NA                NA
## 10                NA                 NA               NA                NA
## 14                NA                 NA               NA                NA
##    min_pitch_dumbbell min_yaw_dumbbell amplitude_roll_dumbbell
## 1                  NA               NA                      NA
## 2                  NA               NA                      NA
## 5                  NA               NA                      NA
## 9                  NA               NA                      NA
## 10                 NA               NA                      NA
## 14                 NA               NA                      NA
##    amplitude_pitch_dumbbell total_accel_dumbbell var_accel_dumbbell
## 1                        NA                   37                 NA
## 2                        NA                   37                 NA
## 5                        NA                   37                 NA
## 9                        NA                   37                 NA
## 10                       NA                   37                 NA
## 14                       NA                   37                 NA
##    avg_roll_dumbbell stddev_roll_dumbbell var_roll_dumbbell avg_pitch_dumbbell
## 1                 NA                   NA                NA                 NA
## 2                 NA                   NA                NA                 NA
## 5                 NA                   NA                NA                 NA
## 9                 NA                   NA                NA                 NA
## 10                NA                   NA                NA                 NA
## 14                NA                   NA                NA                 NA
##    stddev_pitch_dumbbell var_pitch_dumbbell avg_yaw_dumbbell
## 1                     NA                 NA               NA
## 2                     NA                 NA               NA
## 5                     NA                 NA               NA
## 9                     NA                 NA               NA
## 10                    NA                 NA               NA
## 14                    NA                 NA               NA
##    stddev_yaw_dumbbell var_yaw_dumbbell gyros_dumbbell_x gyros_dumbbell_y
## 1                   NA               NA             0.00            -0.02
## 2                   NA               NA             0.00            -0.02
## 5                   NA               NA             0.00            -0.02
## 9                   NA               NA             0.00            -0.02
## 10                  NA               NA             0.00            -0.02
## 14                  NA               NA             0.02            -0.02
##    gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z
## 1              0.00             -234               47             -271
## 2              0.00             -233               47             -269
## 5              0.00             -233               48             -270
## 9              0.00             -232               47             -269
## 10             0.00             -235               48             -270
## 14            -0.02             -234               48             -268
##    magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z roll_forearm
## 1               -559               293               -65         28.4
## 2               -555               296               -64         28.3
## 5               -554               292               -68         28.0
## 9               -549               292               -65         27.7
## 10              -558               291               -69         27.7
## 14              -554               295               -68         27.2
##    pitch_forearm yaw_forearm kurtosis_roll_forearm kurtosis_picth_forearm
## 1          -63.9        -153                    NA                     NA
## 2          -63.9        -153                    NA                     NA
## 5          -63.9        -152                    NA                     NA
## 9          -63.8        -152                    NA                     NA
## 10         -63.8        -152                    NA                     NA
## 14         -63.9        -151                    NA                     NA
##    skewness_roll_forearm skewness_pitch_forearm max_roll_forearm
## 1                     NA                     NA               NA
## 2                     NA                     NA               NA
## 5                     NA                     NA               NA
## 9                     NA                     NA               NA
## 10                    NA                     NA               NA
## 14                    NA                     NA               NA
##    max_picth_forearm max_yaw_forearm min_pitch_forearm min_yaw_forearm
## 1                 NA              NA                NA              NA
## 2                 NA              NA                NA              NA
## 5                 NA              NA                NA              NA
## 9                 NA              NA                NA              NA
## 10                NA              NA                NA              NA
## 14                NA              NA                NA              NA
##    amplitude_roll_forearm amplitude_pitch_forearm total_accel_forearm
## 1                      NA                      NA                  36
## 2                      NA                      NA                  36
## 5                      NA                      NA                  36
## 9                      NA                      NA                  36
## 10                     NA                      NA                  36
## 14                     NA                      NA                  36
##    var_accel_forearm gyros_forearm_x gyros_forearm_y gyros_forearm_z
## 1                 NA            0.03            0.00           -0.02
## 2                 NA            0.02            0.00           -0.02
## 5                 NA            0.02            0.00           -0.02
## 9                 NA            0.03            0.00           -0.02
## 10                NA            0.02            0.00           -0.02
## 14                NA            0.00           -0.02           -0.03
##    accel_forearm_x accel_forearm_y accel_forearm_z magnet_forearm_x
## 1              192             203            -215              -17
## 2              192             203            -216              -18
## 5              189             206            -214              -17
## 9              193             204            -214              -16
## 10             190             205            -215              -22
## 14             193             202            -214              -14
##    magnet_forearm_y magnet_forearm_z classe
## 1               654              476      A
## 2               661              473      A
## 5               655              473      A
## 9               653              476      A
## 10              656              473      A
## 14              659              478      A
```
Removing NAs

```r
cleantr <- train1[, colSums(is.na(train1)) == 0]
cleanVa <- valid[, colSums(is.na(valid)) == 0]
test<- test[,colSums(is.na(valid)) == 0]
```
Removing the class column from Training Set.

```r
finalTrain <- cleantr[, -(1:5)]
finalValid <- cleanVa[,-(1:5)]
```
## Creating model:

### Decision Tree

```r
set.seed(12345)
modFit <- train(classe ~ .,method="rpart", data = finalTrain)
modFit
```

```
## CART 
## 
## 11776 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa     
##   0.02432368  0.6310912  0.53332175
##   0.04358488  0.5114667  0.36147484
##   0.11449929  0.3270387  0.06354879
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was cp = 0.02432368.
```
Creating Decision Tree:

```r
fancyRpartPlot(modFit$finalModel)
```

![](Practical-Machine-Learning-Coursera-Project_files/figure-html/unnamed-chunk-9-1.png)<!-- -->
Now, Predicting class using ValidationSet:

```r
pred <- predict(modFit, newdata = finalValid)
cnfMatrix <- confusionMatrix(pred, data = as.factor(finalValid$classe))
cnfMatrix
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1886  152  189    0    5
##          B  286  824  408    0    0
##          C  165  123 1080    0    0
##          D  298  327  600    0   61
##          E   90  249  215    0  888
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5962          
##                  95% CI : (0.5853, 0.6071)
##     No Information Rate : 0.3473          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4838          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.6921   0.4919   0.4334       NA   0.9308
## Specificity            0.9324   0.8875   0.9462   0.8361   0.9196
## Pos Pred Value         0.8450   0.5428   0.7895       NA   0.6158
## Neg Pred Value         0.8506   0.8655   0.7820       NA   0.9897
## Prevalence             0.3473   0.2135   0.3176   0.0000   0.1216
## Detection Rate         0.2404   0.1050   0.1376   0.0000   0.1132
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.8123   0.6897   0.6898       NA   0.9252
```

```r
plot(cnfMatrix$table, col = cnfMatrix$byClass, main = paste("Decision Tree - Accuracy =", round(cnfMatrix$overall['Accuracy'], 3)))
```

![](Practical-Machine-Learning-Coursera-Project_files/figure-html/unnamed-chunk-10-1.png)<!-- -->

### Random Forest

```r
control <- trainControl(method="cv", number=5, verboseIter=TRUE)
modFit1 <- train(classe ~ .,method="rf", trControl=control, data = finalTrain)
```

```
## + Fold1: mtry= 2 
## - Fold1: mtry= 2 
## + Fold1: mtry=27 
## - Fold1: mtry=27 
## + Fold1: mtry=53 
## - Fold1: mtry=53 
## + Fold2: mtry= 2 
## - Fold2: mtry= 2 
## + Fold2: mtry=27 
## - Fold2: mtry=27 
## + Fold2: mtry=53 
## - Fold2: mtry=53 
## + Fold3: mtry= 2 
## - Fold3: mtry= 2 
## + Fold3: mtry=27 
## - Fold3: mtry=27 
## + Fold3: mtry=53 
## - Fold3: mtry=53 
## + Fold4: mtry= 2 
## - Fold4: mtry= 2 
## + Fold4: mtry=27 
## - Fold4: mtry=27 
## + Fold4: mtry=53 
## - Fold4: mtry=53 
## + Fold5: mtry= 2 
## - Fold5: mtry= 2 
## + Fold5: mtry=27 
## - Fold5: mtry=27 
## + Fold5: mtry=53 
## - Fold5: mtry=53 
## Aggregating results
## Selecting tuning parameters
## Fitting mtry = 27 on full training set
```

```r
modFit1
```

```
## Random Forest 
## 
## 11776 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 9421, 9421, 9420, 9421, 9421 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9909985  0.9886128
##   27    0.9964334  0.9954886
##   53    0.9935461  0.9918360
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```

Now, Predicting class using ValidationSet:

```r
pred1 <- predict(modFit1, newdata = finalValid)
cnfMatrix1 <- confusionMatrix(pred1, data = as.factor(finalValid$classe))
cnfMatrix1
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    2 1516    0    0    0
##          C    0    1 1367    0    0
##          D    0    0    3 1283    0
##          E    0    0    0    3 1439
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9989          
##                  95% CI : (0.9978, 0.9995)
##     No Information Rate : 0.2847          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9985          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9993   0.9978   0.9977   1.0000
## Specificity            1.0000   0.9997   0.9998   0.9995   0.9995
## Pos Pred Value         1.0000   0.9987   0.9993   0.9977   0.9979
## Neg Pred Value         0.9996   0.9998   0.9995   0.9995   1.0000
## Prevalence             0.2847   0.1933   0.1746   0.1639   0.1834
## Detection Rate         0.2845   0.1932   0.1742   0.1635   0.1834
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9996   0.9995   0.9988   0.9986   0.9998
```

```r
plot(cnfMatrix1$table, col = cnfMatrix1$byClass, main = paste("Random Forest - Accuracy =", round(cnfMatrix1$overall['Accuracy'], 3)))
```

![](Practical-Machine-Learning-Coursera-Project_files/figure-html/unnamed-chunk-12-1.png)<!-- -->

## Implementing Model on a Test Set:
As the accuracy of model created with Random Forest is high as compare to model created with Decision Tree. So, we will use the Random Forest Model to predict the Test Set class.

```r
predResult <- predict(modFit1, newdata = test)
predResult
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


