# deep_ensemble_learning
Deep Ensemble Learning for Human Action Recognition in Still Images

The code is provided by Xiangchun Yu.
Email:yxchspring0209@foxmail.com
Data: November 08, 2019
###################################################################
Step1. Please extract the compressed file to the current folder.
Such as: deep_ensemble_learning/Action_ForCNN/train...

Step2. Train the VGG16, VGG16_NCNN, VGG19,VGG19_NCNN,ResNet50,ResNet50_NCNN models using the following python scripts,
Main1_VGG16.py
Main1_VGG16_NCNN.py
Main1_VGG19.py
Main1_VGG19_NCNN.py
Main3_ResNet50.py
Main3_ResNet50_NCNN.py
And use the Main4_Evaluate_Results.py test the results of the models mentioned above.

Step3. Train the DELWO1-DELWO3 models using the following python scripts,
Main5_DELWO1.py
Main5_DELWO2.py
Main5_DELWO3.py
And use the Main6_Evaluate_Results_for_DELWOs.py test the results of the models mentioned above.

Step4. Evaluate the models' performance using DELVS1-DELVS3 using the following python scripts,
Main7_DELVS1.py
Main7_DELVS2.py
Main7_DELVS3.py
Note: The final results will be obtained directly using the above python scripts.
