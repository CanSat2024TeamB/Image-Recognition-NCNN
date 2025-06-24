# Image-Recognition-NCNN
ncnn（低スペックデバイス向けのCNNワークフレーム 詳細: <https://github.com/Tencent/ncnn>）を用いた物体検出のライブラリ

実装はC++で行い、pytbind11を使ってコンパイルすることでメインのコードからPythonで動かしています。一般的な三角コーンの画像をYolov9で学習させ、ncnn用のモデルに変換して使用しました。

***

以下、学習の詳細および最終的なモデルの特性

* 学習画像
  * 画像サイズ: 640 × 640 px
  * 画像枚数（train）: 18,774枚
* モデル特性
  * 入力サイズ: 320 × 320 px
  * F値曲線
![F1_curve](https://github.com/user-attachments/assets/ef3feac6-d07d-4d16-8366-60cc4a893557)
  * P曲線
![P_curve](https://github.com/user-attachments/assets/c1a8867a-ca8e-49b6-a5bb-c8b23ca3de13)
  * PR曲線
![PR_curve](https://github.com/user-attachments/assets/42523e06-1f59-4c0b-ae73-f4164eeba7b0)
  * R曲線
![R_curve](https://github.com/user-attachments/assets/1e8fb23f-39f6-4832-9621-be65917b9490)
