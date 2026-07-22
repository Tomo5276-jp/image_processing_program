Image Processing Program

概要
フロア案内板の画像を解析し、建物名・階層・施設名を抽出後、データベースへ保存するシステム
画像認識にはResNetを使用し、文字抽出にはPaddleOCRを使用している。抽出したデータはMySQL
に保存する。

構成
画像前処理(OpenCV)
|
建物名、階層、施設名の領域に分類(ResNet)
|
文字認識・抽出(PaddleOCR)
|
データベースに保存(MySQL)

