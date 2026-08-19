from roboflow import Roboflow

rf = Roboflow(
    api_key="6UQ23zXBHxqqYn2fGWRt"
)

project = rf.workspace("-nd5fh").project("object-detection-g5zgo")

version = project.version(5)

dataset = version.download("yolov8")

print("データセットのダウンロードが完了しました")