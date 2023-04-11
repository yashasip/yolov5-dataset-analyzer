import cv2
import yaml


# paste path to image in yoloV5 dataset
IMAGE_PATH = ""

BOX_COLORS = [
    (0, 0, 128),  # navy
    (255, 255, 0),  # yellow
    (255, 0, 0),  # red
    (0, 255, 0),  # lime
    (255, 255, 255),  # white
    (255, 165, 0),  # orange
    (128, 0, 128),  # purple
    (128, 128, 128),  # gray
    (0, 255, 255),  # blue
    (128, 128, 0),  # olive
]
BOX_LINE_WIDTH = 1
IMAGE_EXT = ".jpg"
LABEL_EXT = ".txt"
IMAGE_DIR = "/images"
LABEL_DIR = "labels"
DATA_YAML = "/data.yaml"
SAVE_PATH = "./result" + IMAGE_EXT
ANNOTATION_TEXT_REQUIRED = True


def getImage(path: str) -> cv2.Mat:
    return cv2.imread(path)


def drawBoundingBox(image: cv2.Mat, vector: dict):
    x_centre = round(vector["x_centre"] * image.shape[1])
    y_centre = round(vector["y_centre"] * image.shape[0])
    width = round(vector["width"] * image.shape[1])
    height = round(vector["height"] * image.shape[0])
    x = x_centre - width // 2
    y = y_centre - height // 2
    cv2.rectangle(
        image, (x, y), (x + width, y + height), (vector["color"]), BOX_LINE_WIDTH
    )
    if ANNOTATION_TEXT_REQUIRED:
        cv2.putText(
            img=image,
            text=vector["class"],
            org=(x, y + 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 0, 0),
            thickness=2,
        )

    return image


def getPaths(imagePath: str) -> tuple[str, str, str]:
    databasePath, subset, _, imageName = imagePath.rsplit(sep="/", maxsplit=3)
    return databasePath, subset, imageName[:-4] + LABEL_EXT


def getVectors(path: str, classes: list[str]) -> list[dict]:
    lines = readFile(path=path)
    vectors = []
    for line in lines:
        label, x_centre, y_centre, width, height = line.split()
        vectorDict = {
            "label": label,
            "class": classes[int(label)],
            "x_centre": float(x_centre),
            "y_centre": float(y_centre),
            "width": float(width),
            "height": float(height),
            "color": BOX_COLORS[int(label)],
        }
        vectors.append(vectorDict)

    return vectors


def readFile(path: str) -> list[str]:
    with open(path) as file:
        return file.readlines()


def getClasses(path: str):
    with open(path) as dataYaml:
        data = yaml.safe_load(dataYaml)
    return data["names"]


def drawAnnotations(image: cv2.Mat, vectors: list[dict]):
    for vector in vectors:
        drawBoundingBox(image=image, vector=vector)
    return image


def saveImage(image: cv2.Mat, savePath: str = SAVE_PATH):
    cv2.imwrite(img=image, filename=savePath)


def main():
    image = getImage(IMAGE_PATH)
    databasePath, subset, labelName = getPaths(IMAGE_PATH)
    classes = getClasses("/".join([databasePath, DATA_YAML]))
    vectors = getVectors(
        path="/".join([databasePath, subset, LABEL_DIR, labelName]), classes=classes
    )
    resultingImage = drawAnnotations(image=image, vectors=vectors)
    saveImage(image=resultingImage)
    print("Annotation Marked")


if __name__ == "__main__":
    main()
