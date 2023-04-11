import os
import yaml

# paste the path to the dataset folder (include dataset folder name)
DATASET_PATH = ""

TRAIN_PATH = DATASET_PATH + "/train"
TEST_PATH = DATASET_PATH + "/test"
VALID_PATH = DATASET_PATH + "/valid"
DATA_YAML_PATH = DATASET_PATH + "/data.yaml"
IMAGES_DIR = "/images"
LABELS_DIR = "/labels"
IMAGE_EXT = ".jpg"
LABEL_EXT = ".txt"


def getFileCountInDirectory(path: str, ext: str) -> int:
    files = os.listdir(path=path)
    fileCount = 0
    for file in files:
        if file.endswith(ext):
            fileCount += 1

    return fileCount


def getFileNamesInDirectory(path: str, ext: str, stripExt: bool = False) -> set[str]:
    files = os.listdir(path=path)
    fileNames = set()
    for file in files:
        if file.endswith(ext):
            if stripExt:
                fileNames.add(file[: -len(ext) + 1])
            else:
                fileNames.add(file)
    return fileNames


def getImageNamesInDirectory(
    path: str, ext: str = IMAGE_EXT, stripExt: bool = False
) -> set[str]:
    return getFileNamesInDirectory(path=path + IMAGES_DIR, ext=ext, stripExt=stripExt)


def getLabelNamesInDirectory(
    path: str, ext: str = LABEL_EXT, stripExt: bool = False
) -> set[str]:
    return getFileNamesInDirectory(path=path + LABELS_DIR, ext=ext, stripExt=stripExt)


def getImageCountInDirectory(path: str, ext: str = IMAGE_EXT) -> int:
    return getFileCountInDirectory(path=path, ext=ext)


def getLabelFileCountInDirectory(path: str, ext: str = LABEL_EXT) -> int:
    return getFileCountInDirectory(path=path, ext=ext)


def readYamlFile(path: str) -> dict[str, str | int]:
    with open(path) as dataYaml:
        data = yaml.safe_load(dataYaml)
    return data


def readFile(path: str) -> list[str]:
    with open(path) as file:
        return file.readlines()


def getClassBasedImageCount(path: str, classes: list) -> dict[str, int]:
    labelFiles = getLabelNamesInDirectory(path)
    classCountMap = {}
    for labelFile in labelFiles:
        lines = readFile(f"{path}/{LABELS_DIR}/{labelFile}")
        for line in lines:
            label = int(line.split(maxsplit=1)[0])
            classCountMap.setdefault(classes[label], 0)
            classCountMap[classes[label]] += 1

    return classCountMap


def getDatasetClasses() -> list[str]:
    dataMap = readYamlFile(path= DATA_YAML_PATH)
    if dataMap["nc"] != len(dataMap["names"]):
        raise Exception("Classes Count not equal to Classes Found in data.yaml")

    return dataMap["names"]


def printClassImageCount(classCountMap: dict[str, int]):
    totalCount = sum([value for value in classCountMap.values()])
    print(f"Total Class Count: {totalCount}")
    for classValue, count in classCountMap.items():
        print(f"{classValue}: {count} {percentage(count, totalCount)}%")


def percentage(part: int, whole: int):
    return round(part / whole * 100, 2)


def getTotalClassCount(
    trainDict: dict[str, int],
    validDict: dict[str, int],
    testDict: dict[str, int],
    classes: list[str],
):
    classCountMap = {}
    for classValue in classes:
        classCountMap[classValue] = (
            trainDict[classValue] + validDict[classValue] + testDict[classValue]
        )
    return classCountMap


def validateSubset(path: str):
    imageNames = getImageNamesInDirectory(path=path, stripExt=True)
    labelNames = getLabelNamesInDirectory(path=path, stripExt=True)
    imagesWithNoLabel = imageNames.difference(labelNames)
    labelsWithNoImage = labelNames.difference(imageNames)
    if len(imagesWithNoLabel) != 0:
        print(f"Path: {path}, Validation Failed for Images!")
        print(f"Label not found for Image: {' ,'.join(imagesWithNoLabel)}")
    if len(labelsWithNoImage) != 0:
        print(f"Path: {path}, Validation Failed for Labels!")
        print(f"Image not found for Label: {' ,'.join(labelsWithNoImage)}")


def validateDataset():
    print("\nValidating Dataset...")
    validateSubset(path=TRAIN_PATH)
    validateSubset(path=VALID_PATH)
    validateSubset(path=TEST_PATH)


def processDataset() -> None:
    print(f"Dataset Name: {DATASET_PATH.rsplit(sep = '/', maxsplit = 1)[1]}")
    classes = getDatasetClasses()
    print(f"Class: {', '.join(classes)}\n")

    trainImages = getImageCountInDirectory(path= TRAIN_PATH + IMAGES_DIR)
    validImages = getImageCountInDirectory(path= VALID_PATH + IMAGES_DIR)
    testImages = getImageCountInDirectory(path= TEST_PATH + IMAGES_DIR)
    total_images = trainImages + validImages + testImages

    print(f"Total Images: {total_images}")
    print(f"Training Images: {trainImages} {percentage(trainImages, total_images)}%")
    print(f"Valid Images: {validImages} {percentage(validImages, total_images)}%")
    print(f"Testing Images: {testImages} {percentage(testImages, total_images)}%")

    print("\nClass Image Count:")
    trainClassCountMap = getClassBasedImageCount(
        path= TRAIN_PATH, classes=classes
    )
    validClassCountMap = getClassBasedImageCount(
        path=VALID_PATH, classes=classes
    )
    testClassCountMap = getClassBasedImageCount(
        path= TEST_PATH, classes=classes
    )
    print("\nTraining Set:")
    printClassImageCount(trainClassCountMap)
    print("\nValidation Set:")
    printClassImageCount(validClassCountMap)
    print("\nTesting Set:")
    printClassImageCount(testClassCountMap)

    print("\nTotal Classes Count:")
    totalClassCountMap = getTotalClassCount(
        trainDict=trainClassCountMap,
        validDict=validClassCountMap,
        testDict=testClassCountMap,
        classes=classes,
    )
    printClassImageCount(classCountMap=totalClassCountMap)


def main():
    processDataset()
    validateDataset()
    print("\nCompleted!")


if __name__ == "__main__":
    main()
