from VisualizeAnnotations import drawAnnotations, getImage, getVectors, saveImage

# enter image path and classes here
IMAGE_PATH = "./input.png"
CLASSES = ["0", "1"]

DEFAULT_LABELS_PATH = "./labels.txt"
SAVE_PATH = "./result.jpg"

def annotateImageWithLabel(imagePath = str, labelPath = str) -> None:
    image = getImage(imagePath)
    vectors = getVectors(labelPath,classes=CLASSES)
    image = drawAnnotations(image=image, vectors=vectors)
        
    saveImage(image=image, savePath=SAVE_PATH)


if __name__ == "__main__":
    annotateImageWithLabel(imagePath=IMAGE_PATH, labelPath=DEFAULT_LABELS_PATH)
