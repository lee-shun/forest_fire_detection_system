from pathlib import Path
from PIL import Image

inputPath = Path("giao/dataset/imgs/Jingling")
inputFiles = inputPath.glob("**/*.png")
outputPath = Path("giao/dataset/imgs/Jingling")
for f in inputFiles:
    outputFile = outputPath / Path(f.stem + ".jpg")
    im = Image.open(f)
    im.save(outputFile)
