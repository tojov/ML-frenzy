from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN 
from matplotlib.patches import Rectangle, Circle
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def markFaces(fileName, resultList):
	img = pyplot.imread(fileName)
	pyplot.imshow(img)
	ax = pyplot.gca()
	for result in resultList:
		x, y, w, h = result['box']
		rect = Rectangle((x, y), w, h, fill = False, color='green')
		ax.add_patch(rect)
		for key, value in result['keypoints'].items():
			dot = Circle(value, radius = 2, color = 'red')
			ax.add_patch(dot)
	pyplot.show()

fileName = 'test2.jpg'
img = pyplot.imread(fileName)
detector = MTCNN()
faces = detector.detect_faces(img)
markFaces(fileName, faces)

