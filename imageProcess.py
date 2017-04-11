import pygame.image as image
from struct import unpack
from six.moves import xrange
def loadImageFromFile(path):
	picSurface = image.load(path)
	pixel = image.tostring(picSurface, "RGB")
	#assert isinstance(pixel, str)
	return picSurface, pixel


def grayfie(buffer, w, h):
	newBuffer = [0 for i in range(w * h)]
	for y in range(h):
		for x in range(w):
			#r = unpack('B', bytes([ buffer[x * 3 + y * 72 * 3] ]))[0]
			#g = unpack('B', bytes([buffer[x * 3 + 1 + y * 72 * 3] ]))[0]
			#b = unpack('B', bytes([buffer[x * 3 + 2 + y * 72 * 3] ]))[0]
			r = buffer[x * 3 + y * 72 * 3]
			g = buffer[x * 3 + 1 + y * 72 * 3]
			b = buffer[x * 3 + 2 + y * 72 * 3]

			if r == g and (r != 255 or g != 255 or g != 255) and b != 0:
				newBuffer[x + y * w] = 255 - min(255, max(0, (r + g + b) / 3 - 60))
			else:
				newBuffer[x + y * w] = 0
	return newBuffer


def denoise(buffer, w, h):
	for y in range(h):
		for x in range(w):
			if (y == 0 or y == h - 1 or x == 0 or x == w - 1):
				buffer[x + y * w] = 0
				continue
			num = 0.0
			for ud in range(-1, 2):
				for lr in range(-1, 2):
					num += buffer[x + lr + (y + ud) * w] / 255.0
			if num < 3.5:
				buffer[x + y * w] = 0


def gauss(buffer, w, h):
	newBuffer = [0 for i in xrange(w * h)]
	for y in range (1, h - 1):
		for x in range(1, w - 1):
			center = 0.0
			sumUp = 0.0
			for ud in range(-1, 2):
				for lr in range(-1, 2):
					if ud == 0 and lr == 0:
						center = buffer[x + lr + (y + ud) * w] * 0.7
					else:
						sumUp += buffer[x + lr + (y + ud) * w] * 0.0375
			newBuffer[x + y * w] = min(255, sumUp + center)
	for i in xrange(w * h):
		buffer[i] = int(newBuffer[i])


def loadImageFromPixel():
	pass

def preProcess(pixelStr):
	pixel = grayfie(pixelStr, 72, 27)
	denoise(pixel, 72, 27)
	denoise(pixel, 72, 27)
	gauss(pixel, 72, 27)
	return pixel


def postProcess(bufferSeperateImg):
	tmp = [[0 for j in range(25 * 25)] for i in range(4)]
	maxX = 0
	minX = 40
	maxY = 0
	minY = 40
	for i in range(4):
		denoise(bufferSeperateImg[i], 40, 40)
		for y in range(40):
			for x in range(40):
				if bufferSeperateImg[i][x + y * 40]:
					if (x < minX):
						minX = x
					if (x > maxX):
						maxX = x
					if (y < minY):
						minY = y
					if (y > maxY):
						maxY = y

		w = maxX - minX
		h = maxY - minY
		xOffset = (25 - w) / 2
		yOffset = (25 - h) / 2
		for y in range(minY, maxY + 1):
			for x in range(minX, maxX + 1):
				tmp[i][xOffset + x - minX + (y - minY + yOffset) * 25] = bufferSeperateImg[i][x + y * 40]
	return tmp