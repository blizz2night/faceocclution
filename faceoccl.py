# -*- coding: utf-8 -*-

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')


def detectFace(frame):
	faces=[]
	result = False
	temp = []
	temp = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	cv2.equalizeHist(temp,temp)
	faces = face_cascade.detectMultiScale(temp, 1.1, 2, 0 | cv2.CASCADE_SCALE_IMAGE, (1, 1))
	if len(faces)>0:
		result=True
	return result, faces

def detectMask(frame,face):
	result = False
	# i = 0
	fmat = frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
	# cv2.imshow(str(i), fmat)
	dy = face[1]+fmat.shape[1] / 7 * 3
	dx = face[0]
	fmat = fmat[fmat.shape[1] / 7 * 3:fmat.shape[1] / 5 * 3]

	edges = cv2.Canny(fmat, 50, 150)

	# lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
	lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 46, minLineLength=edges.shape[1] / 4, maxLineGap=edges.shape[1] / 20)
	# print lines, edges.shape[1] / 5
	if lines is not None:
		lines += (dx, dy, dx, dy)
		result = True
	else:
		lines = []
	return result, lines

def detectMouth(frame,face):
	#print face
	result = False
	mouths = []
	fmat = frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
	dy = face[1]+fmat.shape[1]/3*2
	dx = face[0]
	fmat = fmat[fmat.shape[1]/3*2:]
	fmat = cv2.cvtColor(fmat,cv2.COLOR_BGR2GRAY)
	fmat = cv2.equalizeHist(fmat)
	#print fmat
	mouths = mouth_cascade.detectMultiScale(fmat, 1.1, 3, 0 | cv2.CASCADE_SCALE_IMAGE, (30, 30))
	if len(mouths)>0:
		mouths += (dx, dy, 0, 0)
		result = True
	return result, mouths

def detectSunglass(frame,face):
	result = False
	fmat = frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
	dy = face[1]+fmat.shape[1] / 4
	dx = face[0]
	fmat = fmat[fmat.shape[1] / 4:fmat.shape[1] / 7 * 4]
	fmat = cv2.cvtColor(fmat, cv2.COLOR_BGR2YCR_CB)
	y, u, v = cv2.split(fmat)
	count = np.count_nonzero((u >= 133) & (u <= 173) & (v >= 77) & (v <= 127))
	ratio = float(count) / (fmat.shape[0]*fmat.shape[1])
	if ratio < 0.5:
		result = True
	return result, ratio

def detectEye(frame, face):
	result = False
	eyes = []
	fmat = frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
	dy = face[1]+fmat.shape[1]/4
	dx = face[0]
	fmat = fmat[fmat.shape[1]/4:fmat.shape[1] / 7 * 4]
	fmat = cv2.cvtColor(fmat, cv2.COLOR_BGR2GRAY)
	fmat = cv2.equalizeHist(fmat)
	eyes = eye_cascade.detectMultiScale(fmat, 1.1, 2, 0 | cv2.CASCADE_SCALE_IMAGE, (30, 30))
	if len(eyes) > 0:
		eyes += (dx, dy, 0, 0)
		result = True
	return result, eyes

def isMouthCovered(frame, face):
	result = False
	ret_mask,lines = detectMask(frame,face)
	ret_mouth,mouths = detectMouth(frame,face)
	if ret_mask and ~ret_mouth:
		result = True
	return result, lines, mouths

def isEyeCovered(frame, face):
	result = False
	ret_sg, ratio = detectSunglass(frame, face)
	ret_eye, eyes = detectEye(frame, face)
	if ret_sg and ~ret_eye:
		result = True
	return result, ratio, eyes

def detectMouthCovered(frame):
	result = False
	ret, faces = detectFace(frame)
	occlfaces = []
	if ret:
		for face in faces:
			ret_mouth, lines, mouths = isMouthCovered(frame, face)
			if ret_mouth:
				result = True
				occlfaces.append(face)
	return result, occlfaces

def detectEyeCovered(frame):
	result = False
	ret, faces = detectFace(frame)
	occlfaces = []
	if ret:
		for face in faces:
			ret_eye, ratio, eyes = isEyeCovered(frame, face)
			if ret_eye:
				result = True
				occlfaces.append(face)
	return result, occlfaces

def detectFaceOcclution(frame):
	result = False
	ret, faces = detectFace(frame)
	occlfaces = []
	if ret:
		for face in faces:
			ret_mouth, lines, mouths = isMouthCovered(frame, face)
			ret_eye, ratio, eyes = isEyeCovered(frame, face)
			if ret_mouth or ret_eye:
				result = True
				occlfaces.append(face)
	return result, occlfaces



cap = cv2.VideoCapture(0)
while True:
	ret, frame = cap.read()
	fshow = frame
	if not ret:
		break
	ret, faces = detectFaceOcclution(frame)
	if ret:
		print '面部遮挡'
		for face in faces:
			cv2.rectangle(fshow, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 0, 255), 2)
	# #检测人脸
	# ret, faces = detectFace(frame)
	# if ret:
	# 	for face in faces:
	# 		cv2.rectangle(fshow, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 0, 255), 2)
	# 		# #检测口罩
	# 		# ret, lines = detectMask(frame, face)
	# 		ret, lines, mouths = isMouthCovered(frame, face)
	# 		if ret:
	# 			print '戴口罩'
	# 			# for line in lines:
	# 			# 	x1, y1, x2, y2 = line[0]
	# 			# 	cv2.line(fshow, (x1, y1), (x2, y2), (0, 255, 0), 2)
	# 		# #检测嘴
	# 		# ret, mouths = detectMouth(frame, face)
	# 		# if ret:
	# 		# 	for mouth in mouths:
	# 		# 		cv2.rectangle(fshow, (mouth[0], mouth[1]), (mouth[0]+mouth[2], mouth[1]+mouth[3]), (255, 0, 255), 2)
	# 		# print ratio
	# 		# #检测墨镜
	# 		# ret, ratio = detectSunglass(frame, face)
	# 		# #检测眼睛
	# 		# ret, eyes = detectEye(frame, face)
	# 		ret, ratio, eyes = isEyeCovered(frame, face)
	# 		if ret:
	# 			print '戴墨镜'
	# 			# for eye in eyes:
	# 			# 	cv2.rectangle(frame, (eye[0], eye[1]), (eye[0]+eye[2], eye[1]+eye[3]), (255, 255, 0), 2)
	# 		# print
	cv2.imshow('1', fshow)
	k = cv2.waitKey(5) & 0xff
	if k == 27:
		break