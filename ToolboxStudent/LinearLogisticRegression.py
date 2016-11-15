#    Copyright 2016 Stefan Steidl
#    Friedrich-Alexander-Universität Erlangen-Nürnberg
#    Lehrstuhl für Informatik 5 (Mustererkennung)
#    Martensstraße 3, 91058 Erlangen, GERMANY
#    stefan.steidl@fau.de


#    This file is part of the Python Classification Toolbox.
#
#    The Python Classification Toolbox is free software: 
#    you can redistribute it and/or modify it under the terms of the 
#    GNU General Public License as published by the Free Software Foundation, 
#    either version 3 of the License, or (at your option) any later version.
#
#    The Python Classification Toolbox is distributed in the hope that 
#    it will be useful, but WITHOUT ANY WARRANTY; without even the implied
#    warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
#    See the GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with the Python Classification Toolbox.  
#    If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import numpy.matlib
import math


class LinearLogisticRegression(object):

	def __init__(self, learningRate = 0.01, maxIterations = 100):
		self.__learningRate = learningRate
		self.__maxIterations = maxIterations
		#3 factors for theta since dec. bound. is linear
		self.__theta = np.empty(3)
		return None




	#estimate from labeled training samples; X in each row a new sample point is given; y np-Vector with classes (double values)
	#compute theta with MLM (newton-raphson)(from training samples)
	def fit(self, X, y):
		#jeden sample in X am Ende mit 1 erweitern (3D), um spaeter mit theta multipliz. zu koennen
		self.__X = np.insert(X, 2, 1, axis=1)
		self.__y = y
		#Anzahl eintraege von X (anzahl samples)
		self.__entries = len(X)
		#Maximal zwei moegliche Klassen
		self.__class = np.empty(2)
		self.__class[0] = min(y)
		self.__class[1] = max(y)
		print(X)
		print(y)
		
		#Newton iteration
		for i in range(self.__maxIterations):
			#Vector von sigmoid fct
			sigmoid = self.gFunc(self.__X, self.__theta)
			#print(max(sigmoid))
			#print(min(sigmoid))
			gradient = ((y-sigmoid).dot(self.__X))
			temp=(1-sigmoid)*sigmoid
			temp = temp[:, np.newaxis]
			temp = self.__X*temp
			self.__X = self.__X.transpose()
			Hessian=-1*(self.__X.dot(temp))
			self.__X = self.__X.transpose()
			#print("NORM: ", np.linalg.norm(Hessian))
			if np.linalg.norm(Hessian) < 0.001:
					return None
			self.__theta = self.__theta - ((gradient.dot((1/Hessian)))*self.__learningRate)
			
			
			
			
		return None

	
	#sigmoid fct g(O)=1/(1+e^-Ox)  O=theta
	# X Matrix -> returns Vector (one scalar for each sample)
	def gFunc(self, X, theta):
		temp = np.zeros(len(X))
		temp2 = np.array(-X.dot(theta))
		np.exp(temp2,temp2)
		return  1/(temp2+1)
	
	
	

	
	# plug optimized theta in gFunc() -> check if smaller or higher than 0.5 (1-0.5)
	def predict(self, X):
		#print("#################### NEW PREDICTION ######################")
		Xneu = np.insert(X, 2, 1, axis=1)
		Ausgabe  =self.gFunc(Xneu, self.__theta)
		print("Ausgabe1: ", Ausgabe, "\n")
		
		#np.putmask(Ausgabe, Ausgabe>0.5, self.__class[1])
		#np.putmask(Ausgabe, Ausgabe<=0.5, self.__class[1])
		for i in Ausgabe:
			if Ausgabe[i] >= 0.5:
				Ausgabe[i] = self.__class[0]
			else:
				Ausgabe[i] = self.__class[1]
		
		

		return Ausgabe



















