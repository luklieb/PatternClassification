# -*- coding: latin-1 -*-
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


import math
import numpy


class kNearestNeighbor(object):
	def __init__(self, k):
		self.__mMax = 1e8
		self.__k = k


	def fit(self, X, y):
		# store references to the labeled training data
		self.__X = X
		self.__y = y
		self.__m = len(X)
		self.__mMax = 1e8
		#self.printDebug()


	def predict(self, X):
		# |x-y|^2 = (x - y)^T (x - y) = - 2 * x^T y + x^T x + y^T y 
		# runtime efficient as for-loops are avoided, but runs out of memory 
		# pretty fast for large training and test sets;
		# process only m test samples at a time

		m = int(self.__mMax / self.__m)
		numRuns = math.ceil(len(X) / m)
		
		# number of different classes
		numC = int(max(self.__y))

		
		Z = numpy.zeros(0)
		for i in range(numRuns):
		    Xs = X[i * m: (i + 1) * m]
		    length = len(Xs)
		    d1 = numpy.square(Xs).sum(axis = 1)
		    d2 = numpy.square(self.__X).sum(axis = 1)
		    D = numpy.dot(Xs, self.__X.T)
		    D *= -2
		    D += d1.reshape(-1, 1)
		    D += d2
		    
		    temp = numpy.argsort(D, axis=1)
		    #indexes ind of k closest points for each point
		    ind = numpy.argsort(D, axis = 1)[:, 0:self.__k]
		    
		    #array Z of fixed size to store output to w/o memory-costly append/insert functions
		    Z = numpy.zeros(len(ind)) 
		    #saves the classes of the corresponding k closest points (saved in 2D array ind) to 2D array T
		    T = (numpy.array(self.__y[ind])).astype(int) 
		    
		    j = 0
		    #index = (self.__m) - 1
		    
		    for sub in T:
		    	#sums up the occurence of certain classes in T
		    	binc = numpy.bincount(sub)
		    	#returns the index of the classes with the highest occurence
		    	a = (numpy.argmax(binc)).astype(int)
		    	Z[j] = a
		    	j +=1
		    j = 0


		return Z   
	


'''
	def printDebug(self):
		print("__k: ", self.__k, " __mMax: ", self.__mMax, " __X: ", self.__X, " __y: ", self.__y, " __m: ", self.__m)
		

'''

		
#print(" numRuns: ", numRuns, "\n"," i: ", i, "\n"," lenX: ", len(X), "\n"," X: ", X, "\n", " Xs: ", Xs, "\n"," d1: ", d1, "\n", " d2: ", d2, "\n", " D: ", D, "\n", " temp: ", temp, "\n", " ind: ", ind, "\n", " T: ", T, "\n", " Z: ", Z, "\n")



	
	
