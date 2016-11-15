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


import numpy
import numpy.matlib


class LinearLogisticRegression(object):

    def __init__(self, learningRate = 0.5, maxIterations = 100):
		self.__learningRate = learningRate
		self.__maxIterations = maxIterations
        return None


	#estimate from labeled training samples
    def fit(self, X, y):
		self.__X = X
		self.__y = y
		self.__m = len(X)

        return None
        
        
    #sigmoid fct g(O)=1/(1+e^-Ox)  O=theta
	# X vector
    def gFunc(self, X, theta):
		return  1 / (1 + math.exp(-(transpose(theta).dot(X))))      

	#compute O (theta) with newton (from training samples) -> plug O in gFunc() -> check all classes y with gFunc and return highest posterior prob.
    def predict(self, X):
        
		





		return Z



















