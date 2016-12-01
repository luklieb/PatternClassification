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
from numpy.linalg import norm
from numpy.linalg import inv
from scipy.optimize import minimize



class LinearRegression(object):
    
    def __init__(self, lossFunction = 'l2', lossFunctionParam = 0.001, classification = False):
        #Wenn huber, dann lossFunctionParam =  M der Huber Funktion
        self.__lossFunctionParam = lossFunctionParam
        self.__initialized = False
        #__params sind a,b, willkuerlicher Startwert
        self.__params = np.array([0.05, 0.05])
        self.__lossFunction = lossFunction
        self.__classification = classification
        #print("init \n")
        return None


    def fit(self, X, y):
        #X: Vektor 1xn (zeilenweise x-Koord der Samples)
        #y: Vektor 1xn (zeilenweise y-Koord der Samples) --> keine Unterscheidung verschiedener Klassen
        #fuer optimale a,b scipy.optimize benutzen
        #Nur Regression und nicht "linear regression"-Classification wird betrachtet
        if self.__classification:
            print("Klassifizierung nicht möglich \n")
            return None
        #nur Huber Regression Berechnung moeglich
        if self.__lossFunction == 'l2':
            print("l2 geht nicht \n")
            return None
        self.__entries = len(y)
        #Bei einem Punkt Regression nicht sinnvoll möglich
        if self.__entries == 1:
            return None
        #affines X (mit 1 jeden x-wert erweitert fuer multipl. mit params a+b)
        self.__X = np.concatenate([X,np.ones(self.__entries)], axis=0).reshape(2,self.__entries)
        self.__y = np.array(y)
        #print("Xorig: ", X, "\n")
        #print("X: ", self.__X, ", y: ", self.__y, ", params: ", self.__params ,"\n")

        optimum = minimize(lambda params: self.huber_objfunc(self.__X,self.__y,params,self.__lossFunctionParam), self.__params, method='CG', jac = lambda params: self.huber_objfunc_derivative(self.__X,self.__y,params,self.__lossFunctionParam)   )
        
        #print("huber_objfunc: ", self.huber_objfunc(self.__X, self.__y,self.__params,self.__lossFunctionParam), "\n")
        #print("huber_objfunc_deriv: ", self.huber_objfunc_derivative(self.__X, self.__y,self.__params,self.__lossFunctionParam), "\n")
        if optimum.success == False:
            print("Fehler in Minimize \n")
            #self.__initialized = True
            #return None
        self.__params = optimum.x
        self.__initialized = True


    def huber_objfunc(self, X, y, params, m):
        #print("huber_ob a: ", a, "\n")
        #m ist M (Huberparameter)
        #z fuer self.huber() --> ist vector der Aufrufvariablen (yi - axi - b) = (y-params*X)
        z = y - (params.dot(X))
        temp = self.huber(z, m)
        return np.sum(temp)


    def huber_objfunc_derivative(self, X, y, params, m):
        #m ist M (Huberparameter)
        #z fuer self.huber_derivative() --> ist vector der Aufrufvariablen (yi - axi - b) = (y-params*X)
        #print("X: ", X, ", y: ", y, ", params: ", params ,"\n")
        z = y-(params.dot(X))
        temp = self.huber_derivative(z,m)
        return X.dot(-temp)


    def huber(self, z, m):
        #print("huber z:", z, ", m: ", m, "\n")
        #m ist M (Huberparameter) = self.__lossFunctionParam
        #z ist vector der Aufrufvariablen (yi - axi - b) = (y-params*X)
        temp = np.empty(self.__entries)
        np.putmask(temp,  np.absolute(z) <= m, z*z)
        np.putmask(temp, np.absolute(z) > m, m*(2*np.absolute(z)-m))
        #print("huber: ", temp, "\n")
        return temp


    def huber_derivative(self, z, m):
        #2*z fuer z<=M
        #sign(z)*2*m fuer z>M
        temp =  np.empty(self.__entries)
        np.putmask(temp, np.absolute(z)<= m, 2*z)
        np.putmask(temp, np.absolute(z)>m, np.sign(z)*2*m)
        #print("huber deriv: ", temp, "\n")
        return temp


    def paint(self, qp, featurespace):
        if self.__initialized:
            x_min, y_min, x_max, y_max = featurespace.coordinateSystem.getLimits()
            y1 = self.__params[0] * x_min + self.__params[1]
            x1, y1 = featurespace.coordinateSystem.world2screen(x_min, y1)
            y2 = self.__params[0] * x_max + self.__params[1]
            x2, y2 = featurespace.coordinateSystem.world2screen(x_max, y2)
            qp.drawLine(x1, y1, x2, y2)


    def predict(self, X):
        #ignorieren
        return None


