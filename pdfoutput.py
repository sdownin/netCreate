# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 01:25:22 2015

@author: sdowning
"""

from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt

x1 = np.arange(10)
y1 = x1**2

x2 = np.arange(20)
y2 = x2**2

pp = PdfPages('TEST.pdf')


def function_plot(X,Y):
    plt.figure()
    plt.clf()

    plt.plot(X,Y)
    plt.title('y vs x')
    plt.xlabel('x axis', fontsize = 13)
    plt.ylabel('y axis', fontsize = 13)
    pp.savefig()

function_plot(x1,y1)
function_plot(x2,y2)

pp.close()