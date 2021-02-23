#!/usr/bin/python3

# SORAY CENGİZ ELM 463 PROJECT
# 03.01.2021  17:40
# Kodu çalıştırabilmek ilgili modüllerin yüklü olduğundan emin olunuz

import plotly.express as px
import plotly.graph_objects as go
from skimage import data, filters, measure, morphology
import sys

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import math


def connected_component_label(i_image):
    # Birbiri ile bağlantılı objeleri bulmak için kullanılan fonksiyondur.
    tmp_image = i_image
    number_of_labels, all_labels = cv.connectedComponents(tmp_image)

    label_hue = np.uint8(179*all_labels/np.max(all_labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
    labeled_img[label_hue == 0] = 0
    
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2RGB)
    
    return all_labels,labeled_img

def image_out_w(title, i_image):
    
    cv.imshow(str(title), i_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
        

def show_hist(title, i_image):
    # istenilen görüntünün histogramını gösteren ve ona bir başlık atayan fonksiyon
    plt.figure()
    plt.title(str(title))
    plt.hist(i_image.ravel(),256,[0,256])
    plt.show()

def circle_coordinates(marked_circles_parameters):
    # Hough Transform sonucu elde edilen parametrelerin işlenip derli toplu gösterilmesi
    print(' *** ALL MARKED CIRCLE COORDINATES *** ')
    
    size = marked_circles_parameters.shape[1]
    print('Number of marked circle : ', size)
    
    """
        Bulunan circle'ın merkezinin konumu [c1, c2] şeklinde ifade edilmiştir.
        Bulunan circle'ın çapı ise circle_diameter olarak ifade edilmiştir. 
    
    """
    
    for item in range(size):
        ccols = marked_circles_parameters[0][item][0]
        crows = marked_circles_parameters[0][item][1]
        circle_diameter = marked_circles_parameters[0][item][2]
        
        print(item + 1, '.circle -> ',
             'location : ' ,'[', ccols, crows, ']',' && ', 'diameter = ', circle_diameter)


def show_marked_circles(i_image, marked_circles_parameters):
    # Hough transform sonucunda bulunan muhtemel motorların bulunduğu yuvarlaklar
    # ... topluluğunun gösterimi.
    # ... Hough circle' sonucu elde edilen parametreler burada kullanılıp girdi görüntüsünde
    # ... ilgili yere yuvarlak çizmesini sağlayan bir fonksiyon yazıldı.
    # ... çizilen yuvarlaklar sadece görsel olarak görülmesi açısından beyaz seçildi.
    tmp_image = i_image
    counter = 1
    size = marked_circles_parameters.shape[1]
    
    for item in range(size):
        ccols = marked_circles_parameters[0][item][0]
        crows = marked_circles_parameters[0][item][1]
        circle_diameter = marked_circles_parameters[0][item][2]
        
        cv.circle(tmp_image, (ccols, crows), circle_diameter, (255,255,255), 5)
        cv.putText(tmp_image, str(counter) + '.circle', ((ccols - 50) , (crows + 20)), cv.FONT_HERSHEY_SIMPLEX,
                  1.1 , (255, 0, 0), 2)
        counter += 1

    image_out_ww('Founded circles with Hough Transform : ', tmp_image)
    

def output_of_thresholded_marked_image(i_image, marked_circles_parameters):
    # Hough circle sonucunda bulunan her bir circle'ın girdi görüntüsündeki konumuna 
    # ... +40 satıra ve +40 sütuna ekleme yaparak her bir circle x ve y eksenleri 40 piksel genişletildi
    # ... genişletilmesinin sebebi tam yuvarlak dışında da motorun kalabilme ihtimalidir.
    # ... part_row ve part_col 2*diameter + 40 olarak belirlenmesinin sebebide budur olabildiğince ekstra 
    # ... alana da baklımalıdır çünkü hough transform tam olarak motor çeperini yuvarlak içine alamamış olabilir.
    # ... Raporda daha detaylı görsel odaklı bir anlatımda yapılmıştır.
    
    rows, cols      = i_image.shape
    temporary_image = np.zeros(shape = [rows,cols], dtype = np.uint8) 
    
    size = marked_circles_parameters.shape[1]
    
    for item in range(size):
        ccols = marked_circles_parameters[0][item][0]
        crows = marked_circles_parameters[0][item][1]
        circle_diameter = marked_circles_parameters[0][item][2]
        
        part_row = 2*circle_diameter + 40
        part_col = 2*circle_diameter + 40
    
        part_image = np.zeros(shape = [part_row, part_col], dtype = np.uint8) 
    
        for i in range(crows - circle_diameter - 20, crows + circle_diameter + 20):
            for j in range(ccols - circle_diameter - 20, ccols + circle_diameter + 20):
                if j > 1280 - circle_diameter or i > 800 - circle_diameter:
                    pass
                else:
                    temporary_image[i, j] = i_image[i, j] 
    
    
        for i_part, i in zip(range(0, part_row), range(crows - circle_diameter - 20, crows + circle_diameter + 20)):
            for j_part, j in zip(range(0, part_col) ,range(ccols- circle_diameter - 20, ccols + circle_diameter + 20)):
                if j > 1280 - circle_diameter or i > 800 - circle_diameter:
                    pass
                else:
                    part_image[i_part, j_part] = temporary_image[i, j]
    
    
        Threshold_value = np.amax(part_image)
        Threshold_value = Threshold_value * 20
        Threshold_value = Threshold_value / 100

        part_image[part_image < Threshold_value]  = 0 
        part_image[part_image >= Threshold_value] = 255 
    
        for i_part, i in zip(range(0, part_row), range(crows - circle_diameter - 20, crows + circle_diameter + 20)):
            for j_part, j in zip(range(0, part_col) ,range(ccols- circle_diameter - 20, ccols + circle_diameter + 20)):
                if j > 1280 - circle_diameter or i > 800 - circle_diameter:
                    pass
                else:
                    temporary_image[i, j] = part_image[i_part, j_part]
    
    return temporary_image

