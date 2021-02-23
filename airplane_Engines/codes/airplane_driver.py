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

import airplane_module as am

# MAIN 

# Giriş görüntüsü algoritmaya çağırılmaktadır.
image_input    = cv.imread("input_1.pgm", 0)    
image_new_size = cv.resize(image_input, (1280,800), interpolation = cv.INTER_LINEAR)

# Adative CLAHE uygulanmaktadır.
adaptive_hist_obj = cv.createCLAHE(clipLimit=5.0, tileGridSize=(12,12))
histogram_out     = adaptive_hist_obj.apply(image_new_size)

#show_hist('histogram_of_adaptive_histogram', histogram_out)

# Görüntü yumuşatılmıştır.
image_blur    = cv.GaussianBlur(histogram_out, (5,5), 0)
image_to_loop = image_blur

# Canny uygulanarak kenarlar tespit edilmiştir.
image_edges   = cv.Canny(image_blur, 100, 140)

# Hough Transform uygulanarak çıkış gözlemlenmiştir.
marked_circles = \
    cv.HoughCircles(image_edges, cv.HOUGH_GRADIENT, 1 , 80, param1 = 80, param2 = 20, minRadius = 15, maxRadius = 80)
marked_circles_round = np.uint16(np.around(marked_circles))

# Thresholded görüntü elde edilmiştir.
output_of_thresholded_image         = am.output_of_thresholded_marked_image(image_to_loop, marked_circles_round)
output_inverse_of_thresholded_image = cv.bitwise_not(output_of_thresholded_image)

# Görüntüye sırası ile Erosion ardından Dilation uygulanmıştır.
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))

output_of_erosion_image  = cv.erode(output_inverse_of_thresholded_image, kernel, iterations = 1)
output_of_dilation_image = cv.dilate(output_of_erosion_image, kernel, iterations = 1)

before_connected_components_image = output_of_dilation_image

# Etiket grupları ve etiketlenmiş görüntünün sonucunun görebiliriz.
label_groups            = am.connected_component_label(before_connected_components_image)[0]
output_of_labeled_image = am.connected_component_label(before_connected_components_image)[1]

# Figure işlemleri.
contour = None
y = ()
x = ()

fig = px.imshow(image_new_size, binary_string=True)
fig.update_traces(hoverinfo='skip')

props        = measure.regionprops(label_groups, before_connected_components_image)
properties_1 = ['eccentricity']
properties_2 = ['area']
properties_3 = ['major_axis_length']
properties_4 = ['minor_axis_length']
all_properties = ['eccentricity', 'area', 'major_axis_length', 'minor_axis_length']

# Condition'lardan geçen etiketlerin özelliklerini kaydedebilmek için listeler oluşturuldu.
list_ecc   = []
list_area  = []
list_major = []
list_minor = []

# list_of_label[] hangi etiketlerde doğru şartlar sağlandı sorusunun cevabını tutmaktadır.
motor_counter = 0
list_of_label = []

for index in range(1, label_groups.max()):
    hoverinfo = ''
    label = props[index].label
    for prop_name_1 in properties_1:
        holder_ecc = getattr(props[index], prop_name_1)
        
        if holder_ecc < 0.70:
            list_ecc.append(holder_ecc)
        
            for prop_name_2 in properties_2:
                holder_area = getattr(props[index], prop_name_2)
        
                if holder_area > 600:
                    list_area.append(holder_area)
                
                    for prop_name_3, prop_name_4 in zip(properties_3,properties_4):
                        holder_major = getattr(props[index], prop_name_3)
                        holder_minor = getattr(props[index], prop_name_4)
                        list_major.append(holder_major)
                        list_minor.append(holder_minor)
                        
                        if holder_major - holder_minor < 20:
                            contour = measure.find_contours(label_groups == label, 0.5)[0]
                            y, x = contour.T
                            motor_counter += 1
                            hoverinfo = ''
                            list_of_label.append(index)
                            for prop_names in all_properties:
                                hoverinfo += f'<b>{prop_names}: {getattr(props[index], prop_names):.2f}</b><br>'
                        else:
                            pass
                else:
                    pass  
        else:
            pass
    
    fig.add_trace(go.Scatter(
        x=x, y=y, name=label,
        mode='lines', fill='toself', showlegend=False,
        hovertemplate=hoverinfo, hoveron='points+fills'))
        
fig.show()

