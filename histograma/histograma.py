#!/usr/bin/env python3
# -- coding: utf-8 --

import sys
import argparse
from os import walk
import numpy as np
import cv2 as cv

parser = argparse.ArgumentParser(
    prog='histograma',
    description='Compara imagens utilizando 4 métodos diferentes de comparação de histograma. As imagens devem estar no mesmo diretório do script e seus nomes devem ser formatados exatamente como os das imagens passadas com a especificação. Cada classe (ou personagem) deve possuir duas imagens nomeadas \'nome1.png\' e \'nome2.png\' pois o programa usará slices das string para checar se a classificação foi correta.'
)

parser.add_argument('-p', '--plot', action='store_true', dest='plot', help='faz o programa mostrar as imagens e os histogramas antes da classificação.')
parser.add_argument('-d', '--debug', action='store_true', dest='debug', help='imprime mensagens para debug.')
parser.add_argument('-n', '--normalize', action='store_true', dest='normalize',  help='faz o programa normalizar os histogramas antes da classificação.')
parser.add_argument('-e', '--equalize', action='store_true', dest='equalize',  help='faz o programa equalizar os histogramas das imagens em níveis de cinza antes da classificação. As imagens coloridas são convertidas para HSV e tem seu canal V equalizado, sendo convertidas para RGB depois.')

args = parser.parse_args()

if args.plot:
    from matplotlib import pyplot as plt

# Pega nomes dos arquivos png do diretório corrente
image_filenames = []
for (dirpath, dirnames, filenames) in walk("."):
    for filename in filenames:
        if filename.endswith(".png"):
            image_filenames.append(filename)

# Classificação
comparison_methods = [
    {'number': cv.HISTCMP_CORREL, 'name': 'HISTCMP_CORREL (Correlation)', 'sorting_order': {'reverse': True}},
    {'number': cv.HISTCMP_CHISQR, 'name': 'HISTCMP_CHISQR (Chi-Square)', 'sorting_order': {'reverse': False}},
    {'number': cv.HISTCMP_INTERSECT, 'name': 'HISTCMP_INTERSECT (Intersection)', 'sorting_order': {'reverse': True}},
    {'number': cv.HISTCMP_BHATTACHARYYA, 'name': 'HISTCMP_BHATTACHARYYA (Bhattacharyya distance)', 'sorting_order': {'reverse': False}}
]

def readImage(filename, readMethod):
    global args
    
    img = cv.imread(filename, readMethod)

    if args.equalize:
        if readMethod != cv.IMREAD_COLOR:
            img = cv.equalizeHist(img)
        else:
            HSV_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)        
            # Equalizando o canal 'V' da imagem
            HSV_img[:, :, 2] = cv.equalizeHist(HSV_img[:, :, 2])
            img = cv.cvtColor(HSV_img, cv.COLOR_HSV2BGR)
    
    return img



# Abrindo as imagens em cor
print("Abrindo imagens com \'cv.IMREAD_COLOR\':\n", file=sys.stderr)
images = {}
for filename in image_filenames:
    images[filename] = {'img': readImage(filename, cv.IMREAD_COLOR)}
    
    # Cria histogramas de cada canal da imagem
    BGR_planes = cv.split(images[filename]['img'])
    histSize = [256]
    histRange = (0, 256)
    images[filename]['b_hist'] = cv.calcHist(BGR_planes, [0], None, histSize, histRange, accumulate=False)
    images[filename]['g_hist'] = cv.calcHist(BGR_planes, [1], None, histSize, histRange, accumulate=False)
    images[filename]['r_hist'] = cv.calcHist(BGR_planes, [2], None, histSize, histRange, accumulate=False)

    if args.normalize:
        normalizing_kwargs = { 'alpha': 0, 'beta': 255, 'norm_type': cv.NORM_MINMAX }
        cv.normalize(images[filename]['b_hist'], images[filename]['b_hist'], **normalizing_kwargs)
        cv.normalize(images[filename]['g_hist'], images[filename]['g_hist'], **normalizing_kwargs)
        cv.normalize(images[filename]['r_hist'], images[filename]['r_hist'], **normalizing_kwargs)

    if args.plot:
        # Plota a imagem
        # Converte a ordem da cor da imagem para mostrar pela matplotlib 
        RGB_src = cv.cvtColor(images[filename]['img'], cv.COLOR_BGR2RGB)
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(RGB_src, cmap='brg_r') 
        axarr[0].title.set_text(filename)  

        # Plota o histograma
        histograms = ('b_hist','g_hist','r_hist')
        for i, hist in enumerate(histograms):
            axarr[1].plot(images[filename][hist], color=hist[0:-5])
        axarr[1].title.set_text('Histograma de %s' % filename)  

        plt.show()

for method in comparison_methods:
    print(method['name'], file=sys.stderr)
    hits = 0
    misses = 0
    
    for filename_i in images:
        hist_comparisons = []
        if args.debug:
            print("Comparing %s to the other images using %s..." % (filename_i, method['name']), file=sys.stderr)

        for filename_j in images:
            b_hist_comparison = cv.compareHist(
                images[filename_i]['b_hist'],
                images[filename_j]['b_hist'],
                method['number']
            )
            g_hist_comparison = cv.compareHist(
                images[filename_i]['g_hist'],
                images[filename_j]['g_hist'],
                method['number']
            )
            r_hist_comparison = cv.compareHist(
                images[filename_i]['r_hist'],
                images[filename_j]['r_hist'],
                method['number']
            )

            hist_comparison_score = sum([b_hist_comparison, g_hist_comparison, r_hist_comparison])/3
            hist_comparisons.append({'filename': filename_j, 'score': hist_comparison_score})
        
        # Eu comparei a imagem com ela mesma para ter uma espécie de teste de sanidade pro meu programa
        # A posição 0 do vetor abaixo conterá a própria imagem, então, a classificação que esse algoritmo faz
        # será a próxima comparação mais próxima, ou "best_match", que no caso estará na posição 1 do vetor 
        best_match = sorted(hist_comparisons, key=lambda x : x['score'], **method['sorting_order'])[1] 

        if filename_i[0:-5] == best_match['filename'][0:-5]:
            hits = hits + 1
            print("HIT: ", filename_i, " x ", best_match['filename'])
        else:
            misses = misses + 1

    print('ACERTOS: ', hits, file=sys.stderr)
    print('ERROS: ', misses, file=sys.stderr)
    print(('TAXA DE ACERTOS: %.2f\n' % float(hits/(len(images)-1))), file=sys.stderr)




# Abrindo as imagens em grayscale
print("\nAbrindo imagens com \'cv.IMREAD_GRAYSCALE\':\n", file=sys.stderr)
images = {}
for filename in image_filenames:
    images[filename] = {'img': readImage(filename, cv.IMREAD_GRAYSCALE)}
    
    # Cria o histograma da imagem
    histSize = [256]
    histRange = (0, 256)
    images[filename]['hist'] = cv.calcHist(images[filename]['img'], [0], None, histSize, histRange, accumulate=False)

    if args.normalize:
        normalizing_kwargs = { 'alpha': 0, 'beta': 255, 'norm_type': cv.NORM_MINMAX }
        cv.normalize(images[filename]['hist'], images[filename]['hist'], **normalizing_kwargs)
 
    if args.plot:
        # Plota a imagem
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(images[filename]['img'], cmap='gray') 
        axarr[0].title.set_text(filename)  

        # Plota o histograma
        axarr[1].plot(images[filename]['hist'], color='gray')
        axarr[1].title.set_text('Histograma de %s' % filename)  

        plt.show()

for method in comparison_methods:
    print(method['name'], file=sys.stderr)
    hits = 0
    misses = 0
    
    for filename_i in images:
        hist_comparisons = []
        if args.debug:
            print("Comparing %s to the other images using %s..." % (filename_i, method['name']), file=sys.stderr)

        for filename_j in images:
            hist_comparison = cv.compareHist(
                images[filename_i]['hist'],
                images[filename_j]['hist'],
                method['number']
            )

            hist_comparisons.append({'filename': filename_j, 'score': hist_comparison})
        
        best_match = sorted(hist_comparisons, key=lambda x : x['score'], **method['sorting_order'])[1] 

        if filename_i[0:-5] == best_match['filename'][0:-5]:
            print("HIT: ", filename_i, " x ", best_match['filename'])
            hits = hits + 1
        else:
            misses = misses + 1

    print('ACERTOS: ', hits, file=sys.stderr)
    print('ERROS: ', misses, file=sys.stderr)
    print(('TAXA DE ACERTOS: %.2f\n' % float(hits/(len(images)-1))), file=sys.stderr)
