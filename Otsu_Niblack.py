import copy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from statistics import mean
import math
from skimage import io
import sys


def calcul_threshold_global(image):
    """
    Functie ce returneaza imaginea binarizata 
        in functie de un threshold global calculat
        cu media dintre valoarea minima si maxima globala.

    Parametrii
    ----------
    image : imagine in tonuri de gri
    """

    # Pentru a modifica doar copia imaginii
    img = copy.copy(image)
    rows, cols = img.shape
    # Gasire valoare minima si maxima in imagine
    max = 0
    min = 255
    for row in range(rows):
        for column in range(cols):
            if img[row][column] > max:
                max = img[row][column]
            if img[row][column] < min:
                min = img[row][column]

    # Creare ca si threshold global valoarea medie dintre toate 
    #   intensitatile prezente in imagine
    vals = [min, max]
    threshold = mean(vals)

    # for row in range(rows):
    #     for column in range(cols):
    #         if img[row][column] > threshold:
    #             img[row][column] = 255
    #         else:
    #             img[row][column] = 0
    pass
    return threshold


def deviatia_standard(image, size):
    """
    Functie ce divide o matrice (reprezentand o imagine in tonuri de gri) in 
        submatrici pentru care calculeaza valoarea medie si deviatia standard.

    Parametrii
    ----------
    image : imagine in tonuri de gri.
    size  : int
            Dimensiune submatrice pentru care se calculeaza deviatia standard.
    """

    img = copy.copy(image)
    rows, cols = img.shape
    std_devs = []
    medie = []
    numarator_deviatie = []

    # Se calculeaza mediile pentru fiecare submatrice din intreaga imagine
    for i in range(0, rows, size):
        for j in range(0, cols, size):
            submatrix = img[i:i+size, j:j+size]
            if submatrix.shape[0] == size and submatrix.shape[1] == size:
                medie.append(np.mean(submatrix))

    
    numitor_deviatie = (size*size)-1


    for i in range(0, rows, size):
        for j in range(0, cols, size):
            submatrix = img[i:i+size, j:j+size]
            if submatrix.shape[0] == size and submatrix.shape[1] == size:
                numarator_deviatie = 0
                # Iteram prin fiecare element din submatrice
                for k in range(size):
                    for l in range(size):
                        # Diferenta dintre fiecare element si valoare medie a submatricii
                        diff = submatrix[k][l] - medie[i // size * (cols // size) + j // size]
                        # Fiecare diferenta este ridicata la patrat si adunata la valoarea totala a numaratorului
                        numarator_deviatie += diff ** 2
                # Calcul deviatii standard
                std_dev = math.sqrt(numarator_deviatie / numitor_deviatie)
                # Fiecare valoare calculata este inserata in lista
                std_devs.append(std_dev)
    
    pass
    return medie, std_devs


def threshold_niblack(image, size, k=0.2):
    """
    Functie ce calculeaza threshold local
        pe baza algoritmului Niblack.

    Parametrii
    ----------
    image : imagine in tonuri de gri.
    size  : int
            Dimensiune submatrice pentru care se calculeaza threshold-ul.
    k     : float, optional
            Valoarea parametrului k in calculul threshold-ului. (default=0.2)
    """

    img = copy.copy(image)
    rows, cols = img.shape
    mean_std_devs, std_devs = deviatia_standard(img, size)
    threshold = []

    # Calcul threshold pe baza formulei algoritmului Niblack
    for mean_std_dev, std in zip(mean_std_devs, std_devs):
        val_threshold = mean_std_dev + k * std
        threshold.append(val_threshold)

    for i in range(rows):
        for j in range(cols):
            # Imaginea este impartita in submatrici
            submatrix_index = i // size * (cols // size) + j // size
            # Se verifica daca indexul submatricii este mai mic decat numarul de threshold-uri calculate
            if submatrix_index < len(threshold):  
                if img[i][j] > threshold[submatrix_index]:
                    img[i][j] = 255
                else:
                    img[i][j] = 0

    pass
    return img


def calculare_histograma(image):
    """
    Functie ce calculeaza histograma unei imagini
        in tonuri de gri.

    Parametrii
    ----------
    image : imagine in tonuri de gri.
    """

    hist = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[int(image[i, j])] += 1
    
    pass
    return hist


def threshold_otsu_global(hist):
    """
    Functie ce calculeaza threshold global
        pe baza algoritmului Otsu.

    Parametrii
    ----------
    hist : 1D array
           Histograma unei imagini in tonuri de gri.
    """

    total = sum(hist)

    # Calcul suma ponderata al pixelilor pe tonuri de gri.
    # Ponderea unei intensitati 'hist[i]' este cu atat mai mare
    #   cu cat numarul pixelilor 'i' de acea intensitate este mai mare.
    sum1 = 0
    for i in range(256):
        sum1 += i * hist[i]
    
    sumB = 0
    # wF -> Nr. Pixeli PRIM PLAN / Nr. Total Pixeli
    wF = 0
    # wB -> Nr. Pixeli FUNDAL / Nr. Total Pixeli
    wB = 0
    sigmaMAX = 0
    threshold = 0

    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break

        sumB += i * hist[i]
        uB = sumB / wB
        uF = (sum1 - sumB) / wF

        sigma = wB * wF * (uB - uF) ** 2
        
        # Daca se gaseste o valoare maxima noua penrtu sigma, se actualizeaza threshold-ul
        if sigma > sigmaMAX:
            sigmaMAX = sigma
            threshold = i

    pass
    return threshold


def procesare_submatrice(image, i, j, size):
    """
    Functie folosita pentru aplicarea algoritmului Otsu global pe 
        submatrici ale unei imagini, astfel efectul aplicandu-se local.

    Parametrii
    ----------
    hist  : 1D array
            Histograma unei imagini in tonuri de gri.
    image : imagine in tonuri de gri.
            Histograma unei imagini in tonuri de gri.
    i     : int
            Reprezinta linia din imagine de unde incepe submatricea.
    j     : int
            Reprezinta coloana din imagine de unde incepe submatricea.
    size  : int
            Dimensiune submatrice pentru care se calculeaza threshold-ul.
    """

    # Se extrage submatricea din imaginea originala
    submatrix = image[i:i+size, j:j+size]

    # Se calculeaza histograma
    hist = calculare_histograma(submatrix)

    # Se aplica algoritmul Otsu
    threshold_value = threshold_otsu_global(hist)

    pass
    # Aplicare threshold asupra submatricii
    return aplicare_threshold(submatrix, threshold_value)


def threshold_otsu_local(image, size):
    """
    Functie ce calculeaza threshold local
        pe baza algoritmului Otsu.

    Parametrii
    ----------
    image : imagine in tonuri de gri.
    size  : int
            Dimensiune submatrice pentru care se calculeaza threshold-ul.
    """

    img = copy.copy(image)
    rows, cols = img.shape
    threshold = np.zeros_like(img)

    # Parcurgere imagine in submatrici de 9x9
    for i in range(0, rows, size):
        for j in range(0, cols, size):
            i_end = min(i + size, rows)
            j_end = min(j + size, cols)

            # Procesare fiecare submatrice
            thresholded_block = procesare_submatrice(img, i, j, size)
            # Se insereaza fiecare submatrice procesata in pozitia ei din imaginea procesata
            threshold[i:i_end, j:j_end] = thresholded_block[0:i_end-i, 0:j_end-j]
    
    pass
    return threshold


def aplicare_threshold(image, threshold):
    """
    Functie ce binarizeaza o imagine pe
        baza unui threshold primit ca parametru.
    Daca un pixel are o valoare mai mare decat
        threshold-ul, atunci pixelul devine alb,
        iar daca are o valoare mai mica, devine negru.

    Parametrii
    ----------
    image : imagine in tonuri de gri.
    threshold : valoare calculata in prealabil
    """

    img = copy.copy(image)
    rows, cols = img.shape
    thresholded_image = np.zeros_like(img)
    for i in range(rows):
        for j in range(cols):
            if img[i][j] > threshold:
                thresholded_image[i][j] = 255
            else:
                thresholded_image[i][j] = 0
    
    pass
    return thresholded_image

# Citire imagine originala
if len(sys.argv) > 1:
    image_file = sys.argv[1]
    img = io.imread(image_file)


# Imagine pe care s-a aplicat threshold global
try:
    threshold_global = calcul_threshold_global(img)
    print("Threshold global aplicat              OK")
except Exception as e:
    print("Threshold global aplicat              ERR")
    print("Error details:", e)
img_threshold_global = aplicare_threshold(img, threshold_global)


# Algoritm Niblack
try:
    img_threshold_niblack = threshold_niblack(img, 25, 0.3)
    print("Algoritm Niblack aplicat              OK")
except Exception as e:
    print("Algoritm Niblack aplicat              ERR")
    print("Error details:", e)


# Algoritm Otsu Global
hist = calculare_histograma(img)
try:
    threshold_otsuV1 = threshold_otsu_global(hist)
    print("Algoritm Otsu Global aplicat          OK")
except Exception as e:
    print("Algoritm Otsu Global aplicat          ERR")
    print("Error details:", e)
img_threshold_otsu_global = aplicare_threshold(img, threshold_otsuV1)


# Algoritm Otsu Local
try:
    img_threshold_otsu_local = threshold_otsu_local(img, 9)
    print("Algoritm Otsu Local aplicat           OK")
except Exception as e:
    print("Algoritm Otsu Local aplicat           ERR")
    print("Error details:", e)


# Zona de afisare a imaginilor procesate
plt.figure(figsize=(20, 10))
print("Se afiseaza rezultatele...")

# Imagine originala
plt.subplot(4, 1, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagine Originala')
plt.axis('off')

# Imagine cu threshold global
plt.subplot(4, 1, 2)
plt.title("Threshold Global")
plt.imshow(img_threshold_global, cmap=plt.cm.gray)
plt.axis('off')

# Imagine cu algoritmul Niblack
plt.subplot(4, 1, 3)
plt.imshow(img_threshold_niblack, cmap='gray')
plt.title('Algoritm Niblack')
plt.axis('off')

# Imagine cu algoritmul Otsu
plt.subplot(4, 2, 7)
plt.imshow(img_threshold_otsu_global, cmap='gray')
plt.title('Algoritm Otsu Global')
plt.axis('off')

# Imagine cu algoritmul Otsu Local
plt.subplot(4, 2, 8)
plt.imshow(img_threshold_otsu_local, cmap='gray')
plt.title('Algoritm Otsu Local')
plt.axis('off')

plt.show()