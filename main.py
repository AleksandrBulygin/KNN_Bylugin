'''
В данном репозитории хранится учебное задание по предмету
"Методы машинного обучения". Практика №1 метод к ближайших соседей
Вариант 9. Выполнил студент группы P41192 Булыгин Александр.
'''

# импорт ниструмента для предобработки данных (нормировки)
from sklearn import preprocessing

# импорт инструментов для построения графиков
import matplotlib.pyplot as plt

# импорт библиотеки для создания структур данных
import pandas as pd

'''
импорт библиотеки для создания массивов и встроенных высокооптимизированных
функций
'''
import numpy as np

# библиотека с математическими функциями
import math

# импорт инструмента оценки качества предсказания регрессионной модели
from sklearn.metrics import r2_score

'''
импорт интсрумента, реализующего метод KNN, для сравнения реализованного
алгоритма
'''
from sklearn.neighbors import KNeighborsRegressor


# импорт функции реализации метода KNN
from knn_realization import KNN

# импорт функции реализации метода парзеновского окна для задачи регрессии
from regressionKNN import regressionKNN

# импорт функции реализации алгоритма STOLP для регрессии
from STOLP_for_regression import STOLP


def main():
    '''
    создание DataFrame из датасета, используемого для прогнозирования 
    гидродинамических характеристик парусных яхт по размерам и скорости
    Атрибуты:
        Long_pos - Продольное положение центра плавучести, безразмерная хар-ка
        Pris_coef - Призматический коэффициент, безразмерная
        Ld_rat - Отношение длины к водоизмещению, безразмерная хар-ка
        Bd-rat - Отношение балки к осадке, безразмерная хар-ка
        Lb-rat - Отношение длины луча, безразмерная хар-ка
        Froude_num - Число фруда, безразмерная хар-ка
    
    Измеряемая переменная:
        resist - Остаточное сопротивление на единицу веса водоизмещения,
        безразмерая хар-ка
    '''
    df = pd.read_csv('yacht_hydrodynamics.data'
                     , sep = ' '
                     , names=['Long_pos','Pris_coef', 'Ld_rat', 'Bd-rat',
                              'Lb-rat', 'Froude_num', 'resist'])
    colums = ['Long_pos','Pris_coef', 'Ld_rat', 'Bd-rat', 'Lb-rat',
              'Froude_num']
    
    # Вывод первых 10 элементов в консоль 
    print("First 10 elements of dataset:")
    print(df.head(10))
    
    '''
    Нормализация датасета по всем столбцам к диапазону от 0 до 1
    с помощью иструмента масштабирования MaxMinScaler
    '''
    scaler = preprocessing.MinMaxScaler()
    names = df.columns
    d = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(d, columns=names)
    
    '''
    Вычисление минимального и максимального значения целевой метки для
    обратного преобразования
    '''
    max_resist = float(max(df['resist']))
    min_resist = float(min(df['resist']))
    
    '''
    так как целевая метка записана в последнем столбце можно определить
    номер этого столбца для создания универсального метода КНН
    '''
    n_label = len(df.columns) - 1
    
    '''
    перераспределение эдементов датасета случайным образом для
    формирования репрезентативной выборки в тренировочный и тестовый датасеты
    '''
    scaled_df = scaled_df.sample(frac = 1)
    
    # В тренировочной выборке 250 элементов, а в тестовой 58
    trainSet = scaled_df[:250]
    testSet = scaled_df[250:]
    
    # Вывод в консоль графиков тренировочных и тестовых данных
    print("\nPlots of train and test sets after random distribution:")
    for i in range(n_label):
        firstPlot = trainSet.plot(x = colums[i], y = testSet.columns[n_label], 
                                  kind = 'scatter', color = 'r', 
                                  label = 'trainSet')
        testSet.plot(x = colums[i], y = testSet.columns[n_label], 
                     kind = 'scatter', ax = firstPlot, color = 'b', 
                     label = 'testSet', figsize = (10,10))
    plt.show()
    
    # Переменная для задания количества ближайших соседей
    k = 3
    
    '''
    Вычисление к ближайших сосдей и расстояний до них
    функция определена в файле knn_realization.py
    '''
    sortDist, sortLabels = KNN(k, trainSet, testSet)
    
    '''
    Вывод в консоль информации о ближайших соседей первых объектов из тесто-
    вого датасета
    '''
    print("Information of neighbors of first couple of test set objects:")
    for i in range(len(testSet)):
        for j in range(k):
            print(j+1, " Resistance: ", sortLabels[i][j], 
                  "\tDistance: ", round(sortDist[i][j], 4) )
        print("\ntrue Resistance: ", testSet.iat[i, n_label], "\n")
        print("|-----------------------------------------|")
        if (i == 10):
            break
    
    '''
    Предсказание неизвестного параметра объектов тестового датасета
    с помощью метода парзеновского окна для задачи регрессии
    функция определена в файле regressionKNN.py
    '''
    a_pred = regressionKNN(k, testSet, sortDist, sortLabels)
    
    '''
    Вывод в консоль предсказанных значений первых объектов из тестового набора
    данных
    '''
    print("\nPredictions of first couple of test set objects: ")
    for i in range(len(testSet)):
        true = (testSet.iat[i, n_label] + min_resist)*(max_resist - min_resist)
        pred = (a_pred[i] + min_resist)*(max_resist - min_resist)
        print("true Resistance\t\t", round(true, 4))
        print("Predict Resistance\t", round(pred,4))
        print("|-----------------------------------------|")
        if (i == 10):
            break
    
    # Вывод в консоль метрики R^2 для полученных меток
    print('\nr2_score with k = {}: '.format(k), r2_score(
        testSet[testSet.columns[n_label]].to_numpy(), a_pred))
    
    '''
    Инициализация максимального количесотва соседей и минимальной ошибки
    для реализации алгоритма Leave one out
    '''
    k_max = 30
    min_error = math.inf
    error = []
    
    # Инициализация цикла для алгоритма Leave one out
    for i in range(k_max):
        
        '''
        формирование переменных количества учитываемых соседей и суммарной
        ошибки
        '''
        k_LOO = i+1
        error_rate = 0
        
        # Инициализация внутреннего цикла прохода по тренировочному датасету
        for j, row in trainSet.iterrows():
            
            # Исключение рассматриваемого объекта из тренировочного датасета
            LOO_set = trainSet.drop(labels = [j])
            
            # Вычисление предсказания очередного объекта
            sortDistLOO, sortLabelsLOO = KNN(k_LOO, LOO_set, row,
                                             one_elem_pred = True)
            a_LOO = regressionKNN(k_LOO, row, sortDistLOO, sortLabelsLOO,
                                  one_elem_pred = True)
            
            # прибавление квадрата ошибки 
            error_rate += pow(a_LOO - row[n_label], 2)
        
        error.append(math.sqrt(error_rate))
        
        '''
        Если сумма ошибок меньше минимальной, то текущее количество ближайших
        соседей оптимально
        '''
        if (min_error > error_rate):
            k_opt = k_LOO
            min_error = error_rate
    
    # Построение графика выбора аптимального значения k
    plt.plot(range(1, 31), error)
    plt.scatter(k_opt, math.sqrt(min_error))
    plt.xlabel('k')
    plt.ylabel('error')
    plt.title("LOO plot")
    plt.show()
    
    # Вывод в консоль оптимального значения k
    print('\noptimal k after LOO: ', k_opt)
    
    # Расчёт парамета R^2 с текущим тренировочным датасетом
    sortDist, sortLabels = KNN(k_opt, trainSet, testSet)
    a_pred = regressionKNN(k_opt, testSet, sortDist, sortLabels)
    r2 = r2_score(testSet[testSet.columns[n_label]].to_numpy(), a_pred)
    print('\nr2_score before STOLP: ', r2, '\n')
    
    '''
    Инициализация величины минимального параметра R^2 и параметра delta для
    алгоритма STOLP
    '''
    r2 = 0.9
    delta = 0.35
    
    '''
    Уменьшение тренировочного набора данных с помощью алгоритма STOLP
    функция определена в файле STOLP_for_regression.py
    Для получения наиболее репрезентативного набора в качестве тренировочного
    набора данных взят исходный набор данных
    '''
    tsFit, ts_wo_out = STOLP(scaled_df, k_opt, delta, r2)
    
    # Исключение выбросов из начального датасета
    Set_without_outliers = scaled_df
    for i, row in scaled_df.iterrows():
        LOO_set = scaled_df.drop(labels = [i])
        sortDistLOO, sortResistLOO = KNN(k_opt, LOO_set, row, 
                                         one_elem_pred = True)
        a_elem = regressionKNN(k_opt, row, sortDistLOO, sortResistLOO,
                               one_elem_pred = True)
        variance = abs(a_elem - row[testSet.columns[n_label]])
        if (variance > 0.35):
            Set_without_outliers = Set_without_outliers.drop(labels = [i])
            
    # Исключение из полученного датасета объектов из тренировочного датасета
    # и формирование из результата тренировочного датасета
    testSet = Set_without_outliers[~Set_without_outliers.apply(
        tuple,1).isin(tsFit.apply(tuple,1))]
    
    # Вычисление предсказаний целевых меток тестового датасета
    sortDist_fit, sortLabels_fit = KNN(k_opt, tsFit, testSet)
    a_pred_test = regressionKNN(k_opt, testSet, sortDist_fit, sortLabels_fit)
    
    # Вычисление параметра R^2 и его вывод в консоль
    print('\nr2_score after STOLP: ', r2_score(
        testSet[testSet.columns[n_label]].to_numpy(), a_pred_test))
    
    '''
    Вывод в консоль предсказанных значений первых объектов из тестового набора
    данных
    '''
    print('\nPredictions of test set after STOLP:')
    for i in range(len(testSet)):
        true = (testSet.iat[i, n_label] + min_resist)*(max_resist - min_resist)
        pred = (a_pred_test[i] + min_resist)*(max_resist - min_resist)
        print("true Resistance\t\t", round(true, 4))
        print("Predict Resistance\t", round(pred,4))
        print("|-----------------------------------------|")
        if (i == 10):
            break
    
    # Вывод в консоль графиков тренировочных и тестовых данных
    print('\nplots of train set and rest set after STOLP:')
    for i in range(n_label):
        firstPlot = tsFit.plot(x = colums[i], y = testSet.columns[n_label], kind = 'scatter', color = 'r', label = 'trainSet')
        testSet.plot(x = colums[i], y = testSet.columns[n_label], kind = 'scatter', ax = firstPlot, color = 'b', label = 'testSet', figsize = (10,10), marker = '_', grid = True)
    plt.show()
    
    # Вычисление целевых меток тестового набора данных с помощью инструментов
    # SciKit Learn
    neigh = KNeighborsRegressor(n_neighbors = k_opt)
    
    # Тренировочная выборка
    X = tsFit[['Long_pos','Pris_coef', 'Ld_rat', 'Bd-rat', 'Lb-rat',
               'Froude_num']].to_numpy()
    y = tsFit['resist'].to_numpy()
    neigh.fit(X, y)
    
    # тестовая выборка
    x_test = testSet[['Long_pos','Pris_coef', 'Ld_rat', 'Bd-rat', 'Lb-rat', 
                      'Froude_num']].to_numpy()
    pred_sklearn = neigh.predict(x_test)

    y_test = testSet['resist'].to_numpy()
    
    # Вывод в консоль параметра R^2 для метода KNN из SciKit Learn и
    # реализованного
    print("\nR^2 SciKit Learn: ", r2_score(y_test, pred_sklearn))
    print("R^2 of made function: ", r2_score(y_test, a_pred_test))
          
          
    

if __name__ == "__main__":
	main()