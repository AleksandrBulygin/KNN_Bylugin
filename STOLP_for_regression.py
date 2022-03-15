# импорт библиотеки для создания структур данных
import pandas as pd

'''
импорт библиотеки для создания массивов и встроенных высокооптимизированных
функций
'''
import numpy as np

# импорт инструментов для построения графиков
import matplotlib.pyplot as plt

# импорт инструмента оценки качества предсказания регрессионной модели
from sklearn.metrics import r2_score

# библиотека с математическими функциями
import math

# импорт функции реализации метода KNN
from knn_realization import KNN

# импорт функции реализации метода парзеновского окна для задачи регрессии
from regressionKNN import regressionKNN

def add_prot(row_elem, i_elem, Q_set, label_col):
    '''
    Функция добавления к датафрейму элемента

    Parameters
    ----------
    row_elem : DataFrame
        Элемент, который необходимо прибавить.
    i_elem : int
        Индекс этого элемента.
    Q_set : DataFrame
        Начальный Dataframe.
    label_col : int
        Порядковый намер столбца целевых меток.

    Returns
    -------
    Q_set : DataFrame
        DataFrame с добавленным к нему элементом.

    '''
    
    # инициализация переменной для формирования словаря
    d = {}
    
    '''
    Инициализация цикла для заполнения словаря значениями элемента,
    который необходимо прибавить к датафрейму
    '''
    for i in range(label_col+1):
        
        '''
        заполение контейнера по ключу, соответстующему названию столбца
        DataFrame, значением соответствующего столбца прибавляемого элемента
        '''
        d[Q_set.columns[i]] = [row_elem[i]]
        
    # Преобразование полученного словаря в DataFrame
    elem_df = pd.DataFrame(d, index = [i_elem])
    
    # Объединение DataFrame с полученным элементом
    Q_set = pd.concat([elem_df, Q_set])
    return Q_set

def STOLP(trainSetFunc, k_func, delta, l0, Remove_negative = False):
    
    '''
    Функция реализует алгоритм STOLP для регрессии.
    
    Так как в регресии нет понятия класса принадлежности, то выбросы
    определялись с помощью предсказания значения неизвестного параметра
    по оставшимся объектам тренировочной выборки, которое сравнивалось с
    реальным значением неизвестного параметра, если оно оказывается больше
    заданного парамета delta, то элемент является выбрасом.
    
    В начальный набор Q заносисились объекты, метки которых после применения 
    к ним метода KNN с тренировочной выборкой, из которой исключен
    рассматриваемый объект, различались с реальной метком объекта в диапазоне
    от минимальной заданной разницы до delta.
    
    Parameters
    ----------
    trainSetFunc : DataFlrame
        Тренировочный набор данных.
    k_func : int
        Количество учитываемых соседей.
    delta : float
        Параметр, определяющий границу области выбросов.
    l0 : float 
        Параметр, определяющий минимальное значение R^2 на обучающей выборке.
    Remove_negative : Bool, optional
        Флаг исключения выбросов внутри цикла после добавления
        граничных объектов. The default is False.

    Returns
    -------
    Q : DataFlrame
        Оптимизированный набор тренировочных данных.
    tsSTOLP : DataFrame
        Начальный набор тренировочных данных без выбрасов.
    
    '''
    
    # Максимальное расстояние до k+1 соседа для начального набора Q
    max_dist = 0.7
    
    '''
    Максимальная разница между целевой меткой рассматриваемого объекта
    и ее предсказанием с помощью объектов в наборе Q
    '''
    max_var_in_Q = 0.05
    
    '''
    Минимальная разница между целевой меткой рассматриваемого объекта
    и ее предсказанием с помощью объектов начального тренировочного объекта,
    за исключение рассматриваемого объекта
    '''
    min_var = 0.1
    
    '''
    Параметр, определяющий минимальное расстояние до целевой метки и самого
    дальнего аттрибута между кандидатом на добавление в набор Q и ближайшим
    к нему объектом из набора Q
    '''
    tsStep = 0.1
    '''
    Инициализация массива для формирования тренировочного датасета без
    выбросов
    '''
    tsSTOLP = trainSetFunc
    
    # Инициализация переменной для формирования словаря с названиями столбцов
    dict_row = {}
    
    # Инициализация массива ошибки предсказания
    error_array =[]
    
    # Порядковый номер столбца целевой метки
    n_label = len(trainSetFunc.columns) - 1
    
    # Заполнение словаря названиями столбцов датасета
    for i in range(n_label+1):
      dict_row[trainSetFunc.columns[i]] = []
    
    # Преобразование словаря в пустой DataFrame
    Q = pd.DataFrame(dict_row)
    
    # Инициализация цикла прохода по тренировочному набору данных
    for i, row in trainSetFunc.iterrows():
        
      # Исключение рассматриваемого объекта из тренировочного датасета
      LOO_set = trainSetFunc.drop(labels = [i])
      
      # Вычисление предсказания очередного объекта
      sortDistLOO, sortLabelsLOO = KNN(k_func, LOO_set, row,
                                       one_elem_pred = True)
      a_LOO = regressionKNN(k_func, row, sortDistLOO, sortLabelsLOO, 
                            one_elem_pred = True)
      
      # Целевая метка рассматриваемого объекта
      current_label = row[trainSetFunc.columns[n_label]]
      
      # Разница между пресказанным и реальным значением целевой метки
      variance = abs(a_LOO - current_label)
      
      '''
      Прибавление разницы между delta и полученной разницы в массив ошибки
      для отображения на графике
      '''
      error_array.append(delta - variance)
      
      '''
      Если полученная разница больше delta, то объект является выбросом.
      
      Если объект не выброс, то при условии, что разница больше минимально
      допустимой, или расстояние до k+1 соседа больше допустимого, то
      объекта прибавляется с помощью инициализированной функции к набору Q
      '''
      if (variance > delta):
        tsSTOLP = tsSTOLP.drop(labels = [i])
      elif (variance > min_var or sortDistLOO[k_func] > max_dist):
        Q = add_prot(row, i, Q, n_label)
        
    # Сортировка массива ошибки
    error_array.sort()
    
    # Вывод результатов анализа тренировочного датасета на графике
    error_array = np.array(error_array)
    plt.figure(figsize=(10, 10))
    plt.plot(error_array, label = (chr(948) +" - |prediction - label|"))
    plt.fill_between(range(len(error_array)), max(error_array), 
                     min(error_array), where = (error_array < 0), color = "r", 
                     label = 'outliers')
    plt.fill_between(range(len(error_array)), max(error_array), 
                     min(error_array), 
                     where = (error_array <= min_var) & (error_array >= 0), 
                     color = "y", label = 'boundary objects')
    plt.fill_between(range(len(error_array)), max(error_array), 
                     min(error_array), where = (error_array > min_var), 
                     color = "g", 
                     label = 'objects with good prediction of its label')
    plt.title('margins distribution plot (max available difference between'+
              ' label and prediction of element = '+ 
              chr(948)+' = {})'.format(delta))
    plt.legend(bbox_to_anchor=(1.5,0.5))
    plt.show() 
    
    '''
    Вывод в консоль количества объектов в тренировочном наборе данных без
    выбросов и количества объектов в наборе Q
    '''
    print('num of train set without outliers: ', len(tsSTOLP))
    print('num of Q: ', len(Q))
    
    '''
    Вычисление параметра R^2 для объектов обучающей выборке без выбрасов,
    для которых целевые метки предсказывались с помощью набора Q
    '''
    sortDist_fit, sortResist_fit = KNN(k_func, Q, tsSTOLP)
    a_pred_test = regressionKNN(k_func, tsSTOLP, sortDist_fit, sortResist_fit)
    error = r2_score(tsSTOLP[tsSTOLP.columns[n_label]].to_numpy(), a_pred_test)
    print('r2 score: ', error)
    
    # инициализация счетчика цикла while
    i_loop = 0
    
    '''
    Определение количества атрибутов, которые учитываются при отборе
    кандидатов в набор Q
    '''
    far_fit = n_label - 1
    
    '''
    Инициализация цикла для уменьшения ошибки предсказация тренировочного
    набора данных
    
    Цикл завершается, если параметр R^2 в последней итерации оказалься
    больше заданного занчения
    '''
    while (error < l0):
        
      # Инициализация внутреннего цикла прохода по тренировочному датасету
      for i, row in tsSTOLP.iterrows():
        
        # массив для записи ближайшего к рассматриваемому объекту объекта из
        # набора Q, у которого определенное количество атрибутов ближе всего
        # к данному объекту
        nearest = np.zeros(n_label)
        nearest[...] = math.inf
        
        # массив для записи метки, ближайшей к рассматриваемой, из набора Q
        nearest_label = []
        for m in range(k_func + 1):
          nearest_label.append(math.inf)
          
         # Инициализация внутреннего массива для прохода по набору Q
        for j, row_prot in Q.iterrows():
            
          # массив для записи ближайших к рассматриваемому объекту из
          # тренировочного датасета атрибутов объекта из набора Q
          arr_far = []
          
          # разница между значениями целевых меток рассматриваемого объекта
          # из тренировочного датасета и рассматриваемого объекта из набора Q
          label_dist = abs(row[n_label] - row_prot[n_label])
          
          # Если эта разница меньше минимальной, полученной на предыдущих 
          # итерации цикла для рассматриваемого объекта из тренировочного
          # датасета, то текущая целевая метка становится ближайшей
          if (label_dist < nearest_label[0]):
            nearest_label = [label_dist] + nearest_label[:k_func]
            
          # Вычисление расстояний между соответсвтующими атрибутами 
          # рассматриваемого объекта из тренировочного датасета и 
          # рассматриваемого объекта из набора Q
          for k in range(n_label):
            arr_far.append(abs(row[k] - row_prot[k]))
            
          # сортитовка полученных расстояний
          arr_far = np.sort(arr_far[:n_label])
          '''
          Если заданное количество растояний между соответсвующими атрибутами
          рассматриваемого объекта из тренировочного датасета и 
          рассматриваемого объекта из набора Q меньше ближайшего, то оно
          становится ближайшим
          '''
          if (arr_far[far_fit] < nearest[far_fit]):
            nearest = arr_far
        
        # Вычисление предсказания очередного объекта из тренировочного
        # набора данных с помощью набора Q
        sortDistLOO, sortLabelsLOO = KNN(k_func, Q, row, one_elem_pred = True)
        a_LOO = regressionKNN(k_func, row, sortDistLOO, sortLabelsLOO, 
                              one_elem_pred = True)
        
        # Вычисление разницы между предсказанием и реальным занчением метки
        er_pred = abs(a_LOO - row[n_label])
        
        '''
        Если в наборе Q нет элемента, заданное количество атрибутов которого
        находятся дальше заданного расстояния от соответствующих атрибутов
        рассматриваемого объекта из тренировочного датасета и к ближайших
        целевых меток находятся дальше заданного расстояния, или
        предсказанное с помощью набора Q значение целевой метки различается
        с реальным значением целевой метки рассматриваемого объекта на
        заданное значение, то рассматриваемый элемент прибавляется к набору Q
        '''
        if ((nearest[far_fit] > tsStep and nearest_label[k_func-2] > tsStep) 
            or er_pred > max_var_in_Q):
          Q = add_prot(row, i, Q, n_label)
    
      '''
      Если установлен флаг исключения выбросов внутри цикла после добавления
      граничных объектов, начинается процесс исключения выбрасов
      '''
      if (Remove_negative == True):
          
        # Массив формирования датасета без выбрасов
        Q_wo_out = Q
        for i, row in Q.iterrows():
             
          # Вычисление целевой метки рассматриваемого объекта
          LOO_set = Q.drop(labels = [i])
          sortDistLOO, sortLabelsLOO = KNN(k_func, LOO_set, row, 
                                           one_elem_pred = True)
          a_LOO = regressionKNN(k_func, row, sortDistLOO, sortLabelsLOO, 
                                ne_elem_pred = True)
          
          # Целевая метка рассматриваемого объекта
          current_label = row[trainSetFunc.columns[n_label]]
          
          # Разница между пресказанным и реальным значением целевой метки
          variance = abs(a_LOO - current_label)
          
          # Если полученная разница больше delta, то объект является выбросом
          if (variance > delta):
            Q_wo_out = Q_wo_out.drop(labels = [i])
            
        # Запись в Q начального датасета без выбрасов
        Q = Q_wo_out
        
      # Вывод разделяющей линии в консоль
      print("|-----------------------------------------|")
      
      # инкрементирование счетчика цикла while
      i_loop += 1
      
      # Расчёт парамета R^2 с текущим датасетом Q
      sortDist_fit, sortResist_fit = KNN(k_func, Q, tsSTOLP)
      a_pred_test = regressionKNN(k_func, tsSTOLP, sortDist_fit, 
                                  sortResist_fit)
      error = r2_score(tsSTOLP[tsSTOLP.columns[n_label]].to_numpy(), 
                       a_pred_test)
      
      # Вывод количества элементов в датасете Q и текущего параметра R^2
      # в консоль
      print('num of Q: ', len(Q))
      print('r2_csore: ', error)
      
      # Если цикл while на 10 итерации, то цикл while прерывается
      if (i_loop == 10):
        break
    return Q, tsSTOLP
