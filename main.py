import itertools
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from os import makedirs
from os.path import dirname
from scipy import stats, integrate
import sympy


class ValidateException(Exception):
    pass

def get_from_json(path: str):
    """Получение параметров из json-файла

    Args:
        path: Путь до json-файла

    Return:
        Словарь параметров
    """
    with open(path, 'r') as file:
        return json.load(file)


def validate_params(params: dict):
    """Проверка, что коэффициенты положительные

    Args:
        params:
    """
    for i in params.keys():
        if params[i] < 0:
            raise ValidateException(f"Коэффициенты должны быть положительны! Коэффициент {i} < 0")


def write_to_json(filename: str, data):
    """Запись в новый json-файл

    Args:
        filename: Имя файла для записи
        data: Данные для записи
    """
    makedirs(dirname(filename), exist_ok=True)
    with open(filename, 'w+') as file:
        json.dump(data, file)


def generate_list(function, x: list, n: int) -> list:
    """Генерация псевдослучайной последовательности

    Args:
        function: Функция-генератор псевдослучайной последовательности
        x: Список начальных значений
        n: Длина последовательности
    """
    for i in range(1, n):
        x.append(function(x[i], x[i-1]))
    return x


def most_frequent(List):
    return max(set(List), key = List.count)

def get_period(x):
    """Поиск длины периода псевдослучайной последовательности

    Args:
        x: Псевдослучайная последовательность
    """
    # TODO: Доработать поиск периода, скорей всего не работает в случае когда периода нет
    x.reverse()
    t = 1 # Подпериоды
    T = [] # Массив с найденными подпериодами
    step = 0 # Шаг
    # Выделение периода в псевдослучайной последовательности
    for _ in range(len(x)):
        if x[step] != x[t + step]:
            t += 1
            step = 0
        else:
            T.append(t)
            step += 1
    return most_frequent(T)


def draw_histogram(v, K, m):
    """ Отрисовка гистограмм частот попаданий в интервалы

    Args:
        v: список частот попаданий в интервалы
        K: количество интервалов
        m: параметр m ГПСЧ
    """
    tmp = np.arange(K)
    index = [] # Массив с граничными значениями интервалов
    for i in range(K-1):
        index.append((i+1) * (m // K + 1))
    index.append(m)
    plt.bar(tmp, v, width=0.95)
    plt.xticks(tmp + 0.5, index)
    plt.xlabel("Интервалы")
    plt.ylabel("Частоты")
    plt.title("Гистограмма частот")

    plt.show()

def count_Q(x, n):
    """ Подсчет количества перестановок Q

    Args:
        x: Исследуемая последовательность
        n: Длина последовательности
    """
    Q = 0
    for i in range(n - 1):
        if x[i] > x[i + 1]:
            Q += 1
    return Q


def test_1(x, n, U):
    """ Тест №1

    Args:
        x: Исследуемая последовательность
        n:  Длина последовательности
        U: Квантиль уровня стандартного нормального распределения
    """

    Q = count_Q(x, n) # Подсчет количества перестановок
    # Построение доверительного интервала
    l = Q - (U * math.sqrt(n)) / 2 # Левая граница доверительного интервала
    u = Q + (U * math.sqrt(n)) / 2 # Правая граница доверительного интервала
    M = n / 2 # Математическое ожидание числа перестановок
    if (M >= l and M <= u): # Если мат.ожидание попадает в доверительный интервал, то тест пройден
        # TODO: Отправить результат в main (вывод в файл)
        print(f"Тест 1 пройден для n = {n}. Доверительный интервал: [{'%.3f' % l}, {'%.3f' % u}].")
    else:
        # TODO: Отправить результат в main (вывод в файл)
        print(f"Тест 1 не пройден для n = {n}.")


def confidence_interval_test_2(v, U, n, K):
    """ Построение доверительного интервала для частот

    Args:
        v: Относительная частота попадания элементов в интервал
        U: Квантиль уровня стандартного нормального распределения
        n: Длина последовательности
        K: Количество интервалов разбиения
    """
    l = v - (U / K) * math.sqrt((K - 1) / n) # Левая граница доверительного интервала
    u = v + (U / K) * math.sqrt((K - 1) / n) # Правая граница доверительного интервала
    return l, u


def M_x(X):
    """ Оценка математического ожидания случайной величины

    Args:
        X: Исследуемая последовательность
    """
    M = 0
    for x in X:
        M += x
    return M / len(X)


def D_x(X):
    """ Оценка дисперсии случайной величины

    Args:
        X: Исследуемая последовательность
    """
    Mx = M_x(X)
    D = 0
    for x in X:
        D += (x - Mx) ** 2
    return D / (len(X) - 1)


def test_2(x, n, m, K, U, alpha,  show_hist = True):
    """ Тест №2

    Args:
        x: Исследуемая последовательность
        n: Длина последовательности
        m: Параметр m ГПСЧ
        K: Количество интервалов
        U: Квантиль уровня стандартного нормального распределения
        alpha: Уровень значимости
        show_hist: Флаг отрисовки диагарамм (т.к. для теста 3 не нужно отрисовывать)
    """
    x_n = x[:n] # Взятие подпоследовательности необходимой длины
    hits, bins = np.histogram(x_n, bins=K, range=(0, m)) # Подсчет попаданий элементов в интервалы (hits: Кол-во попаданий. Bins: Интервалы.)
    v = []
    message = ""
    for h in hits:
        v.append(h/n) # Подсчет относительных частот
    if show_hist:
        draw_histogram(v, K, m) # Отрисовка гистограмм
    M = M_x(x_n) # Математическое ожидание
    D = D_x(x_n) # Дисперсия
    P = 1 / K # Теоретическая вероятность попадания в интервал
    for v_i in v:
        l, u = confidence_interval_test_2(v_i, U, n, K) # Доверительный интервал
        # TODO: Вывести доверительные интервалы в файл
        print(f"[{'%.3f' % l}, {'%.3f' % u}]")
        # Если не попадает в доверительный интервал
        if (P < l or P > u):
            message += f"Тест не пройден для частоты v = {v_i} при  n = {n}.\n"
    M_teor = m / 2 # Теоретическое мат.ожидание
    D_teor = m ** 2 / 12 # Теоретическая дисперсия
    # Построение доверительного интервала для мат.ожидания
    l_M = M - U * math.sqrt(D) / math.sqrt(n)
    u_M = M + U * math.sqrt(D) / math.sqrt(n)
    # Если не попадает в доверительный интервал
    if (M_teor < l_M or M_teor > u_M):
        message += f"Тест 2 не пройден для математического ожидания при  n = {n}.\n"
    # Построение доверительного интервала для дисперсии
    l_D = (n-1) * D / stats.chi2.ppf(1 - alpha / 2, n - 1)
    u_D = (n-1) * D / stats.chi2.ppf(alpha / 2, n - 1)
    # Если не попадает в доверительный интервал
    if (D_teor < l_D or D_teor > u_D):
        message += f"Тест 2 не пройден для дисперсии при  n = {n}.\n"
    if message == "":
        # TODO: Передать результаты в main (вывести в файл)
        print(f"Тест 2 пройден для n = {n}.")
    else:
        print(message)


def test_3(x, n, m, K, r,  U, alpha):
    """ Тест 3

    Args:
        x: Исследуемая последовательность
        n: Длина последовательности
        m: Параметр m ГПСЧ
        K: Количество интервалов
        r: Количество подпоследовательностей
        U: Квантиль уровня стандартного нормального распределения
        alpha: Уровень значимости
    """
    t = (n - 1) // r # Длина подпоследовательности
    x_n = []
    # Формирование подпоследовательностей
    for i in range(r):
        tmp = []
        for j in range(t+1):
            tmp.append(x[j*r+i])
        x_n.append(tmp)
    # Проверка подпоследовательностей по тестам 1, 2
    # TODO: Решить что выводить для 3 теста, надо ли дов. интервалы для каждой частоты
    for seq in x_n:
        test_1(seq, len(seq), U)
        test_2(seq, len(seq), m, K, U, alpha, False)


def chi2(x, m, n, alpha):
    """ Критерий хи-квадрат

    Args:
        x: Исследуемая подпоследовательность
        m: Параметр m ГПСЧ
        alpha: Уровень значимости
    :return:
    """
    x = x[-n:]
    K = math.ceil(math.log(n, 2) + 1) # формула Старджесса для подсчета кол-ва интервалов
    hits, bins = np.histogram(x, bins=K, range=(0, m)) # Подсчет попаданий элементов в интервалы (hits: Кол-во попаданий. Bins: Интервалы.)
    P = 1 / K # Теоретическая вероятность попадания в интервал
    # Подсчет значения статистики критерия
    S = 0
    for i in range(K):
        S += (((hits[i] / n) - P) ** 2) / P
    S *= n
    # Подсчет относительных частот опадания в интервал
    v = []
    for h in hits:
        v.append(h/n)
    draw_histogram(v, K, m) # Отрисовка гистограмм
    r = K - 1 # Число степеней свободы
    # Подсчет достигнутого уровня значимости
    P_S = (1 / (2 ** (r / 2) * math.gamma(r/2))) * \
          integrate.quad(lambda j: (j ** (r / 2 - 1)) * (math.exp(-j / 2)), S, np.inf)[0]

    S_krit = stats.chi2.ppf(1 - alpha, K - 1) # Критическое значение статистики

    alpha = (1 / (2 ** (r / 2) * math.gamma(r/2))) * \
          integrate.quad(lambda j: (j ** (r / 2 - 1)) * (math.exp(-j / 2)), S_krit, np.inf)[0]
    # Проверка, отвергается гипотеза или нет
    if P > alpha:
        # TODO: Вывести результат в файл
        print("Гипотеза не отвергается")
    else:
        print("Гипотеза отвергается")


def сrit_omega2_Anderson_Darling(x, m, n, alpha):
    """ Критерий сигма-квадрат-Андерсона-Дарлинга
        (Непараметрический критерий по варианту)

    Args:
        x: Исследуемая последовательность
        m: Параметр m ГПСЧ
        n: Длина последовательности
        alpha: Уровень значимости
    """
    x = x[-n:]
    x.sort() # Сортировка последовательности
    tmp = 0
    F = lambda x: x / m # Функция распределения
    # Подсчет значения статистики критерия
    for i in range(n):
        tmp += ((2 * i - 1) / (2 * n)) * math.log(F(x[i])) \
               + (1 - (2 * i - 1) / (2 * n)) * math.log(1 - F(x[i]))
    S = - n - 2 * tmp
    # Подсчет функции распределения a2(S)
    a2 = 0
    for j in range(170):
        a2 += ((-1) ** j) * (math.gamma(j + 0.5) * (4 * j + 1)) \
              / (math.gamma(0.5) * math.gamma(j + 1)) *\
              math.exp((-(4 * j + 1) ** 2 * math.pi ** 2) / (8 * S)) * \
              integrate.quad(lambda y: math.exp((S / (8 * (y ** 2 + 1))) - ((4 * j + 1) ** 2 * math.pi ** 2 * y ** 2) / (8 * S)), 0, np.inf)[0]
    a2 *= math.sqrt(2 * math.pi) / S
    P = 1 - a2 # Достигнутый уровень значимости
    if P > alpha:
        print("Гипотеза не отвергается")
    else:
        print("Гипотеза отвергается")


if __name__ == "__main__":
    input_data_path = "input_data"
    output_data_path = "output_data"
    # Получение исходных параметров
    params = get_from_json(f"{input_data_path}/main_params.json")
    validate_params(params=params)
    # Получение начальных параметров генератора
    x = get_from_json(f"{input_data_path}/initial_data.json")
    # Функция-генератор последовательности
    f = lambda x, prev_x: (params['a']*(x**2) + params['b']*(prev_x**2) + params['c']) % params['m']
    # Генерация последовательности
    x = generate_list(function=f, x=x['x'], n=params['n'])

    write_to_json(f"{output_data_path}/generated_data.json", {"data": x})

    # Получение периода последовательности
    T = get_period(x.copy())
    if T < 100:
        # TODO: понять что делать если меньше 100
        print("Период меньше 100")
    period_elements_list = x[-T:] # Элементы периода последовательности

    m = params['m'] # Параметр m ГСПЧ
    n_40 = 40
    n_100 = 100
    a = 0.05 # alpha
    U = 1.96 # Квантиль стандартного нормального распределения с уровнем 1-a/2 = 0.975

    test_1(period_elements_list, n_40, U)
    test_1(period_elements_list, n_100, U)

    K_test_2 = 8 # Параметр К для 2 теста

    # Параметры для теста 3
    r = 4
    K_test_3 = 20

    # TODO: Доделать выводы из тестов 

    test_2(period_elements_list, n_40, m, K_test_2, U, a)
    test_2(period_elements_list, n_100, m, K_test_2, U, a)

    test_3(period_elements_list, n_40, m, K_test_3, r,  U, a)
    test_3(period_elements_list, n_100, m, K_test_3, r,  U, a)

    chi2(x.copy(), n_40, m, a)

    сrit_omega2_Anderson_Darling(x.copy(), m, n_100, a)
















