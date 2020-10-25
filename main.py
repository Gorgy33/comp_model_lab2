import json
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from os import makedirs
from os.path import dirname
from scipy import stats, integrate

# Исключения
class ValidateException(Exception):
    pass


class PeriodException(Exception):
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
        if isinstance(params[i], list):
            for elem in params[i]:
                if elem < 0:
                    raise ValidateException(f"Коэффициенты должны быть положительны! Коэффициент {i} "
                                            f"содержит значение < 0. {elem} < 0")
        elif params[i] < 0:
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


def most_frequent(array):
    """ Выбор самого часто встречающегося элемента в списке

    Args:
        array: исследуемый список
    """
    if len(set(array)) == len(array):
        return 0
    return max(set(array), key=array.count)


def get_period(array):
    """Поиск длины периода псевдослучайной последовательности

    Args:
        array: Псевдослучайная последовательность
    """
    array.reverse()
    t = 1  # Подпериоды
    T = []  # Массив с найденными подпериодами
    step = 0  # Шаг
    # Выделение периода в псевдослучайной последовательности
    for _ in range(len(array)):
        if (t + step) < len(array) and array[step] != array[t + step]:
            t += 1
            step = 0
        else:
            T.append(t)
            step += 1
    T.append(t)
    period = most_frequent(T)
    return period if period else len(array)


def draw_histogram(v, K, m, message):
    """ Отрисовка гистограмм частот попаданий в интервалы

    Args:
        v: список частот попаданий в интервалы
        K: количество интервалов
        m: параметр m ГПСЧ
    """
    tmp = np.arange(K)
    index = []  # Массив с граничными значениями интервалов
    for i in range(K-1):
        index.append((i+1) * (m // K + 1))
    index.append(m)
    plt.bar(tmp, v, width=0.95)
    plt.xticks(tmp + 0.5, index)
    plt.xlabel("Интервалы")
    plt.ylabel("Частоты")
    plt.title(f"Гистограмма частот {message}")

    plt.show()


def count_Q(array, n):
    """ Подсчет количества перестановок Q

    Args:
        array: Исследуемая последовательность
        n: Длина последовательности
    """
    count = 0
    for i in range(n - 1):
        if array[i] > array[i + 1]:
            count += 1
    return count


def test1(array, n, u):
    """ Тест №1

    Args:
        array: Исследуемая последовательность
        n:  Длина последовательности
        u: Квантиль уровня стандартного нормального распределения
    """

    count = count_Q(array, n)  # Подсчет количества перестановок
    # Построение доверительного интервала
    left_interval = count - (u * math.sqrt(n)) / 2  # Левая граница доверительного интервала
    right_interval = count + (u * math.sqrt(n)) / 2  # Правая граница доверительного интервала
    math_expectation = n / 2  # Математическое ожидание числа перестановок
    # Если мат.ожидание попадает в доверительный интервал, то тест пройден
    if left_interval <= math_expectation <= right_interval:
        return True, {"confidence interval": [f"[{'%.3f' % left_interval}", f"{'%.3f' % right_interval}]"]}
    else:
        return False, {"confidence interval": [f"[{'%.3f' % left_interval}", f"{'%.3f' % right_interval}]"]}


def confidence_interval_test_2(v, U, n, K):
    """ Построение доверительного интервала для частот

    Args:
        v: Относительная частота попадания элементов в интервал
        U: Квантиль уровня стандартного нормального распределения
        n: Длина последовательности
        K: Количество интервалов разбиения
    """
    left_interval = v - (U / K) * math.sqrt((K - 1) / n)  # Левая граница доверительного интервала
    right_interval = v + (U / K) * math.sqrt((K - 1) / n)  # Правая граница доверительного интервала
    return left_interval, right_interval


def math_expectation_by_x(array):
    """ Оценка математического ожидания случайной величины

    Args:
        array: Исследуемая последовательность
    """
    return sum(array) / len(array)


def dispersion_of_x(array, math_expectation):
    """ Оценка дисперсии случайной величины

    Args:
        array: Исследуемая последовательность
        math_expectation: Математическое ожидание
    """
    dispersion = 0
    for element in array:
        dispersion += (element - math_expectation) ** 2
    return dispersion / (len(array) - 1)


def test2(array, n, m, K, U, alpha, show_hist = True):
    """ Тест №2

    Args:
        array: Исследуемая последовательность
        n: Длина последовательности
        m: Параметр m ГПСЧ
        K: Количество интервалов
        U: Квантиль уровня стандартного нормального распределения
        alpha: Уровень значимости
        show_hist: Флаг отрисовки диагарамм
    """
    x_n = array[:n]  # Взятие подпоследовательности необходимой длины
    # Подсчет попаданий элементов в интервалы (hits: Кол-во попаданий. Bins: Интервалы.)
    hits, bins = np.histogram(x_n, bins=K, range=(0, m))
    frequency = []
    result = {}
    result.update({'errors': []})
    for h in hits:
        frequency.append(h/n)  # Подсчет относительных частот
    if show_hist:
        draw_histogram(frequency, K, m, f"для теста №2 при n ={n}") # Отрисовка гистограмм
    math_expectation = math_expectation_by_x(x_n)  # Математическое ожидание
    dispersion = dispersion_of_x(x_n, math_expectation)  # Дисперсия
    P = 1 / K  # Теоретическая вероятность попадания в интервал
    result.update({'frequency': {}})
    result['frequency'].update({'elements': frequency})
    result['frequency'].update({'errors_elements': []})
    result['frequency'].update({'confidence interval': []})
    for v_i in frequency:
        l, u = confidence_interval_test_2(v_i, U, n, K)  # Доверительный интервал
        result['frequency']['confidence interval'].append(f"[{'%.3f' % l}; {'%.3f' % u}]")
        # print(f"[{'%.3f' % l}, {'%.3f' % u}]")
        # Если не попадает в доверительный интервал
        if P < l or P > u:
            result['frequency']['errors_elements'].append(v_i)
    if result['frequency']['errors_elements']:
        result['errors'].append(f"Test 2 failed for frequencies {result['frequency']['errors_elements']}"
                                f" when  n = {n}.")

    math_expectation_teor = m / 2  # Теоретическое мат.ожидание
    dispersion_teor = m ** 2 / 12  # Теоретическая дисперсия
    # Построение доверительного интервала для мат.ожидания
    left_interval_for_math_expectation = math_expectation - U * math.sqrt(dispersion) / math.sqrt(n)
    right_interval_for_math_expectation = math_expectation + U * math.sqrt(dispersion) / math.sqrt(n)
    # Если не попадает в доверительный интервал
    if math_expectation_teor < left_interval_for_math_expectation\
            or math_expectation_teor > right_interval_for_math_expectation:
        result['errors'].append(f"Test 2 failed for mathematical expectation when  n = {n}.")
    # Построение доверительного интервала для дисперсии
    left_interval_for_dispersion = (n-1) * dispersion / stats.chi2.ppf(1 - alpha / 2, n - 1)
    right_interval_for_dispersion = (n-1) * dispersion / stats.chi2.ppf(alpha / 2, n - 1)
    # Если не попадает в доверительный интервал
    if dispersion_teor < left_interval_for_dispersion or dispersion_teor > right_interval_for_dispersion:
        result['errors'].append(f"Test 2 failed for variance when n = {n}.")
    return (False if result['errors'] else True), result


def test3(array, n, m, K, r, U, alpha):
    """ Тест 3

    Args:
        array: Исследуемая последовательность
        n: Длина последовательности
        m: Параметр m ГПСЧ
        K: Количество интервалов
        r: Количество подпоследовательностей
        U: Квантиль уровня стандартного нормального распределения
        alpha: Уровень значимости
    """
    t = (n - 1) // r  # Длина подпоследовательности
    x_n = []
    # Формирование подпоследовательностей
    for i in range(r):
        tmp = []
        for j in range(t+1):
            tmp.append(array[j * r + i])
        x_n.append(tmp)
    # Проверка подпоследовательностей по тестам 1, 2
    result = {'seq_result': {}}
    success_count = 0
    for seq in x_n:
        result['seq_result'].update(
            {
                f'{seq[0]}-{seq[-1]}':
                    {
                        'test1': test1(seq, len(seq), U),
                        'test2': test2(seq, len(seq), m, K, U, alpha, False)
                    }
            }
        )
        success_count += result['seq_result'][f'{seq[0]}-{seq[-1]}']['test1'][0]\
                         + result['seq_result'][f'{seq[0]}-{seq[-1]}']['test2'][0]

    return (True if success_count == len(x_n)*2 else False), result


def chi2(array, n, m, alpha):
    """ Критерий хи-квадрат

    Args:
        array: Исследуемая подпоследовательность
        n: Длина последовательности
        m: Параметр m ГПСЧ
        alpha: Уровень значимости
    :return:
    """
    interval_count = math.ceil(math.log(n, 2) + 1)  # формула Старджесса для подсчета кол-ва интервалов
    hits, bins = np.histogram(array, bins=interval_count, range=(0, m))  # Подсчет попаданий элементов в интервалы (hits: Кол-во попаданий. Bins: Интервалы.)
    P = 1 / interval_count  # Теоретическая вероятность попадания в интервал
    # Подсчет значения статистики критерия
    S = 0
    for i in range(interval_count):
        S += (((hits[i] / n) - P) ** 2) / P
    S *= n
    # Подсчет относительных частот опадания в интервал
    v = []
    for h in hits:
        v.append(h/n)
    draw_histogram(v, interval_count, m, "для критерия хи-квадрат")  # Отрисовка гистограмм
    r = interval_count - 1  # Число степеней свободы
    # Подсчет достигнутого уровня значимости
    P_S = (1 / (2 ** (r / 2) * math.gamma(r/2))) * \
          integrate.quad(lambda j: (j ** (r / 2 - 1)) * (math.exp(-j / 2)), S, np.inf)[0]

    # S_krit = stats.chi2.ppf(1 - alpha, interval_count - 1)  # Критическое значение статистики
    #
    # alpha = (1 / (2 ** (r / 2) * math.gamma(r/2))) * \
    #       integrate.quad(lambda j: (j ** (r / 2 - 1)) * (math.exp(-j / 2)), S_krit, np.inf)[0]
    # Проверка, отвергается гипотеза или нет
    return P_S > alpha, [f"{'%.3f' % P_S}", f"{'%.3f' % S}"]


def crit_omega2_anderson_darling(array, n, m, alpha):
    """ Критерий сигма-квадрат-Андерсона-Дарлинга
        (Непараметрический критерий по варианту)

    Args:
        array: Исследуемая последовательность
        n: Длина последовательности
        m: Параметр m ГПСЧ
        alpha: Уровень значимости
    """
    array.sort()  # Сортировка последовательности
    S = 0
    # Подсчет значения статистики критерия
    for i, elem in enumerate(array):
        S += ((2 * i - 1) / (2 * n)) * math.log(elem/m) \
               + (1 - (2 * i - 1) / (2 * n)) * math.log(1 - elem/m)
    S = - n - 2 * S
    # Подсчет функции распределения a2(S)
    a2 = 0
    for j in range(170):
        a2 += ((-1) ** j) * (math.gamma(j + 0.5) * (4 * j + 1)) \
              / (math.gamma(0.5) * math.gamma(j + 1)) *\
              math.exp((-(4 * j + 1) ** 2 * math.pi ** 2) / (8 * S)) * \
              integrate.quad(lambda y: math.exp((S / (8 * (y ** 2 + 1))) - ((4 * j + 1) ** 2 * math.pi ** 2 * y ** 2) / (8 * S)), 0, np.inf)[0]
    a2 *= math.sqrt(2 * math.pi) / S
    P = 1 - a2  # Достигнутый уровень значимости
    return P > alpha, [f"{'%.3f' % P}", f"{'%.3f' % S}"]


if __name__ == "__main__":
    input_data_path = "input_data"
    output_data_path = "output_data"
    # Получение исходных параметров
    params = get_from_json(f"{input_data_path}/main_params.json")
    validate_params(params=params)
    # Получение начальных параметров генератора
    x = get_from_json(f"{input_data_path}/initial_data.json")
    # Генерация последовательности
    x = generate_list(
        function=lambda x, prev_x: (params['a']*(x**2) + params['b']*(prev_x**2) + params['c']) % params['m'],
        x=x['x'],
        n=params['n'])

    write_to_json(f"{output_data_path}/self/generated_data.json", {"data": x})
    m = params['m']  # Параметр m ГСПЧ
    randomized_array = [random.randint(1, m-1) for _ in range(params['n'])]
    write_to_json(f"{output_data_path}/standard/generated_data.json", {"data": x})
    array_list = [x, randomized_array]
    for id, x in enumerate(array_list):
        # Получение периода последовательности
        T = get_period(x.copy())
        write_to_json(f"{output_data_path}/{'standard' if id else 'self'}/period.json", {"period_length": T, "period_elements": x[-T:]})
        if T < 100:
            raise PeriodException(f"Период меньше 100. T = {T}")
        period_elements_list = x[-T:]  # Элементы периода последовательности

        test_list = [test1, test2, test3]
        for test in test_list:
            params = get_from_json(f"{input_data_path}/{test.__name__}.json")
            validate_params(params=params)
            alpha = params.get('alpha')
            k = params.get('k')
            U = stats.norm.ppf(1.0 - alpha/2)  # Квантиль стандартного нормального распределения с уровнем 1-a/2
            if k:
                r = params.get('r')
                if r:
                    for n in params.get('n'):
                        answer, data = test(period_elements_list, n, m, k, r, U, alpha)
                        write_to_json(
                            f"{output_data_path}/{'standard' if id else 'self'}/{test.__name__}_{n}_result.json",
                            {"result": answer, "data": data})
                else:
                    for n in params.get('n'):
                        answer, data = test(period_elements_list, n, m, k, U, alpha)
                        write_to_json(
                            f"{output_data_path}/{'standard' if id else 'self'}/{test.__name__}_{n}_result.json",
                            {"result": answer, "data": data})
            else:
                for n in params.get('n'):
                    answer, data = test(period_elements_list, n, U)
                    write_to_json(f"{output_data_path}/{'standard' if id else 'self'}/{test.__name__}_{n}_result.json",
                                  {"result": answer, "data": data})

        criterion_list = [chi2, crit_omega2_anderson_darling]

        for criterion in criterion_list:
            params = get_from_json(f"{input_data_path}/criterion_params.json")
            validate_params(params=params)
            alpha = params.get('alpha')
            n = params.get('n')
            answer, data = criterion(period_elements_list[-n:], n, m, alpha)
            write_to_json(f"{output_data_path}/{'standard' if id else 'self'}/{criterion.__name__}_result.json",
                          {"result": answer, "data": {
                              "achieved level of significance: ": data[0],
                              "statistics value: ": data[1]
                          }})
