import itertools
import json
from os import makedirs
from os.path import dirname


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
    for i in range(1, n):
        x.append(function(x[i], x[i-1]))
    return x


def alg_floyd(f, x, x_prev):
    a = x_prev
    a_prev = x_prev
    b_prev = x_prev
    b = x
    while a != b:
        if a != x_prev:
            a_tmp = a
            a = f(a, a_prev)
            a_prev = a_tmp
        else:
            a_prev = a
            a = x
        c = f(b, b_prev)

        b = f(c, b)
        b_prev = c
    b_prev = b
    b = f(a, a_prev)
    T = 1
    while a != b:
        b_tmp = b
        b = f(b, b_prev)
        b_prev = b_tmp
        T += 1
    return T


def alg_brent(f, x, x_prev):
    c = x_prev
    a = x
    n = 1
    t = 1
    if a == c:
        return 1
    while (True):
        if n == t:
            c_prev = c
            c = a
            t *= 2
        a_prev = a
        a = f(a, x_prev)
        n += 1
        if n >= 3*t/4:
            if a == c:
                break
        else:
            break
    T = 1
    a_prev = a
    a = f(c, c_prev)
    while a != c:
        a_tmp = a
        a = f(a, a_prev)
        a_prev = a_tmp
        T += 1
    return T






def floyd(f, x, x_prev):
    tortoise_prev = x_prev
    tortoise = x  # f(x0) является элементом, следующим за x0.
    hare_prev = tortoise
    hare = f(tortoise, tortoise_prev)
    while tortoise != hare:
        tmp_tortoise = tortoise
        tortoise = f(tortoise, tortoise_prev)
        tortoise_prev = tmp_tortoise
        tmp_hare = hare
        hare = f(hare, hare_prev)
        hare_prev = tmp_hare
        tmp_hare = hare
        hare = f(hare, hare_prev)
        hare_prev = tmp_hare

    mu = 0
    tortoise = x_prev
    tortoise_prev = x_prev
    while tortoise != hare:
        if tortoise == x_prev:
            tortoise = x
        else:
            tmp_tortoise = tortoise
            tortoise = f(tortoise, tortoise_prev)
            tortoise_prev = tmp_tortoise
        tmp_hare = hare
        hare = f(hare, hare_prev)  # Заяц и черепаха двигаются с одинаковой скоростью
        hare_prev = tmp_hare
        mu += 1
    # Находим длину кратчайшего цикла, начинающегося с позиции x_μ
    # Заяц движется на одну позицию вперёд,
    # в то время как черепаха стоит на месте.
    lam = 1
    hare_prev = hare
    hare = f(tortoise, tortoise_prev)
    while tortoise != hare:
        tmp_hare = hare
        hare = f(hare, hare_prev)
        hare_prev = tmp_hare
        lam += 1
    return lam, mu


def brent(f, x, x_prev):
    # Основная фаза: ищем степень двойки
    power = lam = 1
    tortoise = x_prev
    hare_prev = tortoise
    hare = x  # f(x0) — элемент/узел, следующий за x0.
    while tortoise != hare:
        if power == lam:  # время начать новую степень двойки?
            tortoise = hare
            power *= 2
            lam = 0
        tmp_hare = hare
        hare = f(hare, hare_prev)
        hare_prev = tmp_hare
        lam += 1

    # Находим позицию первого повторения длины λ
    mu = 0
    tortoise = hare = x_prev

    for i in range(lam):
        if hare == x_prev:
            hare_prev = hare
            hare = x
        else:
            #   range(lam) образует список со значениями 0, 1, ... , lam-1
            tmp_hare = hare
            hare = f(hare, hare_prev)
            hare_prev = tmp_hare
    # расстояние между черепахой и зайцем теперь равно λ.

    # Теперь черепаха и заяц движутся с одинаковой скоростью, пока не встретятся
    while tortoise != hare:
        if tortoise == x_prev:
            tortoise_prev = tortoise
            tortoise = x
        else:
            tmp_tortoise = tortoise
            tortoise = f(tortoise, tortoise_prev)
            tortoise_prev = tmp_tortoise
        tmp_hare = hare
        hare = f(hare, hare_prev)
        hare_prev = tmp_hare
        mu += 1
    return lam, mu


def period1(x):  # поиск периода
    per = 1
    step = 0
    l = len(x)
    while per + step != len(x):
        if x[step] != x[per + step]:
            per += 1
            step = 0
        else:
            tmp_per = per
            step += 1
    return per


def period2(seq):
    seq.reverse()
    s0, s1 = seq[0], seq[1]
    T = 1

    for i in range(1, len(seq) - 1):
        if (s0 != seq[i] or s1 != seq[i + 1]):
            T += 1
        else:
             break
    return T


def v(n):
    i = 1
    while n >= 2 ** i:
        i += 1
    return (i - 1)


if __name__ == "__main__":
    input_data_path = "input_data"
    output_data_path = "output_data"
    params = get_from_json(f"{input_data_path}/main_params.json")
    validate_params(params=params)
    x = get_from_json(f"{input_data_path}/initial_data.json")
    f = lambda x, prev_x: (params['a']*(x**2) + params['b']*(prev_x**2) + params['c']) % params['m']
    x = generate_list(function=f, x=x['x'], n=params['n'])
    write_to_json(f"{output_data_path}/generated_data.json", {"data": x})
    lam, mu = floyd(f, x[1], x[0])
    print(lam)
    print(mu)
    print(x[mu:mu+lam])
    lam, mu = brent(f, x[1], x[0])
    print(lam)
    print(mu)
    print(x[mu:mu+lam])
    #invx = x
    #.reverse()
    T = alg_floyd(f, x[1], x[0])
    print(T)
    T = alg_brent(f, x[1], x[0])
    print(T)



