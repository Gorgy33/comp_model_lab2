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
    with open(filename, 'w') as file:
        json.dump(data, file)


def generate_list(function, x: list, n: int) -> list:
    for i in range(1, n):
        x.append(function(x[i], x[i-1]))
    return x


if __name__ == "__main__":
    input_data_path = "input_data"
    output_data_path = "output_data"
    params = get_from_json(f"{input_data_path}/main_params.json")
    validate_params(params=params)
    x = get_from_json(f"{input_data_path}/initial_data.json")
    f = lambda x, prev_x: (params['a']*(x**2) + params['b']*(prev_x**2) + params['c']) % params['m']
    x = generate_list(function=f, x=x['x'], n=params['n'])
    write_to_json(f"{output_data_path}/generated_data.json", {"data": x})
