# Сервис синхронного инференса: стоимость медицинской страховки

Веб-сервис на FastAPI, который по данным клиента предсказывает его годовые
расходы на медицинскую страховку (`charges`, доллары США).

Модель — градиентный бустинг (`GradientBoostingRegressor`) в составе
конвейера scikit-learn, который включает всю предобработку данных:
заполнение пропусков, стандартизацию количественных признаков и
One-Hot кодирование категориальных. На тестовой выборке R² ≈ 0,87,
средняя абсолютная ошибка ≈ 2 600 $.

Датасет: [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)
(1 338 записей, 6 признаков).

## Структура проекта

```
data/insurance.csv            датасет
notebooks/medical-cost.ipynb  анализ данных, обучение и сохранение модели
models/pipeline.pkl           сохранённый конвейер (создаётся ноутбуком)
src/main.py                   веб-сервис
src/model_utils.py            загрузка модели и инференс
test/                         модульные тесты
.github/workflows/            CI: проверка стиля кода и тесты
```

## Запуск (Windows, Anaconda Prompt)

```
conda activate D:\envs\cloudsc-env
pip install -r requirements.txt
```

1. Обучить модель: запустить `jupyter notebook`, открыть
   `notebooks/medical-cost.ipynb` и выполнить все ячейки.
   Появится файл `models/pipeline.pkl`.
2. Запустить тесты:
   ```
   set PYTHONPATH=.;src
   pytest test -v
   ```
3. Запустить сервис:
   ```
   cd src
   set MODEL_PATH=..\models\pipeline.pkl
   uvicorn main:app
   ```
4. Документация API: http://localhost:8000/docs

В Linux/macOS переменные задаются так:
`PYTHONPATH=./:./src/ pytest test` и
`MODEL_PATH="../models/pipeline.pkl" uvicorn main:app`.

## API

| Метод и адрес | Назначение | Авторизация |
| --- | --- | --- |
| `GET /healthcheck` | проверка работоспособности | не нужна |
| `POST /predictions` | предсказание расходов на страховку | Bearer-токен `00000` |

Тело запроса `POST /predictions`:

| Поле | Тип | Допустимые значения |
| --- | --- | --- |
| `age` | целое | 18–100 |
| `sex` | строка | `female`, `male` |
| `bmi` | число | больше 10 и меньше 70 |
| `children` | целое | 0–10 |
| `smoker` | строка | `yes`, `no` |
| `region` | строка | `northeast`, `northwest`, `southeast`, `southwest` |

Коды ответов:

| Код | Когда |
| --- | --- |
| 200 | успешно, тело `{"charges": <число>}` |
| 401 | нет токена (`Not authenticated`) или токен неверный (`Invalid authentication credentials`) |
| 422 | данные не прошли проверку: нет поля, неверный тип или значение вне диапазона |

Пример запроса (Windows `cmd`):

```
curl -i -X POST http://localhost:8000/predictions -H "Authorization: Bearer 00000" -H "Content-Type: application/json" -d "{\"age\": 19, \"sex\": \"female\", \"bmi\": 27.9, \"children\": 0, \"smoker\": \"yes\", \"region\": \"southwest\"}"
```
