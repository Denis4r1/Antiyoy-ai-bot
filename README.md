# RL Antiyoy Game 
# Участники
- Богданов Ярослав
- Ильин Павел
- Тураянов Денис
- Шаяхметов Аскар

# Инструкция по запуску
### Запуск API и инструментов

В файле `default.env` указан порт API-сервера и число воркеров:

EXTERNAL_PORT=<адрес сервера>\
GUNICORN_WORKERS=<число воркеров>

Запуск:

```bash
docker-compose -f docker-compose.yml --env-file default.env up --build
```

### Запуск игрового сервера

В файле `default.env` указан порт игрового сервера:

EXTERNAL_PORT_GAMESERVER=<адрес игрового сервера>\
GUNICORN_WORKERS=<число воркеров>

Запуск:

```bash
docker compose -f docker-compose.gameserver.yml --env-file default.env up --build
```

### Демо (иногда работает)
- [Проигрыватель логов с записью игры mcst](http://gameapi.afsh.space/log_player/demo/)
- [Редактор gamestate](http://gameapi.afsh.space/state_editor/)
- [Игровой сервер](http://antiyoygame.afsh.space/)


# Описание структуры

Есть два сервиса 

- **game-interface**: Stateless-сервис для обучения AI. Предоставляет три ручки:  
  - `generate_state` — генерация начального состояния игры,  
  - `get_actions` — получение легальных ходов для переданного состояния,  
  - `apply_action` — применение переданного хода к переданному состоянию и возврат нового состояния.  
- **game-server**: Stateful игровой сервер. Реализует API, WebSockets и функциональность лобби и игровых комнат, где игроки могут создавать комнаты, собирать участников и запускать игры.
## Корневой каталог
- **dataset.md**: Собранные игры
- **default.env**: Файл конфигурации для настройки портов и числа процессов сервисов
- **docker-compose.yml**: docker-compose для game-interface
- **docker-compose.gameserver.yml**: docker-compose для game-server
- **Dockerfile**: Dockerfile для контейнера game-interface
- **Dockerfile.gameserver**: Dockerfile контейнера игрового сервера.
- **gamerules.md**: Правила игры.
- **gunicorn_config.py**: конфиг для запуска игрового сервера
- **requirements.txt**: зависимости для game-interface
- **requirements-gameserver.txt**: зависимости для игрового сервера
- **requirements-dev.txt**: зависимости для github actions


## Каталоги

### `nginx/`

Конфигурационные файлы для nginx

- **default.conf**: конфигурация для game-interface
- **gameserver.conf**: Конфигурация Nginx, специфичная для game-server.


### `src/`

Основной каталог исходного кода

#### `src/ai/`

- **collect_bootstrap_data.py**: Скрипт для сбора игр mcts против mcts.
- **dqn_imp.py**: Реализация dqn.
- **hex_dqn_masked.pth**: веса dqn
- **mcts.py**: Реализация алгоритма Monte Carlo Tree Search (MCTS)

#### `src/game/`

Основная игровая логика.

- **gamecontroller.py**: управление игровой логикой, обработка, верификация действий игроков, обновление игрового состояния
- `core/`:
  - **const.py**: Константы, для отрисовки field
  - **field.py**: Реализация логики игрового поля, на шестиугольной сетке: генерация карты, применение ходов, учет доходов, территорий.
  - **tile.py**: Свойства отдельного тайла на поле

#### `src/game_interface/`

Апи для обучения. Реализует три ручки: 
1) generate_state генерирует исходный стейт
2) get_actions для переданного стейта возвращает легальные ходы
3) apply_action для переданного стейта применяет переданный ход и возвращает новый стейт

- **app.py**: Основная логика приложения для игрового интерфейса на FastAPI.
- **models.py**: Модели данных для запросов и ответов
- **utils.py**: Применение игровой логики

#### `src/server/`

Игровой сервер.

- **app.py**: Основное серверное приложение на FastAPI
- **config.py**: Настройки конфигурации сервера.
- `handlers/`:
  - **api.py**: Обработчики API-эндпоинтов для сервера.
  - **websockets.py**: Обработчики WebSocket для игры в реальном времени
- `models/`:
  - **managers.py**: управление WebSocket-соединениями для лобби и игровых комнат: подключение, отключение, рассылка сообщений, логирование, мониторинг.
  - **room.py**: управление комнатой: добавление удаление игроков, отслеживание готовности, запуск игры.
- `monitoring/`:
  - **logging_config.py**: Конфиг для логирования приложения.
  - **metrics.py**: Определение prometheus метрик приложения.
- `services/`:
  - **game_service.py**: управление комнатами: отслеживание их состояния, создание, удаление.

#### `src/utils/`
- **all_turns_ever.py**: нумерация всех возможных ходов для одной из карт.

### `web/`

Фронтенд

- `static/`:
  - `images/`:
    - **.png**: картинки юнитов, зданий
  - `js/`:
    - **compiled/**: Скомпилированные js файлы
    - **gamepage.ts**: Логика странички с игрой
    - **main.ts**: Логика лобби
  - `maps/`:
    - **map_basic_10x10.json**: небольшая карта
    - **map_basic_small.json**: карта еще меньше
  - `misc/`:
    - **demo_log.jsonl**: демо лог для {apibase}/log_player/demo
- `templates/`:
  - **gamepage.html**: HTML-шаблон для игровой страницы.
  - **log_player.html**: HTML-шаблон для проигрывателя логов
  - **newlobby.html**: HTML-шаблон для лобби
  - **state_editor.html**: HTML-шаблон редактора стейтов.
- **tsconfig.json**: конфиг ts



### Описание данных

Логи игры хранятся в лог-файлах. Каждый ход — это одна строка с JSON-объектом `gamestate`.

Описание структуры объекта находится здесь:

http://gameapi.afsh.space/docs
