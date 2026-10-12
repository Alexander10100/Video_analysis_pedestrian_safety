# Как вносить изменения

## Окружение (один раз)

```bash
pipx install poetry                                       # Poetry 2.x
poetry install                                            # ./.venv + все зависимости
poetry run pre-commit install -t pre-commit -t pre-push   # git-хуки
```

> Виртуальное окружение `.venv` в git **не коммитится**. Воспроизводимость
> обеспечивают `pyproject.toml` (что нужно) и `poetry.lock` (какие именно версии).

## Ветки и Pull Request

- `main_branch` защищена: прямой коммит в неё блокирует хук `no-commit-to-branch`.
- Работа идёт в ветках `feature/…`, `fix/…`, `refactor/…`, `docs/…`.
- Вливание только через Pull Request, CI должен быть зелёным (lint + test).
- После merge в `main_branch` CI сам собирает и публикует Docker-образы.

## Коммиты

Формат [Conventional Commits](https://www.conventionalcommits.org/ru/):
`<тип>: <что сделано>`. Типы: `feat`, `fix`, `refactor`, `docs`, `test`,
`build` (зависимости), `ci`, `chore`.

```
fix: не терять каждый второй кадр в stream_detect
```

## Что происходит при `git commit`

Хуки из `.pre-commit-config.yaml` проверяют **изменённые файлы**:

1. гигиена: пробелы в конце строк, перевод строки в конце файла, валидность YAML/TOML/JSON,
   нет конфликтов слияния, нет больших файлов (> 1 МБ) и приватных ключей;
2. `ruff check --fix`: линтер, часть замечаний исправляет сам;
3. `ruff format`: форматирование;
4. `mypy`: типы в `src/`;
5. `poetry check --lock`: lock-файл соответствует `pyproject.toml`;
6. `poetry export`: `requirements.txt` пересобирается из `poetry.lock`.

Если хук что-то исправил, коммит прерывается. Добавьте исправления (`git add -u`)
и закоммитьте ещё раз. При `git push` дополнительно запускаются все тесты.

## Зависимости

```bash
poetry add <пакет>            # рабочая зависимость
poetry add --group dev <пакет> # только для разработки
poetry lock                    # после ручной правки pyproject.toml
```

Коммитьте `pyproject.toml`, `poetry.lock` и `requirements.txt` вместе.
`requirements.txt` руками не редактируется.

## Код

- Новый код — в `src/pedestrian_safety/`: доменная логика в `core/`, веб в
  `services/`, CLI и пакетная обработка в `pipelines/`.
- Пути и параметры окружения берутся только из `config.get_settings()`, без хардкода.
- `logger = logging.getLogger(__name__)` вместо `print`.
- На новую логику — тест в `tests/`. Проверка перед PR:

```bash
poetry run pre-commit run --all-files && poetry run pytest --cov
```
