"""Запуск всех проверок разом.

    python run_tests.py

Гоняйте перед каждым обновлением на сервере — это дешевле, чем узнавать
о поломке от пользователей.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SUITES = [
    ("tests_pipeline.py", "ядро: разбор ответа LLM, субтитры, тайминги"),
    ("tests_security.py", "безопасность: CSRF, лимиты, пути, заголовки"),
    ("tests_multiuser.py", "многопользовательский режим: изоляция, квоты"),
    ("tests_storage.py", "хранилище: ключи, подписанные ссылки, удаление"),
    ("tests_ui.py", "интерфейс: все кнопки и адреса реально работают"),
]


def main() -> None:
    results = []
    for script, what in SUITES:
        path = ROOT / script
        if not path.exists():
            print(f"— пропущен {script} (нет файла)")
            continue
        print(f"\n{'=' * 62}\n{script} — {what}\n{'=' * 62}")
        proc = subprocess.run([sys.executable, str(path)], cwd=str(ROOT))
        results.append((script, proc.returncode == 0))

    print(f"\n{'=' * 62}")
    for script, ok in results:
        print(f"  {'ПРОЙДЕН ' if ok else 'ПРОВАЛЕН'}  {script}")
    failed = [s for s, ok in results if not ok]
    if failed:
        print(f"\nПровалено наборов: {len(failed)}. Обновлять сервер нельзя.")
        sys.exit(1)
    print("\nВсе проверки пройдены.")


if __name__ == "__main__":
    main()
