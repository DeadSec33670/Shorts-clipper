"""Уведомления о готовности на почту.

ЗАЧЕМ
Обработка идёт 15+ минут. Держать вкладку открытой всё это время никто не
будет, а без письма человек либо забудет про задачу, либо будет заходить
проверять. Письмо снимает и то и другое.

НАСТРОЙКА
Задайте в окружении CLIPPER_SMTP_HOST и остальные параметры (см. env.example).
Если они не заданы, модуль молча ничего не делает: отсутствие настроенной
почты не должно ронять обработку — клипы важнее письма.
"""
from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid


def _cfg() -> dict | None:
    host = (os.environ.get("CLIPPER_SMTP_HOST") or "").strip()
    if not host:
        return None
    return {
        "host": host,
        "port": int(os.environ.get("CLIPPER_SMTP_PORT", "587")),
        "user": (os.environ.get("CLIPPER_SMTP_USER") or "").strip(),
        "password": os.environ.get("CLIPPER_SMTP_PASSWORD") or "",
        "from": (os.environ.get("CLIPPER_SMTP_FROM")
                 or os.environ.get("CLIPPER_SMTP_USER") or "").strip(),
        "from_name": os.environ.get("CLIPPER_SMTP_FROM_NAME", "Shorts Clipper"),
        "ssl": (os.environ.get("CLIPPER_SMTP_SSL", "") or "").lower()
               in ("1", "true", "yes"),
        "site": (os.environ.get("CLIPPER_SITE_URL") or "").strip().rstrip("/"),
    }


def is_configured() -> bool:
    return _cfg() is not None


def send(to: str, subject: str, body: str) -> bool:
    """Отправляет письмо. Возвращает True при успехе.

    Никогда не бросает исключение наружу: сбой почтового сервера не должен
    превращать успешно обработанную задачу в неуспешную.
    """
    cfg = _cfg()
    if not cfg or not to:
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = formataddr((cfg["from_name"], cfg["from"]))
        msg["To"] = to
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid()
        # Письмо служебное, а не рассылка: просим почтовики не отвечать
        # автоответами и не считать это маркетингом.
        msg["Auto-Submitted"] = "auto-generated"
        msg.set_content(body)

        ctx = ssl.create_default_context()
        if cfg["ssl"]:
            with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=ctx,
                                  timeout=30) as smtp:
                if cfg["user"]:
                    smtp.login(cfg["user"], cfg["password"])
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(cfg["host"], cfg["port"], timeout=30) as smtp:
                smtp.ehlo()
                try:
                    smtp.starttls(context=ctx)
                    smtp.ehlo()
                except smtplib.SMTPNotSupportedError:
                    # Сервер без TLS — обычно локальный релей на 127.0.0.1.
                    pass
                if cfg["user"]:
                    smtp.login(cfg["user"], cfg["password"])
                smtp.send_message(msg)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[почта] не удалось отправить на {_mask(to)}: "
              f"{type(e).__name__}: {e}")
        return False


def _mask(email: str) -> str:
    """Прячет адрес в логах: логи читают и хранят дольше, чем нужно."""
    if "@" not in email:
        return "???"
    name, dom = email.split("@", 1)
    return f"{name[:2]}***@{dom}"


def job_finished(email: str, title: str, status: str, clips: int,
                 error: str = "") -> bool:
    """Письмо о завершении обработки."""
    cfg = _cfg()
    site = (cfg or {}).get("site") or ""
    short = (title or "видео")[:80]

    if status == "done":
        subject = f"Готово: {clips} клипов"
        body = (f"Обработка завершена.\n\n"
                f"Видео: {short}\n"
                f"Готовых клипов: {clips}\n\n")
        # Срок хранения — важная часть письма: иначе человек вернётся через
        # месяц и не найдёт свои клипы.
        keep = os.environ.get("CLIPPER_KEEP_CLIPS_DAYS", "30")
        body += f"Клипы хранятся {keep} дней, успейте скачать.\n"
        # Срок берётся из той же переменной, что и очистка, — чтобы письмо
        # не обещало одно, а служба удаляла по другому расписанию.
    elif status == "error":
        subject = "Обработка не удалась"
        body = (f"К сожалению, обработать видео не получилось.\n\n"
                f"Видео: {short}\n"
                f"Причина: {error or 'неизвестна'}\n\n"
                f"Можно попробовать ещё раз.\n")
    else:
        return False

    if site:
        body += f"\nОткрыть: {site}\n"
    body += "\n—\nЭто автоматическое письмо, отвечать на него не нужно.\n"
    body += "Отключить уведомления можно в настройках профиля.\n"
    return send(email, subject, body)
