"""Emails a fussy verdict, with the episode's audio attached, through the homelab's shared Gmail login."""

import smtplib
from email.message import EmailMessage

from .config import SMTP_ALERT_TO, SMTP_HOST, SMTP_PASSWORD, SMTP_PORT, SMTP_USERNAME

ENABLED = bool(SMTP_ALERT_TO and SMTP_USERNAME and SMTP_PASSWORD)


def send(ep, verdict):
    msg = EmailMessage()
    msg["Subject"] = f"Dog is fussing: {verdict.summary}"
    msg["From"] = f"barker <{SMTP_USERNAME}>"
    msg["To"] = SMTP_ALERT_TO
    msg.set_content(
        f"{verdict.summary}\n\n"
        f"Started {ep.start:%a %H:%M:%S}, {ep.duration:.0f} seconds ({ep.tally()}). "
        f"Judged by {verdict.by}, {verdict.confidence} confidence. The clip is attached.\n\n"
        f"{ep.timeline()}\n"
    )
    msg.add_attachment(ep.wav(), maintype="audio", subtype="wav", filename=f"barker-{ep.start:%Y%m%d-%H%M%S}.wav")
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
        smtp.send_message(msg)
