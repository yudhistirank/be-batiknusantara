import pytz
from datetime import datetime

def format_timestamp_indonesia(dt_utc):
    bulan = [
        "Januari", "Februari", "Maret", "April", "Mei", "Juni",
        "Juli", "Agustus", "September", "Oktober", "November", "Desember"
    ]
    wib = pytz.timezone("Asia/Jakarta")
    dt_wib = dt_utc.replace(tzinfo=pytz.utc).astimezone(wib)
    return f"{dt_wib.day:02d} {bulan[dt_wib.month - 1]} {dt_wib.year} {dt_wib.strftime('%H:%M')} WIB"
