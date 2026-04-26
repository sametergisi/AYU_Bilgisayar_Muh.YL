import socket
import threading
import mysql.connector

DB_HOST = "192.250.229.28"
DB_USER = "mikelsan_samet"
DB_PASS = "e3r4t5E3R4T5"
DB_NAME = "mikelsan_ayu"

HOST = "0.0.0.0"
PORT = 5000

START = "AE"
END = "AYU"


def get_db():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        ssl_disabled=True
    )


def parse_packet(packet):
    if not (packet.startswith(START) and packet.endswith(END)):
        return None

    body = packet[2:-3]
    parts = body.split(":")

    if len(parts) != 5:
        return None

    try:
        mic = int(parts[0])
        ldr = int(parts[1])
        mq4 = int(parts[2])
        hum = int(parts[3]) / 10.0
        temp = int(parts[4]) / 10.0

        return mic, ldr, mq4, hum, temp
    except:
        return None


def insert_data(cursor, db, mic, ldr, mq4, hum, temp):
    try:
        sql = """
        INSERT INTO alertrecord (gas, temp, hum, mic, ldr)
        VALUES (%s, %s, %s, %s, %s)
        """

        values = (mq4, temp, hum, mic, ldr)

        cursor.execute(sql, values)
        db.commit()

        print("veri eklendi")

    except Exception as e:
        print("veritabanı hatası:", e)
        db.rollback()


def handle_client(conn, addr):
    print("bağlandı:", addr)

    db = get_db()
    cursor = db.cursor()

    buffer = ""

    while True:
        data = conn.recv(1024)
        if not data:
            break
        buffer += data.decode(errors="ignore")

        # buffer içinde paket var mı diye sürekli bak
        while "AE" in buffer and "AYU" in buffer:

            # paketin başladığı yer
            start = buffer.find("AE")

            # paketin bittiği yer
            end = buffer.find("AYU")

            # paketi kes (başından sonuna kadar)
            packet = buffer[start:end + 3]

            # buffer’dan bu paketi sil
            buffer = buffer[end + 3:]

            print("gelen:", packet)

            # paketi çöz
            parsed = parse_packet(packet)

            if parsed is None:
                print("paket çözülemedi")
                continue

            mic, ldr, mq4, hum, temp = parsed

            print("mic:", mic, "ldr:", ldr, "gaz:", mq4,
                  "nem:", hum, "sıcaklık:", temp)

            insert_data(cursor, db, mic, ldr, mq4, hum, temp)

    cursor.close()
    db.close()
    conn.close()


def start_server():
    print("server çalışıyor:", HOST, PORT)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()

        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(conn, addr)).start()


if __name__ == "__main__":
    start_server()