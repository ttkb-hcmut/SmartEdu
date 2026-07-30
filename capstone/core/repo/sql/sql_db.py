import hashlib
from typing import Dict, Any, Optional

import pymysql
from pymysql.cursors import DictCursor

from core.config import MySQL_conf


class SQL_DB:
    def __init__(self, config: MySQL_conf = MySQL_conf()):
        self.config = config
        self._init_db()

    def _connect(self):
        return pymysql.connect(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.db_name,
            cursorclass=DictCursor,
            autocommit=True,
        )

    def _init_db(self):
        ## database itself must pre-exist (CREATE DATABASE), driver only connects to it
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS students (
                        id VARCHAR(255) PRIMARY KEY,
                        username VARCHAR(255) UNIQUE,
                        name VARCHAR(255),
                        email VARCHAR(255),
                        password VARCHAR(255),
                        is_admin TINYINT DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

    def get_student_by_id(self, student_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM students WHERE id = %s", (student_id,))
                return cursor.fetchone()

    def get_student_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM students WHERE username = %s", (username,))
                return cursor.fetchone()

    def create_student(self, student_id: str, username: str = "", name: str = "", email: str = "", password: str = "", is_admin: bool = False):
        # hash password simply (in production, use bcrypt)
        hashed_pw = hashlib.sha256(password.encode()).hexdigest() if password else ""

        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT IGNORE INTO students (id, username, name, email, password, is_admin) VALUES (%s, %s, %s, %s, %s, %s)",
                    (student_id, username, name, email, hashed_pw, 1 if is_admin else 0)
                )

    def delete_student(self, student_id: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM students WHERE id = %s", (student_id,))

    def is_admin(self, student_id: str) -> bool:
        # read admin flag for a student
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT is_admin FROM students WHERE id = %s", (student_id,))
                row = cursor.fetchone()
                return bool(row["is_admin"]) if row else False

    def authenticate(self, username: str, password: str) -> bool:
        hashed_pw = hashlib.sha256(password.encode()).hexdigest()
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id FROM students WHERE username = %s AND password = %s", (username, hashed_pw))
                return cursor.fetchone() is not None
