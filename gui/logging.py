import datetime
from PyQt5 import QtWidgets, QtGui
from config import cfg

def append_log_with_limit(text_edit: QtWidgets.QTextEdit, text: str):
    cursor = text_edit.textCursor()
    cursor.movePosition(QtGui.QTextCursor.End)
    cursor.insertText(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {text}\n")
    doc = text_edit.document()
    if doc.blockCount() > cfg.max_log_lines:
        cursor.movePosition(QtGui.QTextCursor.Start)
        for _ in range(doc.blockCount() - cfg.max_log_lines):
            cursor.select(QtGui.QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
    text_edit.setTextCursor(cursor)
    text_edit.ensureCursorVisible()